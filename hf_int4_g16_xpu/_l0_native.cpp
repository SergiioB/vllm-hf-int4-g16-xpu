// Zero-copy SYCL/Level-Zero interop launcher for hardened Xe2 HF-INT4-G16 kernels.
//
// The zebin is imported once into the current torch XPU device's SYCL context
// with zeModuleCreate + sycl::make_kernel_bundle/make_kernel.  Every launch is
// submitted through the current in-order SYCL queue (q.submit), so stream
// ordering and torch.xpu graph capture are owned by the runtime rather than by
// a raw command-list append.  Tensor data_ptr() values are torch USM pointers;
// there are no host copies or per-call zero-fill bias allocations.
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#define ZE_CHECK(x) do { \
  ze_result_t _r = (x); \
  TORCH_CHECK(_r == ZE_RESULT_SUCCESS, "Level-Zero error ", (int)_r, " at ", #x); \
} while (0)

namespace {

constexpr auto L0 = sycl::backend::ext_oneapi_level_zero;
using Own = sycl::ext::oneapi::level_zero::ownership;

struct DevState {
  ze_module_handle_t zmod = nullptr;
  std::shared_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>> bundle;
  std::unordered_map<std::string, sycl::kernel> kernels;
  std::vector<char> module_bytes;
};

std::mutex g_mu;
std::unordered_map<std::string, std::unique_ptr<DevState>> g_states;
std::unordered_map<std::string, torch::Tensor> g_ci32_dummy;

std::vector<char> readfile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  TORCH_CHECK(f.good(), "cannot open Level-Zero kernel binary: ", path);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

sycl::queue& current_queue(int device_index) {
  auto stream = c10::xpu::getCurrentXPUStream(device_index);
  sycl::queue& q = stream.queue();
  TORCH_CHECK(q.get_backend() == L0, "torch XPU stream is not Level-Zero SYCL");
  TORCH_CHECK(q.is_in_order(), "torch XPU stream is not an in-order SYCL queue");
  return q;
}

std::string hex_ptr(const void* p) {
  std::ostringstream oss;
  oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(p);
  return oss.str();
}

inline uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

std::string sha256_hex(const std::vector<char>& bytes) {
  static constexpr uint32_t round_k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  };
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto process = [&](const uint8_t* chunk) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
      w[i] = (uint32_t(chunk[4 * i]) << 24) | (uint32_t(chunk[4 * i + 1]) << 16) |
             (uint32_t(chunk[4 * i + 2]) << 8) | uint32_t(chunk[4 * i + 3]);
    }
    for (int i = 16; i < 64; i++) {
      const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = hh + s1 + ch + round_k[i] + w[i];
      const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  };

  const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
  const size_t len = bytes.size();
  size_t off = 0;
  while (off + 64 <= len) {
    process(data + off);
    off += 64;
  }
  uint8_t tail[128] = {};
  const size_t rem = len - off;
  if (rem != 0) std::memcpy(tail, data + off, rem);
  tail[rem] = 0x80;
  const size_t blocks = rem >= 56 ? 2 : 1;
  const uint64_t bits = uint64_t(len) * 8;
  for (int i = 0; i < 8; i++) tail[blocks * 64 - 1 - i] = uint8_t(bits >> (8 * i));
  for (size_t i = 0; i < blocks; i++) process(tail + 64 * i);

  std::ostringstream out;
  out << std::hex;
  for (uint32_t v : h) {
    out.width(8);
    out.fill('0');
    out << v;
  }
  return out.str();
}

const std::vector<std::pair<const char*, uint32_t>>& required_kernel_abis() {
  static const std::vector<std::pair<const char*, uint32_t>> specs = {
      {"fused_fwht128_actq", 5},
      {"fused_fwht64_actq", 5},
      {"hf_dpas_m8_sk4", 12},
      {"hf_dpas_m16_sk4", 12},
      {"hf_dpas_m32_sk4", 12},
      {"hf_dpas_m64_n16_wg16", 12},
      {"hf_dpas_m128_n16_wg8", 12},
      {"hf_dpas_m128_n16", 12},
      {"hf_decode_fused_m1", 9},
      {"hf_decode_fused_m2", 9},
      {"hf_decode_fused_m4", 9},
      {"hf_decode_fused_had64_m1", 9},
      {"hf_decode_fused_had64_m2", 9},
      {"hf_decode_fused_had64_m4", 9},
      {"hf_moe_pair_had128_bf16", 12},
      {"hf_moe_pair_had128_f32", 12},
      {"hf_moe_pair_had64_bf16", 12},
      {"hf_moe_pair_had64_f32", 12},
      {"hf_moe_bucket_zero", 2},
      {"hf_moe_bucket_count", 4},
      {"hf_moe_bucket_scan", 5},
      {"hf_moe_bucket_scatter", 5},
      {"hf_moe_rotate_had128", 4},
      {"hf_moe_rotate_had64", 4},
      {"hf_moe_tile_had128_bf16", 13},
      {"hf_moe_tile_had128_f32", 13},
      {"hf_moe_tile_had64_bf16", 13},
      {"hf_moe_tile_had64_f32", 13},
  };
  return specs;
}

DevState& state_for(int device_index, const std::string& binpath_or_digest) {
  // State is keyed by the native Level-Zero context and device plus the
  // SHA-256 of the exact zebin bytes passed to zeModuleCreate.  Callers that
  // admitted the binary once via admit_module() pass "sha256:<hex>" directly,
  // avoiding any per-launch file read or re-hashing.
  sycl::queue& q = current_queue(device_index);
  const auto ctx = q.get_context();
  const auto dev = q.get_device();
  const auto ze_ctx = sycl::get_native<L0>(ctx);
  const auto ze_dev = sycl::get_native<L0>(dev);
  const std::string prefix = hex_ptr(reinterpret_cast<const void*>(ze_ctx)) + ":" +
      hex_ptr(reinterpret_cast<const void*>(ze_dev)) + ":";

  if (binpath_or_digest.rfind("sha256:", 0) == 0) {
    const std::string key = prefix + binpath_or_digest;
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_states.find(key);
    TORCH_CHECK(it != g_states.end(), "unadmitted native module digest: ", binpath_or_digest);
    return *it->second;
  }

  auto bytes = readfile(binpath_or_digest);
  const std::string digest = sha256_hex(bytes);
  const std::string key = prefix + "sha256:" + digest;

  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_states.find(key);
  if (it != g_states.end()) return *it->second;

  ze_module_desc_t md{};
  md.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  md.format = ZE_MODULE_FORMAT_NATIVE;
  md.inputSize = bytes.size();
  md.pInputModule = reinterpret_cast<const uint8_t*>(bytes.data());
  ze_module_handle_t zmod = nullptr;
  ZE_CHECK(zeModuleCreate(ze_ctx, ze_dev, &md, &zmod, nullptr));

  auto st = std::make_unique<DevState>();
  st->module_bytes = std::move(bytes);
  st->zmod = zmod;
  // Ownership transfers intentionally.  The state is deliberately leaked for
  // process lifetime so graph executables can never observe a premature zebin
  // or kernel destruction.
  st->bundle = std::make_shared<sycl::kernel_bundle<sycl::bundle_state::executable>>(
      sycl::make_kernel_bundle<L0, sycl::bundle_state::executable>({zmod, Own::transfer}, ctx));

  // Pre-create every dispatchable kernel while the outer state mutex is held.
  // After publication the kernels map is immutable, so dispatch is lock-free
  // and cannot race a lazy emplace from another Python/threadpool thread.
  for (const auto& [name, expected_args] : required_kernel_abis()) {
    ze_kernel_desc_t kd{};
    kd.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kd.pKernelName = name;
    ze_kernel_handle_t zk = nullptr;
    ZE_CHECK(zeKernelCreate(st->zmod, &kd, &zk));
    auto sk = sycl::make_kernel<L0>({*st->bundle, zk, Own::transfer}, ctx);
    const auto nargs = sk.get_info<sycl::info::kernel::num_args>();
    TORCH_CHECK(nargs == expected_args, "imported kernel ", name, " arity ", nargs,
                " != launcher ABI ", expected_args, " (stale zebin?)");
    st->kernels.emplace(name, std::move(sk));
  }

  auto [it2, inserted] = g_states.emplace(key, std::move(st));
  TORCH_CHECK(inserted, "duplicate native module state key");
  return *it2->second;
}

std::string admit_module(int device_index, const std::string& binpath) {
  const c10::DeviceGuard guard(c10::Device(c10::DeviceType::XPU, device_index));
  auto& st = state_for(device_index, binpath);
  return "sha256:" + sha256_hex(st.module_bytes);
}

const sycl::kernel& kernel_obj(DevState& st, const sycl::context& ctx,
                               const std::string& name, uint32_t expected_args) {
  (void)ctx;
  auto it = st.kernels.find(name);
  TORCH_CHECK(it != st.kernels.end(), "native kernel ", name, " was not pre-created");
  const auto nargs = it->second.get_info<sycl::info::kernel::num_args>();
  TORCH_CHECK(nargs == expected_args, "imported kernel ", name, " arity ", nargs,
              " != launcher ABI ", expected_args, " (stale zebin?)");
  return it->second;
}

inline void set_ptr_arg(sycl::handler& h, int index, const void* p) {
  void* v = const_cast<void*>(p);
  h.set_arg(index, v);
}

inline sycl::nd_range<3> nd_from_opencl_groups(uint32_t gx, uint32_t gy, uint32_t gz,
                                                uint32_t lx, uint32_t ly, uint32_t lz) {
  // Astra/Triton mapping: SYCL ranges are (z,y,x) for OpenCL (x,y,z), and are
  // work-item counts rather than work-group counts.
  return sycl::nd_range<3>(
      sycl::range<3>((size_t)gz * lz, (size_t)gy * ly, (size_t)gx * lx),
      sycl::range<3>((size_t)lz, (size_t)ly, (size_t)lx));
}

torch::Tensor ci32_dummy_for(const torch::TensorOptions& options, int device_index) {
  // The dummy is process-lifetime scratch, but it is still native-allocation
  // state.  Key it by the same native context/device identity as the module
  // cache rather than by a bare torch device index.
  sycl::queue& q = current_queue(device_index);
  const auto ze_ctx = sycl::get_native<L0>(q.get_context());
  const auto ze_dev = sycl::get_native<L0>(q.get_device());
  const std::string key = hex_ptr(reinterpret_cast<const void*>(ze_ctx)) + ":" +
      hex_ptr(reinterpret_cast<const void*>(ze_dev));
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_ci32_dummy.find(key);
  if (it == g_ci32_dummy.end()) {
    it = g_ci32_dummy.emplace(key, torch::empty({1}, options)).first;
  }
  return it->second;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
}

void check_aligned(const char* name, const void* p) {
  TORCH_CHECK(aligned16(p), name, " base pointer is not 16-byte aligned");
}

// Mirrors native_l0.block2d_surface_error: cl_intel_subgroup_2d_block_io 6.13.X.6.
constexpr int64_t kBlock2dMaxExtent = int64_t(1) << 24;
constexpr int64_t kInt32Max = 2147483647;
// Token rows per grouped MoE tile (native_l0.MOE_TILE_ROWS).
constexpr int64_t kMoeTileRows = 16;

void check_int32(const char* name, int64_t value) {
  TORCH_CHECK(value >= 0 && value <= kInt32Max, name, "=", value, " does not fit an int32 kernel argument");
}

void check_block2d_surface(const char* name, const void* base, int64_t width, int64_t height, int64_t pitch) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(base) % 64 == 0, name,
              " effective base address is not 64-byte aligned");
  TORCH_CHECK(width >= 64 && width <= kBlock2dMaxExtent && width % 4 == 0, name,
              " 2D block-io surface width ", width, " B must be in [64, 2^24] and a multiple of 4");
  TORCH_CHECK(height > 0 && height <= kBlock2dMaxExtent, name,
              " 2D block-io surface height ", height, " must be in [1, 2^24]");
  TORCH_CHECK(pitch >= width && pitch % 16 == 0 && pitch <= kBlock2dMaxExtent, name,
              " 2D block-io pitch ", pitch, " B must be >= width, a multiple of 16 and <= 2^24");
}

// The kernels read 16 output columns per sub-group, so N%16==0 keeps every
// tile inside the surface.  `rows` is K/8.
void check_packed_weight_surface(const void* base, int64_t rows, int64_t n) {
  TORCH_CHECK(n >= 16 && n % 16 == 0,
              "native requires N>=16 and N%16==0 (surface width 4N a multiple of 64 B), got N ", n);
  TORCH_CHECK(n <= kBlock2dMaxExtent / 4, "q_pack surface width 4N exceeds 2^24 B for N ", n);
  check_block2d_surface("q_pack", base, 4 * n, rows, 4 * n);
}

void check_same_device(const torch::Tensor& ref, const torch::Tensor& other, const char* name) {
  TORCH_CHECK(other.device() == ref.device(), name, " is on ", other.device(),
              " but input is on ", ref.device(), "; all native operands must share one XPU device");
}

void record_input_stream(const torch::Tensor& t, int device_index) {
  if (!t.defined() || !t.is_xpu() || t.numel() == 0) return;
  const auto& dp = t.storage().data_ptr();
  if (dp.get() != nullptr) {
    c10::xpu::XPUCachingAllocator::recordStream(dp, c10::xpu::getCurrentXPUStream(device_index));
  }
}

void record_optional_stream(const std::optional<torch::Tensor>& t, int device_index) {
  if (t.has_value() && t->defined()) record_input_stream(*t, device_index);
}

void check_common(const torch::Tensor& q, const torch::Tensor& u, const torch::Tensor& d) {
  TORCH_CHECK(q.is_xpu() && u.is_xpu() && d.is_xpu(), "packed tensors must be XPU tensors");
  TORCH_CHECK(q.device() == u.device() && q.device() == d.device(),
              "q_pack/u_pack/d_row must be on the same XPU device");
  TORCH_CHECK(q.is_contiguous() && u.is_contiguous() && d.is_contiguous(), "packed tensors must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::kInt, "q_pack must be int32 bit container");
  TORCH_CHECK(u.scalar_type() == at::kShort, "u_pack must be int16 bit container");
  TORCH_CHECK(d.scalar_type() == at::kFloat, "d_row must be fp32");
  TORCH_CHECK(q.dim() == 2 && u.dim() == 2, "q_pack/u_pack must be rank-2");
  TORCH_CHECK(d.dim() == 1, "d_row must be rank-1");
  TORCH_CHECK(q.size(0) * 8 == u.size(0) * 64, "q/u K mismatch");
  TORCH_CHECK(q.size(1) == u.size(1) && q.size(1) == d.numel(), "q/u/D N mismatch");
  TORCH_CHECK(q.size(0) * 8 <= 17408, "K exceeds INT32-accumulation bound 17408");
  check_packed_weight_surface(q.data_ptr(), q.size(0), q.size(1));
  check_aligned("q_pack", q.data_ptr());
  check_aligned("u_pack", u.data_ptr());
  check_aligned("d_row", d.data_ptr());
}

const float* optional_bias_ptr(const std::optional<torch::Tensor>& bias, int64_t n,
                               const torch::Device& device) {
  if (!bias.has_value() || !bias->defined()) return nullptr;
  TORCH_CHECK(bias->is_xpu() && bias->scalar_type() == at::kFloat && bias->is_contiguous(),
              "bias_corr must be a contiguous fp32 XPU tensor");
  TORCH_CHECK(bias->dim() == 1, "bias_corr must be rank-1");
  TORCH_CHECK(bias->device() == device, "bias_corr must be on the same XPU device as the input");
  TORCH_CHECK(bias->numel() == n, "bias_corr length ", bias->numel(), " != N ", n);
  check_aligned("bias_corr", bias->data_ptr());
  return reinterpret_cast<const float*>(bias->data_ptr());
}

// The Hadamard block is a pure function of the global K (format.had_block_for_k).
int had_block_of(int64_t k) {
  TORCH_CHECK(k > 0 && k % 64 == 0, "K must be a multiple of 128 (Had128) or 64 (Had64), got ", k);
  return k % 128 == 0 ? 128 : 64;
}

void launch_decode_piece(const sycl::kernel& k, sycl::queue& q,
                         const char* xptr, const char* signs, const char* qp,
                         const char* up, const char* d, const float* bias,
                         char* outptr, int n, int kdim) {
  q.submit([&](sycl::handler& h) {
    set_ptr_arg(h, 0, xptr);
    set_ptr_arg(h, 1, signs);
    set_ptr_arg(h, 2, qp);
    set_ptr_arg(h, 3, up);
    set_ptr_arg(h, 4, d);
    set_ptr_arg(h, 5, bias);
    set_ptr_arg(h, 6, outptr);
    h.set_arg(7, n);
    h.set_arg(8, kdim);
    h.parallel_for(nd_from_opencl_groups((uint32_t)((n + 31) / 32), 1, 1, 128, 1, 1), k);
  });
}

} // namespace

void fwht_actq(torch::Tensor x, torch::Tensor signs, torch::Tensor xi8, torch::Tensor sx,
               const std::string& binpath) {
  const c10::DeviceGuard guard(x.device());
  const int device_index = x.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& q = current_queue(device_index);
  TORCH_CHECK(x.dim() == 2, "native fwht_actq input must be rank-2");
  TORCH_CHECK(x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "native fwht_actq requires contiguous bf16 XPU input");
  TORCH_CHECK(signs.dim() == 1 && signs.is_xpu() && signs.scalar_type() == at::kChar && signs.is_contiguous(),
              "signs must be contiguous rank-1 int8 XPU");
  TORCH_CHECK(xi8.dim() == 2 && xi8.is_xpu() && xi8.scalar_type() == at::kChar && xi8.is_contiguous(),
              "xi8 output must be contiguous rank-2 int8 XPU");
  TORCH_CHECK(sx.dim() == 1 && sx.is_xpu() && sx.scalar_type() == at::kFloat && sx.is_contiguous(),
              "sx output must be contiguous rank-1 fp32 XPU");
  check_same_device(x, signs, "signs");
  check_same_device(x, xi8, "xi8");
  check_same_device(x, sx, "sx");
  check_int32("M", x.size(0));
  check_int32("K", x.size(1));
  int M = (int)x.size(0), K = (int)x.size(1);
  TORCH_CHECK(K % 64 == 0 && K <= 17408, "native fwht_actq requires K%64==0 and K<=17408");
  TORCH_CHECK(signs.numel() == K, "signs length ", signs.numel(), " != K ", K);
  TORCH_CHECK(xi8.numel() == x.numel(), "xi8 output numel mismatch");
  TORCH_CHECK(sx.numel() == M, "sx length ", sx.numel(), " != M ", M);
  check_aligned("x", x.data_ptr());
  check_aligned("signs", signs.data_ptr());
  check_aligned("xi8", xi8.data_ptr());
  check_aligned("sx", sx.data_ptr());
  if (M == 0) return;
  record_input_stream(x, device_index);
  record_input_stream(signs, device_index);
  record_input_stream(xi8, device_index);
  record_input_stream(sx, device_index);
  const auto& k = kernel_obj(st, q.get_context(),
                             had_block_of(K) == 128 ? "fused_fwht128_actq" : "fused_fwht64_actq", 5);
  const void* xp = x.data_ptr();
  const void* sp = signs.data_ptr();
  const void* xi = xi8.data_ptr();
  const void* sxp = sx.data_ptr();
  q.submit([&](sycl::handler& h) {
    set_ptr_arg(h, 0, xp);
    set_ptr_arg(h, 1, sp);
    set_ptr_arg(h, 2, xi);
    set_ptr_arg(h, 3, sxp);
    h.set_arg(4, K);
    h.parallel_for(nd_from_opencl_groups((uint32_t)M, 1, 1, 256, 1, 1), k);
  });
}

torch::Tensor dpas_w4a8(torch::Tensor xi8, torch::Tensor sx, torch::Tensor q, torch::Tensor u,
                        torch::Tensor d, std::optional<torch::Tensor> bias, bool write_i32,
                        const std::string& binpath) {
  const c10::DeviceGuard guard(xi8.device());
  const int device_index = xi8.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& que = current_queue(device_index);
  check_common(q, u, d);
  TORCH_CHECK(xi8.dim() == 2, "native dpas xi8 input must be rank-2");
  TORCH_CHECK(xi8.is_xpu() && xi8.scalar_type() == at::kChar && xi8.is_contiguous());
  TORCH_CHECK(sx.dim() == 1 && sx.is_xpu() && sx.scalar_type() == at::kFloat && sx.is_contiguous());
  check_same_device(xi8, q, "q_pack");
  check_same_device(xi8, sx, "sx");
  check_int32("M", xi8.size(0));
  check_int32("K", xi8.size(1));
  check_int32("N", q.size(1));
  int M = (int)xi8.size(0), K = (int)xi8.size(1), N = (int)q.size(1);
  TORCH_CHECK(K == q.size(0) * 8, "xi8/q K mismatch");
  TORCH_CHECK(K % 64 == 0 && K <= 17408, "native dpas requires K%64==0 and K<=17408");
  TORCH_CHECK(sx.numel() == M, "sx length ", sx.numel(), " != M ", M);
  check_aligned("xi8", xi8.data_ptr());
  check_aligned("sx", sx.data_ptr());
  auto out = torch::empty({M, N}, xi8.options().dtype(at::kBFloat16));
  torch::Tensor ci32;
  if (write_i32) {
    ci32 = torch::empty({M, N}, xi8.options().dtype(at::kInt));
  } else {
    ci32 = ci32_dummy_for(xi8.options().dtype(at::kInt), device_index);
  }
  if (M == 0) {
    return write_i32 ? ci32 : out;
  }
  check_block2d_surface("int8 activation", xi8.data_ptr(), K, M, K);
  check_block2d_surface("bf16 output", out.data_ptr(), int64_t(2) * N, M, int64_t(2) * N);

  std::string name; int sg_m, wg_n;
  bool small_sk4 = M <= 32;
  if (M <= 8)       { name = "hf_dpas_m8_sk4";        sg_m = 8;   wg_n = 2; }
  else if (M <= 16) { name = "hf_dpas_m16_sk4";       sg_m = 16;  wg_n = 2; }
  else if (M <= 32) { name = "hf_dpas_m32_sk4";       sg_m = 32;  wg_n = 2; }
  else if (M <= 64) { name = "hf_dpas_m64_n16_wg16";  sg_m = 64;  wg_n = 16; }
  else if (N <= 5120 && M <= 1024) {
    // Narrow-output tiles at partial-wave counts benefit from twice as many
    // column work-groups; wide/large tiles use the exp8a PANEL_N=8 baseline.
    name = "hf_dpas_m128_n16_wg8"; sg_m = 128; wg_n = 8;
  } else {
    name = "hf_dpas_m128_n16"; sg_m = 128; wg_n = 16;
  }
  const auto& k = kernel_obj(st, que.get_context(), name, 12);
  const int dump = write_i32 ? 1 : 0;
  const float* biasp = optional_bias_ptr(bias, N, xi8.device());
  const void* ap = xi8.data_ptr();
  const void* qp = q.data_ptr();
  const void* up = u.data_ptr();
  const void* sxp = sx.data_ptr();
  const void* dp = d.data_ptr();
  const void* op = out.data_ptr();
  const void* cp = ci32.data_ptr();
  uint32_t lx = small_sk4 ? 128u : (uint32_t)(wg_n * 16);
  uint32_t gx = (uint32_t)((N + wg_n * 16 - 1) / (wg_n * 16));
  uint32_t gy = (uint32_t)((M + sg_m - 1) / sg_m);
  record_input_stream(xi8, device_index);
  record_input_stream(sx, device_index);
  record_input_stream(q, device_index);
  record_input_stream(u, device_index);
  record_input_stream(d, device_index);
  record_optional_stream(bias, device_index);
  que.submit([&](sycl::handler& h) {
    set_ptr_arg(h, 0, ap);
    set_ptr_arg(h, 1, qp);
    set_ptr_arg(h, 2, up);
    set_ptr_arg(h, 3, sxp);
    set_ptr_arg(h, 4, dp);
    set_ptr_arg(h, 5, biasp);
    set_ptr_arg(h, 6, op);
    set_ptr_arg(h, 7, cp);
    h.set_arg(8, M); h.set_arg(9, N); h.set_arg(10, K); h.set_arg(11, dump);
    h.parallel_for(nd_from_opencl_groups(gx, gy, 1, lx, 1, 1), k);
  });
  return write_i32 ? ci32 : out;
}

torch::Tensor fwht_dpas_w4a8(torch::Tensor x, torch::Tensor signs, torch::Tensor q, torch::Tensor u,
                             torch::Tensor d, std::optional<torch::Tensor> bias, const std::string& binpath) {
  TORCH_CHECK(x.dim() == 2, "native fwht_dpas input must be rank-2");
  check_int32("M", x.size(0));
  check_int32("K", x.size(1));
  int M = (int)x.size(0), K = (int)x.size(1);
  const c10::DeviceGuard guard(x.device());
  auto xi8 = torch::empty({M, K}, x.options().dtype(at::kChar));
  auto sx = torch::empty({M}, x.options().dtype(at::kFloat));
  fwht_actq(x, signs, xi8, sx, binpath);
  return dpas_w4a8(xi8, sx, q, u, d, bias, false, binpath);
}

torch::Tensor decode_fused(torch::Tensor x, torch::Tensor signs, torch::Tensor q, torch::Tensor u,
                           torch::Tensor d, std::optional<torch::Tensor> bias, const std::string& binpath) {
  const c10::DeviceGuard guard(x.device());
  const int device_index = x.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& que = current_queue(device_index);
  check_common(q, u, d);
  TORCH_CHECK(x.dim() == 2, "native decode_fused input must be rank-2");
  TORCH_CHECK(x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "native decode_fused requires contiguous bf16 XPU input");
  TORCH_CHECK(signs.dim() == 1 && signs.is_xpu() && signs.scalar_type() == at::kChar && signs.is_contiguous(),
              "signs must be contiguous rank-1 int8 XPU");
  check_same_device(x, q, "q_pack");
  check_same_device(x, signs, "signs");
  check_int32("M", x.size(0));
  check_int32("K", x.size(1));
  check_int32("N", q.size(1));
  int M = (int)x.size(0), K = (int)x.size(1), N = (int)q.size(1);
  TORCH_CHECK(M >= 1 && M <= 4, "native fused decode supports M in {1,2,3,4}");
  TORCH_CHECK(K == q.size(0) * 8, "x/q K mismatch");
  TORCH_CHECK(K % 64 == 0 && K <= 17408, "native decode requires K%64==0 and K<=17408");
  TORCH_CHECK(signs.numel() == K, "signs length ", signs.numel(), " != K ", K);
  check_aligned("x", x.data_ptr());
  check_aligned("signs", signs.data_ptr());
  auto out = torch::empty({M, N}, x.options().dtype(at::kBFloat16));
  const float* biasp = optional_bias_ptr(bias, N, x.device());
  const char* xp = reinterpret_cast<const char*>(x.data_ptr());
  const char* sp = reinterpret_cast<const char*>(signs.data_ptr());
  const char* qp = reinterpret_cast<const char*>(q.data_ptr());
  const char* up = reinterpret_cast<const char*>(u.data_ptr());
  const char* dp = reinterpret_cast<const char*>(d.data_ptr());
  char* op = reinterpret_cast<char*>(out.data_ptr());
  record_input_stream(x, device_index);
  record_input_stream(signs, device_index);
  record_input_stream(q, device_index);
  record_input_stream(u, device_index);
  record_input_stream(d, device_index);
  record_optional_stream(bias, device_index);

  const std::string stem = had_block_of(K) == 128 ? "hf_decode_fused_m" : "hf_decode_fused_had64_m";
  if (M == 3) {
    // Never launch the fixed four-row kernel on a three-row allocation.  Two
    // ordered pieces keep both scratch/output bounds exact and are graph-recordable.
    const auto& k2 = kernel_obj(st, que.get_context(), stem + "2", 9);
    const auto& k1 = kernel_obj(st, que.get_context(), stem + "1", 9);
    launch_decode_piece(k2, que, xp, sp, qp, up, dp, biasp, op, N, K);
    launch_decode_piece(k1, que, xp + (size_t)2 * K * 2, sp, qp, up, dp, biasp,
                        op + (size_t)2 * N * 2, N, K);
    return out;
  }

  const auto& k = kernel_obj(st, que.get_context(), stem + std::to_string(M), 9);
  launch_decode_piece(k, que, xp, sp, qp, up, dp, biasp, op, N, K);
  return out;
}

// Grouped routed-expert W4A16: one output row per (token, slot) pair.  Pair p
// reads x row p / in_div and the packed weights of expert ids[p].  Routing is
// read on the device: the launch shape depends only on (pairs, N).
torch::Tensor moe_pair(torch::Tensor x, torch::Tensor signs, torch::Tensor q, torch::Tensor u, torch::Tensor d,
                       std::optional<torch::Tensor> bias, torch::Tensor ids, int64_t in_div, bool out_f32,
                       const std::string& binpath) {
  const c10::DeviceGuard guard(x.device());
  const int device_index = x.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& que = current_queue(device_index);
  TORCH_CHECK(x.dim() == 2 && x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "native moe_pair requires a contiguous rank-2 bf16 XPU input");
  TORCH_CHECK(signs.dim() == 1 && signs.is_xpu() && signs.scalar_type() == at::kChar && signs.is_contiguous(),
              "signs must be contiguous rank-1 int8 XPU");
  TORCH_CHECK(q.dim() == 3 && u.dim() == 3 && d.dim() == 2, "moe q/u/d must be rank 3/3/2");
  TORCH_CHECK(q.is_xpu() && u.is_xpu() && d.is_xpu(), "packed tensors must be XPU tensors");
  TORCH_CHECK(q.is_contiguous() && u.is_contiguous() && d.is_contiguous(), "packed tensors must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::kInt && u.scalar_type() == at::kShort && d.scalar_type() == at::kFloat,
              "moe packed dtypes must be int32/int16/fp32");
  TORCH_CHECK(ids.dim() == 1 && ids.is_xpu() && ids.scalar_type() == at::kInt && ids.is_contiguous(),
              "expert ids must be a contiguous rank-1 int32 XPU tensor");
  check_same_device(x, signs, "signs");
  check_same_device(x, q, "q_pack");
  check_same_device(x, u, "u_pack");
  check_same_device(x, d, "d_row");
  check_same_device(x, ids, "expert_ids");
  const int64_t K = x.size(1), E = d.size(0), N = d.size(1), P = ids.size(0);
  const int block = had_block_of(K);
  TORCH_CHECK(signs.numel() == K, "signs length ", signs.numel(), " != K ", K);
  TORCH_CHECK(E > 0, "empty expert table (E == 0)");
  TORCH_CHECK(q.size(0) == E && q.size(1) == K / 8 && q.size(2) == N, "moe q_pack shape mismatch");
  TORCH_CHECK(u.size(0) == E && u.size(1) == K / 64 && u.size(2) == N, "moe u_pack shape mismatch");
  TORCH_CHECK(in_div > 0 && P == x.size(0) * in_div, "pairs ", P, " != rows ", x.size(0), " * in_div ", in_div);
  check_int32("N", N);
  check_int32("K", K);
  check_int32("E", E);
  check_int32("in_div", in_div);
  check_int32("pairs", P);
  check_packed_weight_surface(q.data_ptr(), K / 8, N);
  TORCH_CHECK((K / 8) * N * 4 % 64 == 0, "per-expert q_pack stride is not a multiple of 64 B");
  check_aligned("x", x.data_ptr());
  check_aligned("signs", signs.data_ptr());
  check_aligned("q_pack", q.data_ptr());
  check_aligned("u_pack", u.data_ptr());
  check_aligned("d_row", d.data_ptr());
  check_aligned("expert_ids", ids.data_ptr());
  const float* biasp = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_xpu() && bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
                    bias->dim() == 2 && bias->size(0) == E && bias->size(1) == N,
                "moe bias_corr must be a contiguous fp32 XPU tensor [E,N]");
    TORCH_CHECK(bias->device() == x.device(), "bias_corr must be on the same XPU device as the input");
    check_aligned("bias_corr", bias->data_ptr());
    biasp = reinterpret_cast<const float*>(bias->data_ptr());
  }
  auto out = torch::empty({P, N}, x.options().dtype(out_f32 ? at::kFloat : at::kBFloat16));
  if (P == 0) return out;
  const std::string name = std::string("hf_moe_pair_had") + std::to_string(block) + (out_f32 ? "_f32" : "_bf16");
  const auto& k = kernel_obj(st, que.get_context(), name, 12);
  record_input_stream(x, device_index);
  record_input_stream(signs, device_index);
  record_input_stream(q, device_index);
  record_input_stream(u, device_index);
  record_input_stream(d, device_index);
  record_input_stream(ids, device_index);
  record_optional_stream(bias, device_index);
  const void* xp = x.data_ptr();
  const void* sp = signs.data_ptr();
  const void* qp = q.data_ptr();
  const void* up = u.data_ptr();
  const void* dp = d.data_ptr();
  const void* ip = ids.data_ptr();
  const void* op = out.data_ptr();
  const int n32 = (int)N, k32 = (int)K, e32 = (int)E, div32 = (int)in_div;
  const uint32_t gx = (uint32_t)((N + 31) / 32), gy = (uint32_t)P;
  que.submit([&](sycl::handler& h) {
    set_ptr_arg(h, 0, xp);
    set_ptr_arg(h, 1, sp);
    set_ptr_arg(h, 2, qp);
    set_ptr_arg(h, 3, up);
    set_ptr_arg(h, 4, dp);
    set_ptr_arg(h, 5, biasp);
    set_ptr_arg(h, 6, ip);
    set_ptr_arg(h, 7, op);
    h.set_arg(8, n32); h.set_arg(9, k32); h.set_arg(10, e32); h.set_arg(11, div32);
    h.parallel_for(nd_from_opencl_groups(gx, gy, 1, 128, 1, 1), k);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Grouped-by-expert prefill.  All routing work happens on the device, in
// launches whose shapes depend only on (pairs, E, N, K), so a captured graph
// follows new routes on replay.  Scratch is allocated per call on the current
// stream (and from the capture pool while a graph records).
// ---------------------------------------------------------------------------
template <typename SetArgs>
void submit_nd(sycl::queue& que, const sycl::kernel& k, size_t gx, size_t gy, size_t lx, SetArgs&& set_args) {
  que.submit([&](sycl::handler& h) {
    set_args(h);
    h.parallel_for(nd_from_opencl_groups((uint32_t)gx, (uint32_t)gy, 1, (uint32_t)lx, 1, 1), k);
  });
}

int64_t moe_max_tiles(int64_t pairs, int64_t buckets) {
  // sum over buckets of ceil(count / rows) <= pairs / rows + (non-empty buckets)
  return std::min(pairs, pairs / kMoeTileRows + std::min(buckets, pairs));
}

// The tile kernel uses the bucket buffers as activation, expert and output
// addresses and does not bounds-check them.  They are therefore only reachable
// through this capsule: Python can neither read, build nor replace them, and
// moe_tile accepts them only for the (pairs, E, device) they were built for.
struct GroupedBuckets {
  torch::Tensor perm, tiles, ntiles;
  int64_t pairs, num_experts;
  int device_index;
};

constexpr const char* kBucketsName = "hf_int4_g16.GroupedBucketDescriptor";

const GroupedBuckets& grouped_buckets(const pybind11::capsule& capsule) {
  TORCH_CHECK(PyCapsule_IsValid(capsule.ptr(), kBucketsName),
              "buckets must be the GroupedBucketDescriptor returned by moe_bucket");
  return *static_cast<const GroupedBuckets*>(PyCapsule_GetPointer(capsule.ptr(), kBucketsName));
}

// Copies for tests and benchmarks; writing to them cannot change the descriptor.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_bucket_snapshot(const pybind11::capsule& capsule) {
  const auto& b = grouped_buckets(capsule);
  const c10::DeviceGuard guard(b.perm.device());
  return {b.perm.clone(), b.tiles.clone(), b.ntiles.clone()};
}

pybind11::capsule moe_bucket(torch::Tensor ids, int64_t E, const std::string& binpath) {
  TORCH_CHECK(ids.dim() == 1 && ids.is_xpu() && ids.scalar_type() == at::kInt && ids.is_contiguous(),
              "expert ids must be a contiguous rank-1 int32 XPU tensor");
  check_aligned("expert_ids", ids.data_ptr());
  const c10::DeviceGuard guard(ids.device());
  const int device_index = ids.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& que = current_queue(device_index);
  const int64_t P = ids.size(0);
  TORCH_CHECK(E > 0, "empty expert table (E == 0)");
  TORCH_CHECK(P > 0 && P <= kInt32Max / 4, "grouped MoE needs 0 < pairs <= 2^29, got ", P);
  check_int32("E + 1", E + 1);
  const int64_t buckets = E + 1;  // the last bucket collects out-of-range ids
  const auto options = ids.options();
  auto counts = torch::empty({buckets}, options);
  auto perm = torch::empty({P}, options);
  auto tiles = torch::empty({moe_max_tiles(P, buckets), 3}, options);
  auto ntiles = torch::empty({1}, options);
  for (const auto* t : {&ids, &counts, &perm, &tiles, &ntiles}) record_input_stream(*t, device_index);
  const void* ip = ids.data_ptr();
  const void* cp = counts.data_ptr();
  const void* pp = perm.data_ptr();
  const void* tp = tiles.data_ptr();
  const void* np = ntiles.data_ptr();
  const int e32 = (int)E, p32 = (int)P, b32 = (int)buckets, rows32 = (int)kMoeTileRows;
  const auto ctx = que.get_context();

  submit_nd(que, kernel_obj(st, ctx, "hf_moe_bucket_zero", 2), (buckets + 15) / 16, 1, 16, [&](sycl::handler& h) {
    set_ptr_arg(h, 0, cp); h.set_arg(1, b32);
  });
  submit_nd(que, kernel_obj(st, ctx, "hf_moe_bucket_count", 4), (P + 15) / 16, 1, 16, [&](sycl::handler& h) {
    set_ptr_arg(h, 0, ip); set_ptr_arg(h, 1, cp); h.set_arg(2, e32); h.set_arg(3, p32);
  });
  submit_nd(que, kernel_obj(st, ctx, "hf_moe_bucket_scan", 5), 1, 1, 1, [&](sycl::handler& h) {
    set_ptr_arg(h, 0, cp); set_ptr_arg(h, 1, tp); set_ptr_arg(h, 2, np); h.set_arg(3, b32); h.set_arg(4, rows32);
  });
  submit_nd(que, kernel_obj(st, ctx, "hf_moe_bucket_scatter", 5), (P + 15) / 16, 1, 16, [&](sycl::handler& h) {
    set_ptr_arg(h, 0, ip); set_ptr_arg(h, 1, cp); set_ptr_arg(h, 2, pp); h.set_arg(3, e32); h.set_arg(4, p32);
  });
  auto owned = std::make_unique<GroupedBuckets>(GroupedBuckets{perm, tiles, ntiles, P, E, device_index});
  pybind11::capsule capsule(static_cast<const void*>(owned.get()), kBucketsName, [](PyObject* o) {
    delete static_cast<GroupedBuckets*>(PyCapsule_GetPointer(o, kBucketsName));
  });
  owned.release();
  return capsule;
}

torch::Tensor moe_rotate(torch::Tensor x, torch::Tensor signs, const std::string& binpath) {
  TORCH_CHECK(x.dim() == 2 && x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
              "native moe_rotate requires a contiguous rank-2 bf16 XPU input");
  TORCH_CHECK(signs.dim() == 1 && signs.is_xpu() && signs.scalar_type() == at::kChar && signs.is_contiguous(),
              "signs must be contiguous rank-1 int8 XPU");
  const c10::DeviceGuard guard(x.device());
  const int device_index = x.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& que = current_queue(device_index);
  check_same_device(x, signs, "signs");
  const int64_t M = x.size(0), K = x.size(1);
  check_int32("M", M);
  check_int32("K", K);
  const int block = had_block_of(K);
  TORCH_CHECK(signs.numel() == K, "signs length ", signs.numel(), " != K ", K);
  check_aligned("x", x.data_ptr());
  check_aligned("signs", signs.data_ptr());
  auto out = torch::empty({M, K}, x.options().dtype(at::kFloat));
  if (M == 0) return out;
  check_aligned("rotated activation", out.data_ptr());
  const auto& k = kernel_obj(st, que.get_context(), "hf_moe_rotate_had" + std::to_string(block), 4);
  for (const auto* t : {&x, &signs, &out}) record_input_stream(*t, device_index);
  const void* xp = x.data_ptr();
  const void* sp = signs.data_ptr();
  const void* op = out.data_ptr();
  const int k32 = (int)K;
  submit_nd(que, k, 1, M, 128, [&](sycl::handler& h) {
    set_ptr_arg(h, 0, xp); set_ptr_arg(h, 1, sp); set_ptr_arg(h, 2, op); h.set_arg(3, k32);
  });
  return out;
}

torch::Tensor moe_tile(torch::Tensor xr, torch::Tensor q, torch::Tensor u, torch::Tensor d,
                       std::optional<torch::Tensor> bias, const pybind11::capsule& capsule, int64_t in_div,
                       bool out_f32, const std::string& binpath) {
  TORCH_CHECK(xr.dim() == 2 && xr.is_xpu() && xr.scalar_type() == at::kFloat && xr.is_contiguous(),
              "native moe_tile requires a contiguous rank-2 fp32 rotated activation");
  TORCH_CHECK(q.dim() == 3 && u.dim() == 3 && d.dim() == 2, "moe q/u/d must be rank 3/3/2");
  TORCH_CHECK(q.is_xpu() && u.is_xpu() && d.is_xpu() && q.is_contiguous() && u.is_contiguous() && d.is_contiguous(),
              "packed tensors must be contiguous XPU tensors");
  TORCH_CHECK(q.scalar_type() == at::kInt && u.scalar_type() == at::kShort && d.scalar_type() == at::kFloat,
              "moe packed dtypes must be int32/int16/fp32");
  const c10::DeviceGuard guard(xr.device());
  const int device_index = xr.get_device();
  auto& st = state_for(device_index, binpath);
  sycl::queue& que = current_queue(device_index);
  const auto& buckets = grouped_buckets(capsule);
  const int64_t K = xr.size(1), E = d.size(0), N = d.size(1), P = buckets.pairs;
  TORCH_CHECK(buckets.device_index == device_index, "bucket descriptor was built on XPU device ",
              buckets.device_index, ", the activation is on device ", device_index);
  TORCH_CHECK(buckets.num_experts == E, "bucket descriptor was built for E=", buckets.num_experts,
              " experts, the expert table has E=", E);
  const int block = had_block_of(K);
  TORCH_CHECK(E > 0, "empty expert table (E == 0)");
  TORCH_CHECK(q.size(0) == E && q.size(1) == K / 8 && q.size(2) == N, "moe q_pack shape mismatch");
  TORCH_CHECK(u.size(0) == E && u.size(1) == K / 64 && u.size(2) == N, "moe u_pack shape mismatch");
  TORCH_CHECK(in_div > 0 && P == xr.size(0) * in_div, "bucket descriptor was built for ", P,
              " pairs, this batch has rows ", xr.size(0), " * in_div ", in_div);
  const int64_t max_tiles = moe_max_tiles(P, E + 1);
  torch::Tensor perm = buckets.perm, tiles = buckets.tiles, ntiles = buckets.ntiles;  // handles, no copy
  check_same_device(xr, q, "q_pack");
  check_same_device(xr, u, "u_pack");
  check_same_device(xr, d, "d_row");
  check_int32("N", N);
  check_int32("K", K);
  check_int32("E", E);
  check_int32("in_div", in_div);
  check_packed_weight_surface(q.data_ptr(), K / 8, N);
  TORCH_CHECK((K / 8) * N * 4 % 64 == 0, "per-expert q_pack stride is not a multiple of 64 B");
  check_aligned("rotated activation", xr.data_ptr());
  check_aligned("u_pack", u.data_ptr());
  check_aligned("d_row", d.data_ptr());
  const float* biasp = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_xpu() && bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
                    bias->dim() == 2 && bias->size(0) == E && bias->size(1) == N,
                "moe bias_corr must be a contiguous fp32 XPU tensor [E,N]");
    check_same_device(xr, *bias, "bias_corr");
    check_aligned("bias_corr", bias->data_ptr());
    biasp = reinterpret_cast<const float*>(bias->data_ptr());
  }
  auto out = torch::empty({P, N}, xr.options().dtype(out_f32 ? at::kFloat : at::kBFloat16));
  const std::string name = std::string("hf_moe_tile_had") + std::to_string(block) + (out_f32 ? "_f32" : "_bf16");
  const auto& k = kernel_obj(st, que.get_context(), name, 13);
  for (const auto* t : {&xr, &q, &u, &d, &perm, &tiles, &ntiles}) record_input_stream(*t, device_index);
  record_optional_stream(bias, device_index);
  const void* xp = xr.data_ptr();
  const void* qp = q.data_ptr();
  const void* up = u.data_ptr();
  const void* dp = d.data_ptr();
  const void* pp = perm.data_ptr();
  const void* tp = tiles.data_ptr();
  const void* np = ntiles.data_ptr();
  const void* op = out.data_ptr();
  const int n32 = (int)N, k32 = (int)K, e32 = (int)E, div32 = (int)in_div;
  submit_nd(que, k, (N + 31) / 32, max_tiles, 128, [&](sycl::handler& h) {
    set_ptr_arg(h, 0, xp);
    set_ptr_arg(h, 1, qp);
    set_ptr_arg(h, 2, up);
    set_ptr_arg(h, 3, dp);
    set_ptr_arg(h, 4, biasp);
    set_ptr_arg(h, 5, pp);
    set_ptr_arg(h, 6, tp);
    set_ptr_arg(h, 7, np);
    set_ptr_arg(h, 8, op);
    h.set_arg(9, n32); h.set_arg(10, k32); h.set_arg(11, e32); h.set_arg(12, div32);
  });
  return out;
}

std::string probe_stream() {
  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
  auto native = sycl::get_native<L0>(q);
  return std::holds_alternative<ze_command_list_handle_t>(native) ? "immediate_command_list" : "command_queue";
}

std::string launch_mechanism() {
  return "sycl_submit_imported_zebin";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("admit_module", &admit_module);
  m.def("fwht_actq", &fwht_actq);
  m.def("dpas_w4a8", &dpas_w4a8);
  m.def("fwht_dpas_w4a8", &fwht_dpas_w4a8);
  m.def("decode_fused", &decode_fused);
  m.def("moe_pair", &moe_pair);
  m.def("moe_bucket", &moe_bucket);
  m.def("moe_bucket_snapshot", &moe_bucket_snapshot);
  m.def("moe_rotate", &moe_rotate);
  m.def("moe_tile", &moe_tile);
  m.def("probe_stream", &probe_stream);
  m.def("launch_mechanism", &launch_mechanism);
}
