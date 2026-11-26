"""Runtime builder and Python interface for the zero-copy Level-Zero backend."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .format import had_block_for_k

_EXT = None
_BUILD_ERROR: Exception | None = None
_KERNEL_BIN = str(Path(__file__).resolve().parent / "native" / "hf_int4_kernels_bmg.bin")
_ADMITTED_DIGESTS: dict[tuple[int, str], str] = {}


# cl_intel_subgroup_2d_block_io, section 6.13.X.6: behaviour is undefined
# unless the base address is 64-byte aligned, 64 <= width <= 2^24 bytes, width
# is a multiple of 4 (our element sizes), 0 < height <= 2^24, and pitch is at
# least the width and a multiple of 16.  The kernels' sub-group size is fixed
# at 16 by intel_reqd_sub_group_size.  check_block2d_surface() in
# _l0_native.cpp enforces the same rules before any launch.
BLOCK2D_ALIGN = 64
BLOCK2D_MIN_WIDTH = 64
BLOCK2D_MAX_EXTENT = 1 << 24
INT32_MAX = (1 << 31) - 1


def block2d_surface_error(name: str, address: int, width: int, height: int, pitch: int) -> str | None:
    """Why a 2D block-io surface is illegal, or None.  Sizes in bytes except ``height``."""
    if address % BLOCK2D_ALIGN:
        return f"{name} effective base address is not {BLOCK2D_ALIGN}-byte aligned (offset {address % BLOCK2D_ALIGN})"
    if not BLOCK2D_MIN_WIDTH <= width <= BLOCK2D_MAX_EXTENT or width % 4:
        return f"{name} 2D block-io surface width {width} B must be in [64, 2^24] and a multiple of 4"
    if not 0 < height <= BLOCK2D_MAX_EXTENT:
        return f"{name} 2D block-io surface height {height} must be in [1, 2^24]"
    if pitch < width or pitch % 16 or pitch > BLOCK2D_MAX_EXTENT:
        return f"{name} 2D block-io pitch {pitch} B must be >= width, a multiple of 16 and <= 2^24"
    return None


def packed_weight_surface_error(q: torch.Tensor) -> str | None:
    """2D block-read legality of ``q_pack`` ``[K//8,N]`` or ``[E,K//8,N]``.

    ``data_ptr()`` includes the view's storage offset.  The kernels read 16
    output columns per sub-group, so ``N`` must be a multiple of 16: every such
    tile then lies inside the surface, and the surface width ``4N`` is a
    multiple of 64 bytes.
    """
    rows, n = int(q.shape[-2]), int(q.shape[-1])
    if n < 16 or n % 16:
        return f"native requires N>=16 and N%16==0 (surface width 4N a multiple of 64 B), got N={n}"
    if q.ndim == 3 and (rows * n * 4) % BLOCK2D_ALIGN:
        return f"per-expert q_pack stride {rows * n * 4} B is not a multiple of {BLOCK2D_ALIGN}"
    return block2d_surface_error("q_pack", q.data_ptr(), 4 * n, rows, 4 * n)


def dpas_surface_error(m: int, k: int, n: int, xi8_address: int = 0) -> str | None:
    """The W4A8 DPAS kernels also 2D-read int8 activations and 2D-write BF16 output."""
    return (block2d_surface_error("int8 activation", xi8_address, k, m, k)
            or block2d_surface_error("bf16 output", 0, 2 * n, m, 2 * n))


def _admitted_module(ext, device: torch.device, binpath: str | None = None) -> str:
    path = binpath or kernel_bin()
    dev_idx = 0 if device.index is None else int(device.index)
    key = (dev_idx, path)
    digest = _ADMITTED_DIGESTS.get(key)
    if digest is None:
        digest = ext.admit_module(dev_idx, path)
        _ADMITTED_DIGESTS[key] = digest
    return digest


def kernel_bin() -> str:
    # Path.resolve() costs tens of microseconds per call in the container and
    # was a measurable decode-host bottleneck.  The artifact location is fixed.
    return _KERNEL_BIN


def native_source() -> str:
    return str(Path(__file__).resolve().parent / "native" / "hf_int4_kernels.cl")


def build_extension(verbose: bool = False):
    global _EXT, _BUILD_ERROR
    if _EXT is not None:
        return _EXT
    if _BUILD_ERROR is not None:
        raise RuntimeError(f"Level-Zero extension previously failed to build: {_BUILD_ERROR}")
    try:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent
        build_dir = root / ".torch_ext_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXT = load(
            name="hf_int4_g16_l0_native",
            sources=[str(root / "_l0_native.cpp")],
            build_directory=str(build_dir),
            extra_include_paths=["/opt/venv/include"],
            extra_ldflags=[
                "-L/opt/venv/lib", "-lsycl", "-lze_loader", "-lc10_xpu", "-ltorch_xpu",
            ],
            verbose=verbose or bool(os.environ.get("HF_INT4_G16_BUILD_VERBOSE")),
        )
        return _EXT
    except Exception as e:  # fail-closed: caller may explicitly choose Triton
        _BUILD_ERROR = e
        raise


def probe_stream() -> str:
    return build_extension().probe_stream()


def _aligned16(t: torch.Tensor | None) -> bool:
    return t is None or (t.data_ptr() % 16) == 0


def _same_device(*tensors: torch.Tensor | None) -> bool:
    devs = {t.device for t in tensors if t is not None}
    return len(devs) == 1


def validate_native_operands(x: torch.Tensor, signs: torch.Tensor | None = None,
                             q: torch.Tensor | None = None, u: torch.Tensor | None = None,
                             d: torch.Tensor | None = None, sx: torch.Tensor | None = None,
                             bias_corr: torch.Tensor | None = None,
                             expected_x_dtype: torch.dtype = torch.bfloat16,
                             dpas: bool = False) -> None:
    """Validate the complete exposed native contract before touching Level Zero.

    Packed q/u/d are optional only for the standalone ACTQ entry point.  When
    supplied they must be supplied together and satisfy the exact bit-container
    contract; validation never manufactures or copies operand storage.
    """
    if x.ndim != 2:
        raise ValueError(f"native input must be rank-2, got {tuple(x.shape)}")
    m, k = x.shape
    tensors = tuple(t for t in (x, signs, q, u, d, sx, bias_corr) if t is not None)
    if not all(t.is_xpu for t in tensors):
        raise ValueError("native operands must all be XPU tensors")
    if not _same_device(*tensors):
        raise ValueError("native operands must all be on the same XPU device")
    if x.dtype != expected_x_dtype:
        raise ValueError(f"native input dtype {x.dtype} != {expected_x_dtype}")
    if signs is not None and signs.dtype != torch.int8:
        raise ValueError(f"signs must be int8, got {signs.dtype}")
    if not x.is_contiguous() or any(t is not None and not t.is_contiguous() for t in (signs, q, u, d, sx, bias_corr)):
        raise ValueError("native operands must be contiguous")
    if bias_corr is not None and bias_corr.dtype != torch.float32:
        raise ValueError("bias_corr must be contiguous fp32")
    had_block_for_k(k)
    if k > 17408:
        raise ValueError(f"native requires K<=17408, got K={k}")
    if signs is not None and signs.numel() != k:
        raise ValueError(f"signs length {signs.numel()} != K {k}")
    packed = (q, u, d)
    if any(t is not None for t in packed):
        if not all(t is not None for t in packed):
            raise ValueError("q_pack/u_pack/d_row must be validated together")
        if q.dtype != torch.int32 or u.dtype != torch.int16 or d.dtype != torch.float32:
            raise ValueError(f"packed dtype mismatch: q={q.dtype} u={u.dtype} d={d.dtype}")
        if q.ndim != 2 or u.ndim != 2 or q.shape[0] * 8 != k or u.shape[0] * 64 != k:
            raise ValueError(f"packed K mismatch for K={k}: q={tuple(q.shape)} u={tuple(u.shape)}")
        if q.shape[1] != u.shape[1] or q.shape[1] != d.numel():
            raise ValueError("packed q/u/d N mismatch")
        n = int(q.shape[1])
        if max(m, n) > INT32_MAX:
            raise ValueError(f"M={m} or N={n} does not fit the kernels' int32 arguments")
        error = packed_weight_surface_error(q)
        if error is None and dpas and m:
            # A caller-owned int8 activation is itself a 2D-read surface.
            address = x.data_ptr() if x.dtype == torch.int8 else 0
            error = dpas_surface_error(int(m), int(k), n, address)
        if error:
            raise ValueError(error)
    if sx is not None and (sx.dtype != torch.float32 or sx.numel() != m):
        raise ValueError(f"sx must be contiguous fp32 with M={m} elements")
    for name, t in [("x", x), ("signs", signs), ("q_pack", q), ("u_pack", u), ("d_row", d), ("sx", sx), ("bias_corr", bias_corr)]:
        if t is not None and not _aligned16(t):
            raise ValueError(f"{name} base pointer is not 16-byte aligned")


def native_supported(x: torch.Tensor, n: int, bias_present: bool = False,
                     signs: torch.Tensor | None = None, q_pack: torch.Tensor | None = None,
                     u_pack: torch.Tensor | None = None, d_row: torch.Tensor | None = None,
                     bias_corr: torch.Tensor | None = None, dpas: bool = False) -> bool:
    """Conservative fast-path gate.  All other cases use the explicit Triton baseline."""
    if not (x.is_xpu and x.is_contiguous()):
        return False
    if x.dtype != torch.bfloat16:
        return False
    if x.ndim != 2 or x.shape[1] % 64 or x.shape[1] > 17408:
        return False
    # Widths the 2D block-io surfaces cannot legally describe (see
    # packed_weight_surface_error) go to the universal Triton kernel.  The DPAS
    # path additionally writes a 2N-byte-wide surface, so it needs N >= 32.
    if n < (32 if dpas else 16) or n % 16:
        return False
    if signs is not None and q_pack is not None and u_pack is not None and d_row is not None:
        try:
            validate_native_operands(x, signs, q_pack, u_pack, d_row, bias_corr=bias_corr, dpas=dpas)
        except (TypeError, ValueError):
            return False
    # Native kernels have one bias pointer.  If both a model bias and bias_corr
    # exist, the Triton kernel performs the exact two-term fp32 epilogue.
    if bias_present:
        return False
    return True


def fwht_actq(x: torch.Tensor, signs: torch.Tensor):
    ext = build_extension()
    validate_native_operands(x, signs)
    m, k = x.shape
    xi8 = torch.empty((m, k), device=x.device, dtype=torch.int8)
    sx = torch.empty((m,), device=x.device, dtype=torch.float32)
    ext.fwht_actq(x, signs, xi8, sx, _admitted_module(ext, x.device))
    return xi8, sx


def _optional_bias(bias_corr):
    # The native ABI is zero-copy.  Non-contiguous correction storage is a
    # validation error, never an implicit copy that could hide a lifetime or
    # stream-ordering bug.
    return bias_corr


def launch_mechanism() -> str:
    return build_extension().launch_mechanism()


def w4a8_native(x, signs, q, u, d, bias_corr=None):
    validate_native_operands(x, signs, q, u, d, bias_corr=_optional_bias(bias_corr), dpas=True)
    ext = build_extension()
    return ext.fwht_dpas_w4a8(x, signs, q, u, d, _optional_bias(bias_corr), _admitted_module(ext, x.device))


def decode_native(x, signs, q, u, d, bias_corr=None):
    validate_native_operands(x, signs, q, u, d, bias_corr=_optional_bias(bias_corr))
    ext = build_extension()
    return ext.decode_fused(x, signs, q, u, d, _optional_bias(bias_corr), _admitted_module(ext, x.device))


def moe_supported(x: torch.Tensor, q_pack: torch.Tensor) -> bool:
    """Fast-path gate for the routed-expert kernels (FP32 accumulation, so no K bound)."""
    return bool(x.is_xpu and x.is_contiguous() and x.dtype == torch.bfloat16 and x.data_ptr() % 16 == 0
                and packed_weight_surface_error(q_pack) is None)


def moe_pair_native(x, expert_ids, signs, q, u, d, bias_corr, in_div: int, out_f32: bool):
    """Grouped routed-expert W4A16 launch; the C++ side validates the full contract."""
    ext = build_extension()
    return ext.moe_pair(x, signs, q, u, d, bias_corr, expert_ids, int(in_div), bool(out_f32),
                        _admitted_module(ext, x.device))


# Token rows per grouped tile; hf_moe_tile_* kernels are built for this size.
MOE_TILE_ROWS = 16


def moe_bucket_native(expert_ids, num_experts: int):
    """Group pair ids by expert on the device into an opaque ``GroupedBucketDescriptor``.

    Ids outside ``[0, num_experts)`` share one extra bucket that writes zeros.
    The descriptor owns its routing buffers and is bound to
    ``(len(expert_ids), num_experts, device)``; :func:`moe_tile_native` rejects
    it for any other batch.  The check reads metadata only.
    """
    ext = build_extension()
    return ext.moe_bucket(expert_ids, int(num_experts), _admitted_module(ext, expert_ids.device))


def moe_bucket_snapshot(buckets):
    """Copies of a descriptor's ``(perm, tiles, ntiles)``, for tests and benchmarks."""
    return build_extension().moe_bucket_snapshot(buckets)


def moe_rotate_native(x, signs):
    """FP32 blockwise Hadamard rotation, the same operations as the pair kernel."""
    ext = build_extension()
    return ext.moe_rotate(x, signs, _admitted_module(ext, x.device))


def moe_tile_native(x_rot, q, u, d, bias_corr, buckets, in_div: int, out_f32: bool):
    ext = build_extension()
    return ext.moe_tile(x_rot, q, u, d, bias_corr, buckets, int(in_div), bool(out_f32),
                        _admitted_module(ext, x_rot.device))


def dpas_native_fixed(xi8, sx, q, u, d, bias_corr=None, write_i32=False):
    # Fixed-int8 input is already quantized but obeys the same device/packed contract.
    validate_native_operands(xi8, None, q, u, d, sx=sx, bias_corr=_optional_bias(bias_corr),
                             expected_x_dtype=torch.int8, dpas=True)
    ext = build_extension()
    ret = ext.dpas_w4a8(xi8, sx, q, u, d, _optional_bias(bias_corr), bool(write_i32), _admitted_module(ext, xi8.device))
    return ret
