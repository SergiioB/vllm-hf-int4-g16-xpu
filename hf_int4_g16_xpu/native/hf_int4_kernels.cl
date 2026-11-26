
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// =========================================================================
// 1. IN-REGISTER 128-POINT FWHT ACROSS 16 SUBGROUP LANES (8 floats/lane)
// =========================================================================
inline void fwht128_sg16(float v[8], const int lane) {
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float o = sub_group_shuffle_xor(v[j], 1);
        v[j] = (lane & 1) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float o = sub_group_shuffle_xor(v[j], 2);
        v[j] = (lane & 2) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float o = sub_group_shuffle_xor(v[j], 4);
        v[j] = (lane & 4) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float o = sub_group_shuffle_xor(v[j], 8);
        v[j] = (lane & 8) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 8; j += 2) {
        float a = v[j], b = v[j + 1];
        v[j] = a + b; v[j + 1] = a - b;
    }
    #pragma unroll
    for (int j = 0; j < 8; j += 4) {
        float a0 = v[j], a1 = v[j + 1], b0 = v[j + 2], b1 = v[j + 3];
        v[j] = a0 + b0; v[j + 2] = a0 - b0;
        v[j + 1] = a1 + b1; v[j + 3] = a1 - b1;
    }
    const float inv_s = 0.088388347648f; // 1 / sqrt(128)
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float a = v[j], b = v[j + 4];
        v[j]     = (a + b) * inv_s;
        v[j + 4] = (a - b) * inv_s;
    }
}

// In-register 64-point FWHT across 16 subgroup lanes (4 floats/lane).  Lane
// bits are element-index bits 0..3 and j is bits 4..5, like the 128 variant.
inline void fwht64_sg16(float v[4], const int lane) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float o = sub_group_shuffle_xor(v[j], 1);
        v[j] = (lane & 1) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float o = sub_group_shuffle_xor(v[j], 2);
        v[j] = (lane & 2) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float o = sub_group_shuffle_xor(v[j], 4);
        v[j] = (lane & 4) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float o = sub_group_shuffle_xor(v[j], 8);
        v[j] = (lane & 8) ? (o - v[j]) : (v[j] + o);
    }
    #pragma unroll
    for (int j = 0; j < 4; j += 2) {
        float a = v[j], b = v[j + 1];
        v[j] = a + b; v[j + 1] = a - b;
    }
    const float inv_s = 0.125f; // 1 / sqrt(64), exact in FP32
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        float a = v[j], b = v[j + 2];
        v[j]     = (a + b) * inv_s;
        v[j + 2] = (a - b) * inv_s;
    }
}

// =========================================================================
// 2. SINGLE-PASS FUSED BlockFWHT (128 or 64) + Per-Token INT8 ActQ
// =========================================================================
#define DEFINE_FWHT_ACTQ(NAME, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT)                              \
__attribute__((intel_reqd_sub_group_size(16)))                                                      \
kernel void NAME(                                                                                   \
    global const ushort* X_bf16,                                                                    \
    global const char*   Signs,                                                                     \
    global char*         X_i8,                                                                      \
    global float*        Sx,                                                                        \
    const int K                                                                                     \
) {                                                                                                 \
    local float sg_max[16];                                                                         \
                                                                                                    \
    const int m = get_group_id(0);                                                                  \
    const int lid = get_local_id(0);                                                                \
    const int sg = lid >> 4;                                                                        \
    const int lane = lid & 15;                                                                      \
    const int nblocks = K >> (BLK_SHIFT);                                                           \
                                                                                                    \
    global const ushort* row_in = X_bf16 + (long)m * K;                                             \
    float local_max = 1e-12f;                                                                       \
                                                                                                    \
    for (int blk = sg; blk < nblocks; blk += 16) {                                                  \
        const int base = blk << (BLK_SHIFT);                                                        \
        VEC_T raw = BLOCK_READ(row_in + base);                                                      \
        float v[NJ];                                                                                \
        _Pragma("unroll")                                                                           \
        for (int j = 0; j < (NJ); j++) {                                                            \
            int idx = base + j * 16 + lane;                                                         \
            v[j] = intel_convert_as_bfloat16_float(raw[j]) * (float)Signs[idx];                     \
        }                                                                                           \
        FWHT(v, lane);                                                                              \
        _Pragma("unroll")                                                                           \
        for (int j = 0; j < (NJ); j++) {                                                            \
            local_max = fmax(local_max, fabs(v[j]));                                                \
        }                                                                                           \
    }                                                                                               \
                                                                                                    \
    float sgm = sub_group_reduce_max(local_max);                                                    \
    if (lane == 0) sg_max[sg] = sgm;                                                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                   \
                                                                                                    \
    float rmax = sub_group_reduce_max(sg_max[lane]);                                                \
    const float scale = rmax / 127.0f;                                                              \
    const float inv_scale = 127.0f / rmax;                                                          \
    if (lid == 0) Sx[m] = scale;                                                                    \
                                                                                                    \
    global char* row_out = X_i8 + (long)m * K;                                                      \
    for (int blk = sg; blk < nblocks; blk += 16) {                                                  \
        const int base = blk << (BLK_SHIFT);                                                        \
        VEC_T rawq = BLOCK_READ(row_in + base);                                                     \
        float vq[NJ];                                                                               \
        _Pragma("unroll")                                                                           \
        for (int j = 0; j < (NJ); j++) {                                                            \
            int idx = base + j * 16 + lane;                                                         \
            vq[j] = intel_convert_as_bfloat16_float(rawq[j]) * (float)Signs[idx];                   \
        }                                                                                           \
        FWHT(vq, lane);                                                                             \
        _Pragma("unroll")                                                                           \
        for (int j = 0; j < (NJ); j++) {                                                            \
            int idx = base + j * 16 + lane;                                                         \
            float q = clamp(rint(vq[j] * inv_scale), -127.0f, 127.0f);                              \
            row_out[idx] = (char)((int)q);                                                          \
        }                                                                                           \
    }                                                                                               \
}

DEFINE_FWHT_ACTQ(fused_fwht128_actq, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7)
DEFINE_FWHT_ACTQ(fused_fwht64_actq, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6)

// =========================================================================
// 3. BIT-EXACT SWAR 4-WEIGHT-AT-ONCE IN-REGISTER UNPACK + SAFE U LOAD
// =========================================================================
inline int swar_unpack_4(const uint nib4, const uint u, const uint y_lo, const uint y_inv) {
    const uint prod = nib4 * u;
    return as_int(((prod | 0x80808080u) - y_lo) ^ ((prod ^ y_inv) & 0x80808080u));
}

inline int8 unpack_k32_s8(
    const uint w0, const uint w1, const uint w2, const uint w3,
    const uint u0, const uint u1
) {
    const uint y0_lo  = ((8u * u0) & 0x7Fu) * 0x01010101u;
    const uint y0_inv = (~(8u * u0))        * 0x01010101u;
    const uint y1_lo  = ((8u * u1) & 0x7Fu) * 0x01010101u;
    const uint y1_inv = (~(8u * u1))        * 0x01010101u;
    return (int8)(
        swar_unpack_4( w0       & 0x0F0F0F0Fu, u0, y0_lo, y0_inv),
        swar_unpack_4((w0 >> 4) & 0x0F0F0F0Fu, u0, y0_lo, y0_inv),
        swar_unpack_4( w1       & 0x0F0F0F0Fu, u0, y0_lo, y0_inv),
        swar_unpack_4((w1 >> 4) & 0x0F0F0F0Fu, u0, y0_lo, y0_inv),
        swar_unpack_4( w2       & 0x0F0F0F0Fu, u1, y1_lo, y1_inv),
        swar_unpack_4((w2 >> 4) & 0x0F0F0F0Fu, u1, y1_lo, y1_inv),
        swar_unpack_4( w3       & 0x0F0F0F0Fu, u1, y1_lo, y1_inv),
        swar_unpack_4((w3 >> 4) & 0x0F0F0F0Fu, u1, y1_lo, y1_inv)
    );
}

inline uint load_u4_safe(global const ushort* U_packed, const int kb, const int N, const int n0, const int lane) {
    if (n0 + 16 <= N) {
        return (uint)intel_sub_group_block_read_us(U_packed + (long)kb * N + n0);
    }
    const int col = n0 + lane;
    return (col < N) ? (uint)U_packed[(long)kb * N + col] : 0u;
}

// =========================================================================
// 4A. TALL-M STREAMING-A FUSED HF-INT4-G16 -> XMX DPAS KERNEL (SAFE OOB!)
// =========================================================================
#define DEFINE_HF_DPAS_TALL(NAME, SG_M, WG_M, WG_N, PF, PANEL_N)                                 \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const char*   A_i8,                                                                   \
    global const uint*   W_packed,                                                               \
    global const ushort* U_packed,                                                               \
    global const float*  Sx,                                                                     \
    global const float*  D_row,                                                                  \
    global const float*  Bias,                                                                   \
    global ushort*       C_bf16,                                                                 \
    global int*          C_i32,                                                                  \
    const int M, const int N, const int K, const int dump_i32                                    \
) {                                                                                              \
    const int lane = get_sub_group_local_id();                                                   \
    const int sg = get_sub_group_id();                                                           \
    int gx = get_group_id(0), gy = get_group_id(1);                                              \
    if (PANEL_N > 1) {                                                                           \
        const int nx = get_num_groups(0), ny = get_num_groups(1);                                \
        const int flat = gy * nx + gx;                                                           \
        const int blk = flat / ((PANEL_N) * ny);                                                 \
        const int rem = flat % ((PANEL_N) * ny);                                                 \
        const int x0 = blk * (PANEL_N);                                                          \
        const int pw = min((PANEL_N), nx - x0);                                                  \
        gy = rem / pw; gx = x0 + (rem % pw);                                                     \
    }                                                                                            \
    const int m0 = (gy * (WG_M) + (sg % (WG_M))) * (SG_M);                                       \
    const int n0 = (gx * (WG_N) + (sg / (WG_M))) * 16;                                           \
    if (m0 >= M || n0 >= N) return;                                                              \
                                                                                                 \
    int8 acc[(SG_M) / 8];                                                                        \
    _Pragma("unroll")                                                                            \
    for (int i = 0; i < (SG_M) / 8; i++) acc[i] = (int8)(0);                                     \
                                                                                                 \
    const int n_kb = K >> 6;                                                                     \
    for (int kb = 0; kb < n_kb; kb++) {                                                          \
        if (PF > 0 && kb + PF < n_kb) {                                                          \
            intel_sub_group_2d_block_prefetch_32b_8r16x1c(                                       \
                (global void*)W_packed, N * 4, K >> 3, N * 4, (int2)(n0, (kb + PF) * 8));        \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < (SG_M) / 32; r++)                                                \
                intel_sub_group_2d_block_prefetch_16b_32r16x2c(                                  \
                    (global void*)A_i8, K, M, K, (int2)((kb + PF) * 32, m0 + 32 * r));           \
        }                                                                                        \
        uint w[8];                                                                               \
        intel_sub_group_2d_block_read_32b_8r16x1c(                                               \
            (global void*)W_packed, N * 4, K >> 3, N * 4, (int2)(n0, kb * 8), w);                \
        const uint u4 = load_u4_safe(U_packed, kb, N, n0, lane);                                 \
        const int8 b0 = unpack_k32_s8(w[0], w[1], w[2], w[3], (u4 & 0xFu) + 1u, ((u4 >> 4) & 0xFu) + 1u); \
        const int8 b1 = unpack_k32_s8(w[4], w[5], w[6], w[7], ((u4 >> 8) & 0xFu) + 1u, ((u4 >> 12) & 0xFu) + 1u); \
        _Pragma("unroll")                                                                        \
        for (int r = 0; r < (SG_M) / 32; r++) {                                                  \
            ushort ab[64];                                                                       \
            intel_sub_group_2d_block_read_16b_32r16x2c(                                          \
                (global void*)A_i8, K, M, K, (int2)(kb * 32, m0 + 32 * r), ab);                  \
            _Pragma("unroll")                                                                    \
            for (int q = 0; q < 4; q++) {                                                        \
                acc[r * 4 + q] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, &ab[q * 8])),      b0, acc[r * 4 + q]); \
                acc[r * 4 + q] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, &ab[32 + q * 8])), b1, acc[r * 4 + q]); \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
                                                                                                 \
    const int col = n0 + lane;                                                                   \
    const float dc = (col < N) ? D_row[col] : 0.0f;                                              \
    const float bc = (Bias != 0 && col < N) ? Bias[col] : 0.0f;                                              \
    _Pragma("unroll")                                                                            \
    for (int i = 0; i < (SG_M) / 8; i++) {                                                       \
        float8 sx_v;                                                                             \
        _Pragma("unroll")                                                                        \
        for (int r = 0; r < 8; r++) {                                                            \
            const int row = m0 + 8 * i + r;                                                      \
            sx_v[r] = (row < M) ? Sx[row] : 0.0f;                                                \
        }                                                                                        \
        if (dump_i32 && col < N) {                                                               \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < 8; r++) {                                                        \
                const int row = m0 + 8 * i + r;                                                  \
                if (row < M) C_i32[(long)row * N + col] = acc[i][r];                             \
            }                                                                                    \
        }                                                                                        \
        float8 fv = convert_float8(acc[i]) * (sx_v * dc) + bc;                                   \
        ushort8 out_bf;                                                                          \
        _Pragma("unroll")                                                                        \
        for (int r = 0; r < 8; r++) out_bf[r] = intel_convert_bfloat16_as_ushort(fv[r]);         \
        if (n0 + 16 <= N && m0 + 8 * i + 8 <= M) {                                               \
            intel_sub_group_2d_block_write_16b_8r16x1c(                                          \
                (global void*)C_bf16, N * 2, M, N * 2, (int2)(n0, m0 + 8 * i), (private ushort*)&out_bf); \
        } else if (col < N) {                                                                    \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < 8; r++) {                                                        \
                const int row = m0 + 8 * i + r;                                                  \
                if (row < M) C_bf16[(long)row * N + col] = out_bf[r];                            \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

// =========================================================================
// 4B. GLOBAL SPLIT-K=2 FOR TALL-M (M=512 mlp_down diagnosis candidate!)
//     Writes exact int32 partials [SPLIT_K, M, N] -> reduced & epilogue fused
// =========================================================================
#define DEFINE_HF_DPAS_TALL_GSK(NAME, SG_M, WG_M, WG_N, PF, PANEL_N, G_SK)                       \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const char*   A_i8,                                                                   \
    global const uint*   W_packed,                                                               \
    global const ushort* U_packed,                                                               \
    global int*          Part_i32,                                                               \
    const int M, const int N, const int K                                                        \
) {                                                                                              \
    const int lane = get_sub_group_local_id();                                                   \
    const int sg = get_sub_group_id();                                                           \
    const int sk = get_group_id(2);                                                              \
    int gx = get_group_id(0), gy = get_group_id(1);                                              \
    if (PANEL_N > 1) {                                                                           \
        const int nx = get_num_groups(0), ny = get_num_groups(1);                                \
        const int flat = gy * nx + gx;                                                           \
        const int blk = flat / ((PANEL_N) * ny);                                                 \
        const int rem = flat % ((PANEL_N) * ny);                                                 \
        const int x0 = blk * (PANEL_N);                                                          \
        const int pw = min((PANEL_N), nx - x0);                                                  \
        gy = rem / pw; gx = x0 + (rem % pw);                                                     \
    }                                                                                            \
    const int m0 = (gy * (WG_M) + (sg % (WG_M))) * (SG_M);                                       \
    const int n0 = (gx * (WG_N) + (sg / (WG_M))) * 16;                                           \
    if (m0 >= M || n0 >= N) return;                                                              \
                                                                                                 \
    int8 acc[(SG_M) / 8];                                                                        \
    _Pragma("unroll")                                                                            \
    for (int i = 0; i < (SG_M) / 8; i++) acc[i] = (int8)(0);                                     \
                                                                                                 \
    const int n_kb = K >> 6;                                                                     \
    const int kb_per = (n_kb + (G_SK) - 1) / (G_SK);                                             \
    const int kb0 = sk * kb_per;                                                                 \
    const int kb1 = min(kb0 + kb_per, n_kb);                                                     \
    for (int kb = kb0; kb < kb1; kb++) {                                                         \
        if (PF > 0 && kb + PF < kb1) {                                                           \
            intel_sub_group_2d_block_prefetch_32b_8r16x1c(                                       \
                (global void*)W_packed, N * 4, K >> 3, N * 4, (int2)(n0, (kb + PF) * 8));        \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < (SG_M) / 32; r++)                                                \
                intel_sub_group_2d_block_prefetch_16b_32r16x2c(                                  \
                    (global void*)A_i8, K, M, K, (int2)((kb + PF) * 32, m0 + 32 * r));           \
        }                                                                                        \
        uint w[8];                                                                               \
        intel_sub_group_2d_block_read_32b_8r16x1c(                                               \
            (global void*)W_packed, N * 4, K >> 3, N * 4, (int2)(n0, kb * 8), w);                \
        const uint u4 = load_u4_safe(U_packed, kb, N, n0, lane);                                 \
        const int8 b0 = unpack_k32_s8(w[0], w[1], w[2], w[3], (u4 & 0xFu) + 1u, ((u4 >> 4) & 0xFu) + 1u); \
        const int8 b1 = unpack_k32_s8(w[4], w[5], w[6], w[7], ((u4 >> 8) & 0xFu) + 1u, ((u4 >> 12) & 0xFu) + 1u); \
        _Pragma("unroll")                                                                        \
        for (int r = 0; r < (SG_M) / 32; r++) {                                                  \
            ushort ab[64];                                                                       \
            intel_sub_group_2d_block_read_16b_32r16x2c(                                          \
                (global void*)A_i8, K, M, K, (int2)(kb * 32, m0 + 32 * r), ab);                  \
            _Pragma("unroll")                                                                    \
            for (int q = 0; q < 4; q++) {                                                        \
                acc[r * 4 + q] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, &ab[q * 8])),      b0, acc[r * 4 + q]); \
                acc[r * 4 + q] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, &ab[32 + q * 8])), b1, acc[r * 4 + q]); \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    const int col = n0 + lane;                                                                   \
    global int* dst_base = Part_i32 + (long)sk * M * N;                                          \
    if (col < N) {                                                                               \
        _Pragma("unroll")                                                                        \
        for (int i = 0; i < (SG_M) / 8; i++) {                                                   \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < 8; r++) {                                                        \
                const int row = m0 + 8 * i + r;                                                  \
                if (row < M) dst_base[(long)row * N + col] = acc[i][r];                          \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

kernel void reduce_gsk2_epilogue(
    global const int*   Part_i32,
    global const float* Sx,
    global const float* D_row,
    global const float* Bias,
    global ushort*      C_bf16,
    global int*         C_i32,
    const int M, const int N, const int dump_i32
) {
    const long idx = (long)get_global_id(0);
    const long tot = (long)M * N;
    if (idx >= tot) return;
    const int row = (int)(idx / N);
    const int col = (int)(idx % N);
    const int s = Part_i32[idx] + Part_i32[tot + idx];
    if (dump_i32) C_i32[idx] = s;
    const float fv = (float)s * (Sx[row] * D_row[col]) + ((Bias != 0) ? Bias[col] : 0.0f);
    C_bf16[idx] = intel_convert_bfloat16_as_ushort(fv);
}

// =========================================================================
// 4C. HARDENED SMALL-M (M=8, 16, 32) WITH PREDICATED ACTIVE GUARD!
//     ZERO early return before barrier(CLK_LOCAL_MEM_FENCE) -> deadlock-free
//     for ALL arbitrary M, N, K shapes (even N % 32 != 0)!
// =========================================================================
#define DEFINE_HF_DPAS_SMALLM_SLM(NAME, SG_M, K_SPLIT, WG_N)                                     \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const char*   A_i8,                                                                   \
    global const uint*   W_packed,                                                               \
    global const ushort* U_packed,                                                               \
    global const float*  Sx,                                                                     \
    global const float*  D_row,                                                                  \
    global const float*  Bias,                                                                   \
    global ushort*       C_bf16,                                                                 \
    global int*          C_i32,                                                                  \
    const int M, const int N, const int K, const int dump_i32                                    \
) {                                                                                              \
    local int slm_acc[K_SPLIT][WG_N][SG_M][16];                                                  \
    const int lane = get_sub_group_local_id();                                                   \
    const int sg = get_sub_group_id();                                                           \
    const int ks = sg % (K_SPLIT);                                                               \
    const int wn = sg / (K_SPLIT);                                                               \
    const int m0 = get_group_id(1) * (SG_M);                                                     \
    const int n0 = (get_group_id(0) * (WG_N) + wn) * 16;                                         \
    const bool active = (m0 < M && n0 < N);                                                      \
                                                                                                 \
    int8 acc[(SG_M) / 8];                                                                        \
    _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++) acc[i] = (int8)(0);                   \
                                                                                                 \
    if (active) {                                                                                \
        const int n_kb = K >> 6;                                                                 \
        for (int kb = ks; kb < n_kb; kb += (K_SPLIT)) {                                          \
            if (kb + 2 * (K_SPLIT) < n_kb) {                                                     \
                intel_sub_group_2d_block_prefetch_32b_8r16x1c(                                   \
                    (global void*)W_packed, N * 4, K >> 3, N * 4, (int2)(n0, (kb + 2*(K_SPLIT)) * 8)); \
            }                                                                                    \
            uint w[8];                                                                           \
            intel_sub_group_2d_block_read_32b_8r16x1c(                                           \
                (global void*)W_packed, N * 4, K >> 3, N * 4, (int2)(n0, kb * 8), w);            \
            const uint u4 = load_u4_safe(U_packed, kb, N, n0, lane);                             \
            const int8 b0 = unpack_k32_s8(w[0], w[1], w[2], w[3], (u4 & 0xFu) + 1u, ((u4 >> 4) & 0xFu) + 1u); \
            const int8 b1 = unpack_k32_s8(w[4], w[5], w[6], w[7], ((u4 >> 8) & 0xFu) + 1u, ((u4 >> 12) & 0xFu) + 1u); \
            _Pragma("unroll")                                                                    \
            for (int i = 0; i < (SG_M) / 8; i++) {                                               \
                ushort a_buf[16];                                                                \
                intel_sub_group_2d_block_read_16b_8r16x2c(                                       \
                    (global void*)A_i8, K, M, K, (int2)(kb * 32, m0 + 8 * i), a_buf);            \
                acc[i] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, &a_buf[0])), b0, acc[i]); \
                acc[i] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, &a_buf[8])), b1, acc[i]); \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    _Pragma("unroll")                                                                            \
    for (int i = 0; i < (SG_M) / 8; i++)                                                         \
        _Pragma("unroll")                                                                        \
        for (int r = 0; r < 8; r++) slm_acc[ks][wn][8 * i + r][lane] = acc[i][r];                \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
    if (ks == 0 && active) {                                                                     \
        const int col = n0 + lane;                                                               \
        const float dc = (col < N) ? D_row[col] : 0.0f;                                          \
        const float bc = (Bias != 0 && col < N) ? Bias[col] : 0.0f;                                          \
        _Pragma("unroll")                                                                        \
        for (int i = 0; i < (SG_M) / 8; i++) {                                                   \
            int8 tot = (int8)(0);                                                                \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < 8; r++) {                                                        \
                int s = 0;                                                                       \
                _Pragma("unroll")                                                                \
                for (int k = 0; k < (K_SPLIT); k++) s += slm_acc[k][wn][8 * i + r][lane];        \
                tot[r] = s;                                                                      \
                if (dump_i32 && col < N && (m0 + 8 * i + r) < M)                                 \
                    C_i32[(long)(m0 + 8 * i + r) * N + col] = s;                                 \
            }                                                                                    \
            float8 sx_v;                                                                         \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < 8; r++) sx_v[r] = (m0 + 8 * i + r < M) ? Sx[m0 + 8 * i + r] : 0.0f; \
            float8 fv = convert_float8(tot) * (sx_v * dc) + bc;                                  \
            ushort8 out_bf;                                                                      \
            _Pragma("unroll")                                                                    \
            for (int r = 0; r < 8; r++) out_bf[r] = intel_convert_bfloat16_as_ushort(fv[r]);     \
            if (n0 + 16 <= N && m0 + 8 * i + 8 <= M) {                                           \
                intel_sub_group_2d_block_write_16b_8r16x1c(                                      \
                    (global void*)C_bf16, N * 2, M, N * 2, (int2)(n0, m0 + 8 * i), (private ushort*)&out_bf); \
            } else if (col < N) {                                                                \
                _Pragma("unroll")                                                                \
                for (int r = 0; r < 8; r++) {                                                    \
                    const int row = m0 + 8 * i + r;                                              \
                    if (row < M) C_bf16[(long)row * N + col] = out_bf[r];                        \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

DEFINE_HF_DPAS_SMALLM_SLM(hf_dpas_m8_sk4,   8, 4, 2)
DEFINE_HF_DPAS_SMALLM_SLM(hf_dpas_m16_sk4, 16, 4, 2)
DEFINE_HF_DPAS_SMALLM_SLM(hf_dpas_m32_sk4, 32, 4, 2)
DEFINE_HF_DPAS_TALL(hf_dpas_m32_n16,       32, 1,  8, 2, 8)
DEFINE_HF_DPAS_TALL(hf_dpas_m64_n16,       64, 1,  8, 2, 8)
DEFINE_HF_DPAS_TALL(hf_dpas_m64_n16_wg16,  64, 1, 16, 2, 8)
DEFINE_HF_DPAS_TALL(hf_dpas_m128_n16,     128, 1, 16, 2, 8)
DEFINE_HF_DPAS_TALL(hf_dpas_m128_n16_wg8, 128, 1,  8, 2, 8)
DEFINE_HF_DPAS_TALL(hf_dpas_m128_n16_wg8_p1, 128, 1, 8, 2, 1)
DEFINE_HF_DPAS_TALL(hf_dpas_m128_n16_p1,  128, 1, 16, 2, 1)
DEFINE_HF_DPAS_TALL(hf_dpas_m128_n16_p4,  128, 1, 16, 2, 4)
DEFINE_HF_DPAS_TALL_GSK(hf_dpas_m128_n16_gsk2, 128, 1, 16, 2, 8, 2)

// =========================================================================
// 5. HARDENED OPENCL BF16 DPAS BASELINE (PREDICATED SMALL-M BARRIER!)
// =========================================================================
#define DEFINE_BF16_DPAS_TALL(NAME, SG_M, WG_M, WG_N, PF, PANEL_N)                               \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const ushort* A_bf16,                                                                 \
    global const ushort* B_bf16,                                                                 \
    global ushort*       C_bf16,                                                                 \
    const int M, const int N, const int K                                                        \
) {                                                                                              \
    const int lane = get_sub_group_local_id();                                                   \
    const int sg = get_sub_group_id();                                                           \
    int gx = get_group_id(0), gy = get_group_id(1);                                              \
    if (PANEL_N > 1) {                                                                           \
        const int nx = get_num_groups(0), ny = get_num_groups(1);                                \
        const int flat = gy * nx + gx;                                                           \
        const int blk = flat / ((PANEL_N) * ny);                                                 \
        const int rem = flat % ((PANEL_N) * ny);                                                 \
        const int x0 = blk * (PANEL_N);                                                          \
        const int pw = min((PANEL_N), nx - x0);                                                  \
        gy = rem / pw; gx = x0 + (rem % pw);                                                     \
    }                                                                                            \
    const int m0 = (gy * (WG_M) + (sg % (WG_M))) * (SG_M);                                       \
    const int n0 = (gx * (WG_N) + (sg / (WG_M))) * 16;                                           \
    if (m0 >= M || n0 >= N) return;                                                              \
    float8 acc[(SG_M) / 8];                                                                      \
    _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++) acc[i] = (float8)(0.0f);              \
    for (int k = 0; k < K; k += 32) {                                                            \
        if (PF > 0 && k + 32 * PF < K) {                                                         \
            intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void*)B_bf16, N*2, K, N*2, (int2)(n0, k + 32*PF)); \
            intel_sub_group_2d_block_prefetch_16b_16r16x1c((global void*)B_bf16, N*2, K, N*2, (int2)(n0, k + 32*PF + 16)); \
            _Pragma("unroll") for (int r = 0; r < (SG_M) / 32; r++)                              \
                intel_sub_group_2d_block_prefetch_16b_32r16x2c((global void*)A_bf16, K*2, M, K*2, (int2)(k + 32*PF, m0 + 32*r)); \
        }                                                                                        \
        uint bb0[8], bb1[8];                                                                     \
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)B_bf16, N*2, K, N*2, (int2)(n0, k),      bb0); \
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)B_bf16, N*2, K, N*2, (int2)(n0, k + 16), bb1); \
        const int8 bv0 = as_int8(vload8(0, bb0));                                                \
        const int8 bv1 = as_int8(vload8(0, bb1));                                                \
        _Pragma("unroll") for (int r = 0; r < (SG_M) / 32; r++) {                                \
            ushort ab[64];                                                                       \
            intel_sub_group_2d_block_read_16b_32r16x2c((global void*)A_bf16, K*2, M, K*2, (int2)(k, m0 + 32*r), ab); \
            _Pragma("unroll") for (int q = 0; q < 4; q++) {                                      \
                acc[r*4+q] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(0, &ab[q*8])),      bv0, acc[r*4+q]); \
                acc[r*4+q] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(0, &ab[32 + q*8])), bv1, acc[r*4+q]); \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    const int col = n0 + lane;                                                                   \
    _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++) {                                     \
        ushort8 out_bf;                                                                          \
        _Pragma("unroll") for (int r = 0; r < 8; r++) out_bf[r] = intel_convert_bfloat16_as_ushort(acc[i][r]); \
        if (n0 + 16 <= N && m0 + 8 * i + 8 <= M) {                                               \
            intel_sub_group_2d_block_write_16b_8r16x1c((global void*)C_bf16, N*2, M, N*2, (int2)(n0, m0 + 8*i), (private ushort*)&out_bf); \
        } else if (col < N) {                                                                    \
            _Pragma("unroll") for (int r = 0; r < 8; r++) {                                      \
                const int row = m0 + 8 * i + r;                                                  \
                if (row < M) C_bf16[(long)row * N + col] = out_bf[r];                            \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

#define DEFINE_BF16_SMALLM(NAME, SG_M, K_SPLIT, WG_N)                                            \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const ushort* A_bf16,                                                                 \
    global const ushort* B_bf16,                                                                 \
    global ushort*       C_bf16,                                                                 \
    const int M, const int N, const int K                                                        \
) {                                                                                              \
    local float slm_acc[K_SPLIT][WG_N][SG_M][16];                                                \
    const int lane = get_sub_group_local_id();                                                   \
    const int sg = get_sub_group_id();                                                           \
    const int ks = sg % (K_SPLIT);                                                               \
    const int wn = sg / (K_SPLIT);                                                               \
    const int m0 = get_group_id(1) * (SG_M);                                                     \
    const int n0 = (get_group_id(0) * (WG_N) + wn) * 16;                                         \
    const bool active = (m0 < M && n0 < N);                                                      \
    float8 acc[(SG_M) / 8];                                                                      \
    _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++) acc[i] = (float8)(0.0f);              \
    if (active) {                                                                                \
        for (int k = ks * 32; k < K; k += (K_SPLIT) * 32) {                                      \
            uint bb0[8], bb1[8];                                                                 \
            intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)B_bf16, N*2, K, N*2, (int2)(n0, k),      bb0); \
            intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void*)B_bf16, N*2, K, N*2, (int2)(n0, k + 16), bb1); \
            const int8 bv0 = as_int8(vload8(0, bb0));                                            \
            const int8 bv1 = as_int8(vload8(0, bb1));                                            \
            _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++) {                             \
                ushort ab[16];                                                                   \
                intel_sub_group_2d_block_read_16b_8r16x2c((global void*)A_bf16, K*2, M, K*2, (int2)(k, m0 + 8*i), ab); \
                acc[i] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(0, &ab[0])), bv0, acc[i]); \
                acc[i] = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(vload8(0, &ab[8])), bv1, acc[i]); \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++)                                       \
        _Pragma("unroll") for (int r = 0; r < 8; r++) slm_acc[ks][wn][8*i+r][lane] = acc[i][r];  \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
    if (ks == 0 && active) {                                                                     \
        const int col = n0 + lane;                                                               \
        _Pragma("unroll") for (int i = 0; i < (SG_M) / 8; i++) {                                 \
            ushort8 out_bf;                                                                      \
            _Pragma("unroll") for (int r = 0; r < 8; r++) {                                      \
                float s = 0.0f;                                                                  \
                _Pragma("unroll") for (int k = 0; k < (K_SPLIT); k++) s += slm_acc[k][wn][8*i+r][lane]; \
                out_bf[r] = intel_convert_bfloat16_as_ushort(s);                                 \
            }                                                                                    \
            if (n0 + 16 <= N && m0 + 8 * i + 8 <= M) {                                           \
                intel_sub_group_2d_block_write_16b_8r16x1c((global void*)C_bf16, N*2, M, N*2, (int2)(n0, m0 + 8*i), (private ushort*)&out_bf); \
            } else if (col < N) {                                                                \
                _Pragma("unroll") for (int r = 0; r < 8; r++) {                                  \
                    const int row = m0 + 8 * i + r;                                              \
                    if (row < M) C_bf16[(long)row * N + col] = out_bf[r];                        \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
}

DEFINE_BF16_SMALLM(bf16_m8_sk4,    8, 4, 2)
DEFINE_BF16_SMALLM(bf16_m16_sk4,  16, 4, 2)
DEFINE_BF16_DPAS_TALL(bf16_m32_n16,   32, 1, 8, 2, 8)
DEFINE_BF16_DPAS_TALL(bf16_m128_n16, 128, 1, 16, 2, 8)

// =========================================================================
// 6. HARDENED M=1..4 FUSED IN-REGISTER FWHT-128 + HF-INT4-G16 DECODE
// =========================================================================
#define HF_DECODE_BODY(M_VAL, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT, TO_OUT)                       \
    const int n0 = get_group_id(0) * 32;                                                            \
    const int sg = get_sub_group_id();                                                              \
    const int lane = get_sub_group_local_id();                                                      \
    if (n0 >= N) return;                                                                            \
    const bool has1 = (n0 + 16 < N);                                                                \
                                                                                                    \
    const int n_blk = K >> (BLK_SHIFT);                                                             \
    float acc0[M_VAL], acc1[M_VAL];                                                                 \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) { acc0[m] = 0.0f; acc1[m] = 0.0f; }           \
                                                                                                    \
    for (int blk = sg; blk < n_blk; blk += 8) {                                                     \
        const int base = blk << (BLK_SHIFT);                                                        \
        float xv[M_VAL][NJ];                                                                        \
        float sgn[NJ];                                                                              \
        _Pragma("unroll") for (int j = 0; j < (NJ); j++) sgn[j] = (float)Signs[base + j * 16 + lane]; \
        _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                         \
            VEC_T xraw = BLOCK_READ(X_base + (long)m * K + base);                                   \
            _Pragma("unroll") for (int j = 0; j < (NJ); j++)                                        \
                xv[m][j] = intel_convert_as_bfloat16_float(xraw[j]) * sgn[j];                       \
            FWHT(xv[m], lane);                                                                      \
        }                                                                                           \
        _Pragma("unroll") for (int h = 0; h < (NJ) / 4; h++) {                                      \
            const int kb = blk * ((NJ) / 4) + h;                                                    \
            uint w0[8], w1[8];                                                                      \
            intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_base, N*4, K>>3, N*4, (int2)(n0,      kb*8), w0); \
            if (has1) {                                                                             \
                intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_base, N*4, K>>3, N*4, (int2)(n0 + 16, kb*8), w1); \
            } else {                                                                                \
                _Pragma("unroll") for (int j = 0; j < 8; j++) w1[j] = 0u;                           \
            }                                                                                       \
            const uint u4_0 = load_u4_safe(U_base, kb, N, n0, lane);                                \
            const uint u4_1 = has1 ? load_u4_safe(U_base, kb, N, n0 + 16, lane) : 0u;               \
            _Pragma("unroll") for (int g4 = 0; g4 < 4; g4++) {                                      \
                const float ug0 = (float)(((u4_0 >> (4 * g4)) & 0xFu) + 1u);                        \
                const float ug1 = (float)(((u4_1 >> (4 * g4)) & 0xFu) + 1u);                        \
                const uint wa0 = w0[2 * g4], wb0 = w0[2 * g4 + 1];                                  \
                const uint wa1 = w1[2 * g4], wb1 = w1[2 * g4 + 1];                                  \
                _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                 \
                    const float xl = xv[m][h * 4 + g4];                                             \
                    const float xsum = sub_group_reduce_add(xl);                                    \
                    float s0 = -8.0f * xsum, s1 = -8.0f * xsum;                                     \
                    float xb;                                                                       \
                    xb = sub_group_broadcast(xl,  0); s0 = fma(xb, (float)( wa0        & 0xFu), s0); s1 = fma(xb, (float)( wa1        & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  1); s0 = fma(xb, (float)((wa0 >>  8) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >>  8) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  2); s0 = fma(xb, (float)((wa0 >> 16) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 16) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  3); s0 = fma(xb, (float)((wa0 >> 24) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 24) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  4); s0 = fma(xb, (float)((wa0 >>  4) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >>  4) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  5); s0 = fma(xb, (float)((wa0 >> 12) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 12) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  6); s0 = fma(xb, (float)((wa0 >> 20) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 20) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  7); s0 = fma(xb, (float)((wa0 >> 28) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 28) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  8); s0 = fma(xb, (float)( wb0        & 0xFu), s0); s1 = fma(xb, (float)( wb1        & 0xFu), s1); \
                    xb = sub_group_broadcast(xl,  9); s0 = fma(xb, (float)((wb0 >>  8) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >>  8) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl, 10); s0 = fma(xb, (float)((wb0 >> 16) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 16) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl, 11); s0 = fma(xb, (float)((wb0 >> 24) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 24) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl, 12); s0 = fma(xb, (float)((wb0 >>  4) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >>  4) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl, 13); s0 = fma(xb, (float)((wb0 >> 12) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 12) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl, 14); s0 = fma(xb, (float)((wb0 >> 20) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 20) & 0xFu), s1); \
                    xb = sub_group_broadcast(xl, 15); s0 = fma(xb, (float)((wb0 >> 28) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 28) & 0xFu), s1); \
                    acc0[m] = fma(s0, ug0, acc0[m]);                                                \
                    acc1[m] = fma(s1, ug1, acc1[m]);                                                \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    }                                                                                               \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                             \
        slm_part[sg][m][0][lane] = acc0[m];                                                         \
        slm_part[sg][m][1][lane] = acc1[m];                                                         \
    }                                                                                               \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                   \
    if (sg == 0) {                                                                                  \
        const int c0 = n0 + lane, c1 = n0 + 16 + lane;                                              \
        const float d0 = (c0 < N) ? D_base[c0] : 0.0f, b0 = (B_base != 0 && c0 < N) ? B_base[c0] : -0.0f; \
        const float d1 = (c1 < N) ? D_base[c1] : 0.0f, b1 = (B_base != 0 && c1 < N) ? B_base[c1] : -0.0f; \
        _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                         \
            float t0 = 0.0f, t1 = 0.0f;                                                             \
            _Pragma("unroll") for (int s = 0; s < 8; s++) {                                         \
                t0 += slm_part[s][m][0][lane]; t1 += slm_part[s][m][1][lane];                       \
            }                                                                                       \
            if (c0 < N) C_base[(long)m * N + c0] = TO_OUT(fma(t0, d0, b0));                         \
            if (c1 < N) C_base[(long)m * N + c1] = TO_OUT(fma(t1, d1, b1));                         \
        }                                                                                           \
    }

#define HF_BF16_OUT(v) intel_convert_bfloat16_as_ushort(v)
#define HF_F32_OUT(v) (v)

#define DEFINE_HF_DECODE_FUSED(NAME, M_VAL, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT)                 \
__attribute__((intel_reqd_sub_group_size(16)))                                                      \
kernel void NAME(                                                                                   \
    global const ushort* X_base,                                                                    \
    global const char*   Signs,                                                                     \
    global const uint*   W_base,                                                                    \
    global const ushort* U_base,                                                                    \
    global const float*  D_base,                                                                    \
    global const float*  B_base,                                                                    \
    global ushort*       C_base,                                                                    \
    const int N, const int K                                                                        \
) {                                                                                                 \
    local float slm_part[8][M_VAL][2][16];                                                          \
    HF_DECODE_BODY(M_VAL, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT, HF_BF16_OUT)                      \
}

DEFINE_HF_DECODE_FUSED(hf_decode_fused_m1, 1, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_m2, 2, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_m4, 4, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_had64_m1, 1, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_had64_m2, 2, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_had64_m4, 4, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6)

// =========================================================================
// 6B. GROUPED ROUTED-EXPERT (MoE) W4A16 PAIR KERNEL
//     One work-group row per (token, slot) pair p = get_group_id(1).  The pair
//     reads input row p / in_div and the packed weights of expert Ids[p] from
//     3D [E, K/8, N] storage.  Routing is read on the device, so the launch
//     shape is independent of which experts were picked (graph-replay safe).
//     An expert id outside [0, E) writes zeros.  That branch is uniform for the
//     whole work-group, so no barrier is skipped by only part of it.
// =========================================================================
#define DEFINE_HF_MOE_PAIR(NAME, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT, OUT_T, TO_OUT)             \
__attribute__((intel_reqd_sub_group_size(16)))                                                      \
kernel void NAME(                                                                                   \
    global const ushort* X,                                                                         \
    global const char*   Signs,                                                                     \
    global const uint*   W,                                                                         \
    global const ushort* U,                                                                         \
    global const float*  D,                                                                         \
    global const float*  Bias,                                                                      \
    global const int*    Ids,                                                                       \
    global OUT_T*        C,                                                                         \
    const int N, const int K, const int E, const int in_div                                         \
) {                                                                                                 \
    local float slm_part[8][1][2][16];                                                              \
    const int p = get_group_id(1);                                                                  \
    const int e = Ids[p];                                                                           \
    global OUT_T* C_base = C + (long)p * N;                                                         \
    if (e < 0 || e >= E) {                                                                          \
        const int col = get_group_id(0) * 32 + get_local_id(0);                                     \
        if (get_local_id(0) < 32 && col < N) C_base[col] = (OUT_T)0;                                \
        return;                                                                                     \
    }                                                                                               \
    global const ushort* X_base = X + (long)(p / in_div) * K;                                       \
    global const uint*   W_base = W + (long)e * (K >> 3) * N;                                       \
    global const ushort* U_base = U + (long)e * (K >> 6) * N;                                       \
    global const float*  D_base = D + (long)e * N;                                                  \
    global const float*  B_base = (Bias != 0) ? Bias + (long)e * N : Bias;                          \
    HF_DECODE_BODY(1, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT, TO_OUT)                               \
}

DEFINE_HF_MOE_PAIR(hf_moe_pair_had128_bf16, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7, ushort, HF_BF16_OUT)
DEFINE_HF_MOE_PAIR(hf_moe_pair_had128_f32, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7, float, HF_F32_OUT)
DEFINE_HF_MOE_PAIR(hf_moe_pair_had64_bf16, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6, ushort, HF_BF16_OUT)
DEFINE_HF_MOE_PAIR(hf_moe_pair_had64_f32, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6, float, HF_F32_OUT)

// =========================================================================
// 6C. GROUPED-BY-EXPERT W4A16 (MoE PREFILL)
//     The same math as the pair kernel, with the routed pairs bucketed by
//     expert on the device so one work-group reads and unpacks an expert's
//     weight panel once for up to HF_MOE_TILE_ROWS token rows.  Per output
//     row the operation sequence is the pair kernel's: the same FWHT, the same
//     16-wide FMA chain, the same eight per-subgroup partials in K order, the
//     same reduction and epilogue.  Results are bit-identical to it, wherever
//     a pair lands in a tile.  Every launch shape depends only on
//     (pairs, E, N, K).  Integer atomics order the buckets; no output is ever
//     accumulated atomically.
// =========================================================================
inline int hf_moe_bucket_of(const int e, const int E) {
    return (e >= 0 && e < E) ? e : E;  // bucket E: out-of-range ids
}

kernel void hf_moe_bucket_zero(global int* Counts, const int B) {
    const int b = get_global_id(0);
    if (b < B) Counts[b] = 0;
}

kernel void hf_moe_bucket_count(global const int* Ids, volatile global int* Counts, const int E, const int P) {
    const int p = get_global_id(0);
    if (p < P) atomic_inc(&Counts[hf_moe_bucket_of(Ids[p], E)]);
}

// One work-item.  Exclusive scan in place (Counts becomes each bucket's write
// cursor) and one (bucket, start, length) descriptor per tile.
kernel void hf_moe_bucket_scan(global int* Counts, global int* Tiles, global int* NTiles,
                               const int B, const int ROWS) {
    int offset = 0, tile = 0;
    for (int b = 0; b < B; b++) {
        const int count = Counts[b];
        Counts[b] = offset;
        for (int s = 0; s < count; s += ROWS) {
            Tiles[3 * tile] = b;
            Tiles[3 * tile + 1] = offset + s;
            Tiles[3 * tile + 2] = min(ROWS, count - s);
            tile++;
        }
        offset += count;
    }
    NTiles[0] = tile;
}

kernel void hf_moe_bucket_scatter(global const int* Ids, volatile global int* Cursor, global int* Perm,
                                  const int E, const int P) {
    const int p = get_global_id(0);
    if (p < P) Perm[atomic_inc(&Cursor[hf_moe_bucket_of(Ids[p], E)])] = p;
}

// One work-group per activation row: rotate it once, in FP32, for every slot
// and every output column that will read it.
#define DEFINE_HF_MOE_ROTATE(NAME, NJ, VEC_T, BLOCK_READ, FWHT, BLK_SHIFT)                          \
__attribute__((intel_reqd_sub_group_size(16)))                                                      \
kernel void NAME(global const ushort* X, global const char* Signs, global float* Xr, const int K) { \
    const int row = get_group_id(1);                                                                \
    const int sg = get_sub_group_id();                                                              \
    const int lane = get_sub_group_local_id();                                                      \
    const int n_blk = K >> (BLK_SHIFT);                                                             \
    for (int blk = sg; blk < n_blk; blk += 8) {                                                     \
        const int base = blk << (BLK_SHIFT);                                                        \
        VEC_T xraw = BLOCK_READ(X + (long)row * K + base);                                          \
        float v[NJ];                                                                                \
        _Pragma("unroll") for (int j = 0; j < (NJ); j++)                                            \
            v[j] = intel_convert_as_bfloat16_float(xraw[j]) * (float)Signs[base + j * 16 + lane];   \
        FWHT(v, lane);                                                                              \
        _Pragma("unroll") for (int j = 0; j < (NJ); j++)                                            \
            Xr[(long)row * K + base + j * 16 + lane] = v[j];                                        \
    }                                                                                               \
}

DEFINE_HF_MOE_ROTATE(hf_moe_rotate_had128, 8, ushort8, intel_sub_group_block_read_us8, fwht128_sg16, 7)
DEFINE_HF_MOE_ROTATE(hf_moe_rotate_had64, 4, ushort4, intel_sub_group_block_read_us4, fwht64_sg16, 6)

// Tile body for a row class of M_VAL (the tile holds `len` <= M_VAL real rows;
// padding rows repeat the last one and are never stored).  The expert's Q and
// U panel for 64 input positions is read once, its nibbles are expanded to
// FP32 once per 16 positions, and both are applied to all M_VAL rows.
#define HF_MOE_TILE_BODY(M_VAL, NJ, BLK_SHIFT, OUT_ELEM, TO_OUT)                                    \
    {                                                                                               \
    int pr[M_VAL];                                                                                  \
    long xoff[M_VAL];                                                                               \
    float acc0[M_VAL], acc1[M_VAL];                                                                 \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                             \
        pr[m] = Perm[start + min(m, len - 1)];                                                      \
        xoff[m] = (long)(pr[m] / in_div) * K;                                                       \
        acc0[m] = 0.0f; acc1[m] = 0.0f;                                                             \
    }                                                                                               \
    for (int blk = sg; blk < n_blk; blk += 8) {                                                     \
        const int base = blk << (BLK_SHIFT);                                                        \
        _Pragma("unroll") for (int h = 0; h < (NJ) / 4; h++) {                                      \
            const int kb = blk * ((NJ) / 4) + h;                                                    \
            uint w0[8], w1[8];                                                                      \
            intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_base, N*4, K>>3, N*4, (int2)(n0,      kb*8), w0); \
            if (has1) {                                                                             \
                intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_base, N*4, K>>3, N*4, (int2)(n0 + 16, kb*8), w1); \
            } else {                                                                                \
                _Pragma("unroll") for (int j = 0; j < 8; j++) w1[j] = 0u;                           \
            }                                                                                       \
            const uint u4_0 = load_u4_safe(U_base, kb, N, n0, lane);                                \
            const uint u4_1 = has1 ? load_u4_safe(U_base, kb, N, n0 + 16, lane) : 0u;               \
            _Pragma("unroll") for (int g4 = 0; g4 < 4; g4++) {                                      \
                const float ug0 = (float)(((u4_0 >> (4 * g4)) & 0xFu) + 1u);                        \
                const float ug1 = (float)(((u4_1 >> (4 * g4)) & 0xFu) + 1u);                        \
                const uint wa0 = w0[2 * g4], wb0 = w0[2 * g4 + 1];                                  \
                const uint wa1 = w1[2 * g4], wb1 = w1[2 * g4 + 1];                                  \
                float wf0[16], wf1[16];                                                             \
                HF_MOE_UNPACK16(wf0, wa0, wb0)                                                      \
                HF_MOE_UNPACK16(wf1, wa1, wb1)                                                      \
                const int at = base + (h * 4 + g4) * 16;                                            \
                _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                 \
                    const float xl = as_float(intel_sub_group_block_read(                           \
                        (const global uint*)(Xr + xoff[m] + at)));                                  \
                    const float xsum = sub_group_reduce_add(xl);                                    \
                    float s0 = -8.0f * xsum, s1 = -8.0f * xsum;                                     \
                    float xb;                                                                       \
                    HF_MOE_FMA(0)  HF_MOE_FMA(1)  HF_MOE_FMA(2)  HF_MOE_FMA(3)                      \
                    HF_MOE_FMA(4)  HF_MOE_FMA(5)  HF_MOE_FMA(6)  HF_MOE_FMA(7)                      \
                    HF_MOE_FMA(8)  HF_MOE_FMA(9)  HF_MOE_FMA(10) HF_MOE_FMA(11)                     \
                    HF_MOE_FMA(12) HF_MOE_FMA(13) HF_MOE_FMA(14) HF_MOE_FMA(15)                     \
                    acc0[m] = fma(s0, ug0, acc0[m]);                                                \
                    acc1[m] = fma(s1, ug1, acc1[m]);                                                \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    }                                                                                               \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                             \
        slm_part[sg][m][0][lane] = acc0[m];                                                         \
        slm_part[sg][m][1][lane] = acc1[m];                                                         \
    }                                                                                               \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                   \
    if (sg == 0) {                                                                                  \
        const int c0 = n0 + lane, c1 = n0 + 16 + lane;                                              \
        const float d0 = (c0 < N) ? D_base[c0] : 0.0f, b0 = (B_base != 0 && c0 < N) ? B_base[c0] : -0.0f; \
        const float d1 = (c1 < N) ? D_base[c1] : 0.0f, b1 = (B_base != 0 && c1 < N) ? B_base[c1] : -0.0f; \
        for (int m = 0; m < len; m++) {                                                             \
            float t0 = 0.0f, t1 = 0.0f;                                                             \
            _Pragma("unroll") for (int s = 0; s < 8; s++) {                                         \
                t0 += slm_part[s][m][0][lane]; t1 += slm_part[s][m][1][lane];                       \
            }                                                                                       \
            global OUT_ELEM* C_base = C + (long)Perm[start + m] * N;                                \
            if (c0 < N) C_base[c0] = TO_OUT(fma(t0, d0, b0));                                       \
            if (c1 < N) C_base[c1] = TO_OUT(fma(t1, d1, b1));                                       \
        }                                                                                           \
    }                                                                                               \
    }

// Position i of a 16-wide group, in the interleaved nibble order of q_pack.
#define HF_MOE_UNPACK16(F, A, B)                                                                    \
    F[0]  = (float)( A        & 0xFu); F[1]  = (float)((A >>  8) & 0xFu);                           \
    F[2]  = (float)((A >> 16) & 0xFu); F[3]  = (float)((A >> 24) & 0xFu);                           \
    F[4]  = (float)((A >>  4) & 0xFu); F[5]  = (float)((A >> 12) & 0xFu);                           \
    F[6]  = (float)((A >> 20) & 0xFu); F[7]  = (float)((A >> 28) & 0xFu);                           \
    F[8]  = (float)( B        & 0xFu); F[9]  = (float)((B >>  8) & 0xFu);                           \
    F[10] = (float)((B >> 16) & 0xFu); F[11] = (float)((B >> 24) & 0xFu);                           \
    F[12] = (float)((B >>  4) & 0xFu); F[13] = (float)((B >> 12) & 0xFu);                           \
    F[14] = (float)((B >> 20) & 0xFu); F[15] = (float)((B >> 28) & 0xFu);

#define HF_MOE_FMA(I) xb = sub_group_broadcast(xl, I); s0 = fma(xb, wf0[I], s0); s1 = fma(xb, wf1[I], s1);

// Grid: (ceil(N/32), max_tiles).  Tiles past the device-side count return.
// Both early exits are uniform for the whole work-group.
#define DEFINE_HF_MOE_TILE(NAME, NJ, BLK_SHIFT, OUT_T, TO_OUT)                                      \
__attribute__((intel_reqd_sub_group_size(16)))                                                      \
kernel void NAME(                                                                                   \
    global const float*  Xr,                                                                        \
    global const uint*   W,                                                                         \
    global const ushort* U,                                                                         \
    global const float*  D,                                                                         \
    global const float*  Bias,                                                                      \
    global const int*    Perm,                                                                      \
    global const int*    Tiles,                                                                     \
    global const int*    NTiles,                                                                    \
    global OUT_T*        C,                                                                         \
    const int N, const int K, const int E, const int in_div                                         \
) {                                                                                                 \
    local float slm_part[8][16][2][16];                                                             \
    const int tile = get_group_id(1);                                                               \
    if (tile >= NTiles[0]) return;                                                                  \
    const int e = Tiles[3 * tile], start = Tiles[3 * tile + 1], len = Tiles[3 * tile + 2];          \
    const int n0 = get_group_id(0) * 32;                                                            \
    const int sg = get_sub_group_id();                                                              \
    const int lane = get_sub_group_local_id();                                                      \
    if (n0 >= N) return;                                                                            \
    if (e >= E) {                                                                                   \
        const int col = n0 + get_local_id(0);                                                       \
        if (get_local_id(0) < 32 && col < N)                                                        \
            for (int m = 0; m < len; m++) C[(long)Perm[start + m] * N + col] = (OUT_T)0;            \
        return;                                                                                     \
    }                                                                                               \
    const bool has1 = (n0 + 16 < N);                                                                \
    const int n_blk = K >> (BLK_SHIFT);                                                             \
    global const uint*   W_base = W + (long)e * (K >> 3) * N;                                       \
    global const ushort* U_base = U + (long)e * (K >> 6) * N;                                       \
    global const float*  D_base = D + (long)e * N;                                                  \
    global const float*  B_base = (Bias != 0) ? Bias + (long)e * N : Bias;                          \
    if (len > 8) {                                                                                  \
        HF_MOE_TILE_BODY(16, NJ, BLK_SHIFT, OUT_T, TO_OUT)                                         \
    } else if (len > 4) {                                                                           \
        HF_MOE_TILE_BODY(8, NJ, BLK_SHIFT, OUT_T, TO_OUT)                                          \
    } else if (len > 2) {                                                                           \
        HF_MOE_TILE_BODY(4, NJ, BLK_SHIFT, OUT_T, TO_OUT)                                          \
    } else if (len > 1) {                                                                           \
        HF_MOE_TILE_BODY(2, NJ, BLK_SHIFT, OUT_T, TO_OUT)                                          \
    } else {                                                                                        \
        HF_MOE_TILE_BODY(1, NJ, BLK_SHIFT, OUT_T, TO_OUT)                                          \
    }                                                                                               \
}

DEFINE_HF_MOE_TILE(hf_moe_tile_had128_bf16, 8, 7, ushort, HF_BF16_OUT)
DEFINE_HF_MOE_TILE(hf_moe_tile_had128_f32, 8, 7, float, HF_F32_OUT)
DEFINE_HF_MOE_TILE(hf_moe_tile_had64_bf16, 4, 6, ushort, HF_BF16_OUT)
DEFINE_HF_MOE_TILE(hf_moe_tile_had64_f32, 4, 6, float, HF_F32_OUT)

// =========================================================================
// 7. HARDENED MLX-ASYM-G64 M=1..4 DECODE BASELINE
// =========================================================================
#define DEFINE_MLX_DECODE(NAME, M_VAL)                                                           \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const ushort* X_bf16,                                                                 \
    global const uint*   W_mlx,                                                                  \
    global const half*   S_mlx,                                                                  \
    global const half*   B_mlx,                                                                  \
    global ushort*       C_bf16,                                                                 \
    const int N, const int K                                                                     \
) {                                                                                              \
    local float slm_part[8][M_VAL][2][16];                                                       \
    const int n0 = get_group_id(0) * 32;                                                         \
    const int sg = get_sub_group_id();                                                           \
    const int lane = get_sub_group_local_id();                                                   \
    if (n0 >= N) return;                                                                         \
    const bool has1 = (n0 + 16 < N);                                                             \
    const int n_kb = K >> 6;                                                                     \
    float acc0[M_VAL], acc1[M_VAL];                                                              \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) { acc0[m] = 0.0f; acc1[m] = 0.0f; }        \
    for (int kb = sg; kb < n_kb; kb += 8) {                                                      \
        const int kbase = kb << 6;                                                               \
        uint w0[8], w1[8];                                                                       \
        intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_mlx, N*4, K>>3, N*4, (int2)(n0, kb*8), w0); \
        if (has1) {                                                                              \
            intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_mlx, N*4, K>>3, N*4, (int2)(n0 + 16, kb*8), w1); \
        } else {                                                                                 \
            _Pragma("unroll") for (int j = 0; j < 8; j++) w1[j] = 0u;                            \
        }                                                                                        \
        const int c0 = n0 + lane, c1 = n0 + 16 + lane;                                           \
        const float sc0 = (c0 < N) ? (float)S_mlx[(long)kb * N + c0] : 0.0f;                     \
        const float bi0 = (c0 < N) ? (float)B_mlx[(long)kb * N + c0] : 0.0f;                     \
        const float sc1 = (c1 < N) ? (float)S_mlx[(long)kb * N + c1] : 0.0f;                     \
        const float bi1 = (c1 < N) ? (float)B_mlx[(long)kb * N + c1] : 0.0f;                     \
        _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                      \
            float s0 = 0.0f, s1 = 0.0f, xsum = 0.0f;                                             \
            _Pragma("unroll") for (int g4 = 0; g4 < 4; g4++) {                                   \
                float xl = intel_convert_as_bfloat16_float(X_bf16[(long)m * K + kbase + g4 * 16 + lane]); \
                xsum += sub_group_reduce_add(xl);                                                \
                const uint wa0 = w0[2 * g4], wb0 = w0[2 * g4 + 1];                               \
                const uint wa1 = w1[2 * g4], wb1 = w1[2 * g4 + 1];                               \
                float xb;                                                                        \
                xb = sub_group_broadcast(xl,  0); s0 = fma(xb, (float)( wa0        & 0xFu), s0); s1 = fma(xb, (float)( wa1        & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  1); s0 = fma(xb, (float)((wa0 >>  8) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >>  8) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  2); s0 = fma(xb, (float)((wa0 >> 16) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 16) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  3); s0 = fma(xb, (float)((wa0 >> 24) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 24) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  4); s0 = fma(xb, (float)((wa0 >>  4) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >>  4) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  5); s0 = fma(xb, (float)((wa0 >> 12) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 12) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  6); s0 = fma(xb, (float)((wa0 >> 20) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 20) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  7); s0 = fma(xb, (float)((wa0 >> 28) & 0xFu), s0); s1 = fma(xb, (float)((wa1 >> 28) & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  8); s0 = fma(xb, (float)( wb0        & 0xFu), s0); s1 = fma(xb, (float)( wb1        & 0xFu), s1); \
                xb = sub_group_broadcast(xl,  9); s0 = fma(xb, (float)((wb0 >>  8) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >>  8) & 0xFu), s1); \
                xb = sub_group_broadcast(xl, 10); s0 = fma(xb, (float)((wb0 >> 16) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 16) & 0xFu), s1); \
                xb = sub_group_broadcast(xl, 11); s0 = fma(xb, (float)((wb0 >> 24) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 24) & 0xFu), s1); \
                xb = sub_group_broadcast(xl, 12); s0 = fma(xb, (float)((wb0 >>  4) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >>  4) & 0xFu), s1); \
                xb = sub_group_broadcast(xl, 13); s0 = fma(xb, (float)((wb0 >> 12) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 12) & 0xFu), s1); \
                xb = sub_group_broadcast(xl, 14); s0 = fma(xb, (float)((wb0 >> 20) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 20) & 0xFu), s1); \
                xb = sub_group_broadcast(xl, 15); s0 = fma(xb, (float)((wb0 >> 28) & 0xFu), s0); s1 = fma(xb, (float)((wb1 >> 28) & 0xFu), s1); \
            }                                                                                    \
            acc0[m] = fma(s0, sc0, fma(xsum, bi0, acc0[m]));                                     \
            acc1[m] = fma(s1, sc1, fma(xsum, bi1, acc1[m]));                                     \
        }                                                                                        \
    }                                                                                            \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                          \
        slm_part[sg][m][0][lane] = acc0[m];                                                      \
        slm_part[sg][m][1][lane] = acc1[m];                                                      \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
    if (sg == 0) {                                                                               \
        const int c0 = n0 + lane, c1 = n0 + 16 + lane;                                           \
        _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                      \
            float t0 = 0.0f, t1 = 0.0f;                                                          \
            _Pragma("unroll") for (int s = 0; s < 8; s++) {                                      \
                t0 += slm_part[s][m][0][lane]; t1 += slm_part[s][m][1][lane];                    \
            }                                                                                    \
            if (c0 < N) C_bf16[(long)m * N + c0] = intel_convert_bfloat16_as_ushort(t0);         \
            if (c1 < N) C_bf16[(long)m * N + c1] = intel_convert_bfloat16_as_ushort(t1);         \
        }                                                                                        \
    }                                                                                            \
}

DEFINE_MLX_DECODE(mlx_decode_m1, 1)
DEFINE_MLX_DECODE(mlx_decode_m2, 2)
DEFINE_MLX_DECODE(mlx_decode_m4, 4)
