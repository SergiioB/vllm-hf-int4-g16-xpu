
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

// =========================================================================
// 2. SINGLE-PASS FUSED ContigFWHT128 + Per-Token INT8 ActQ
// =========================================================================
__attribute__((intel_reqd_sub_group_size(16)))
kernel void fused_fwht128_actq(
    global const ushort* X_bf16,
    global const char*   Signs,
    global char*         X_i8,
    global float*        Sx,
    const int K
) {
    local float sg_max[16];

    const int m = get_group_id(0);
    const int lid = get_local_id(0);
    const int sg = lid >> 4;
    const int lane = lid & 15;
    const int nblocks = K >> 7;

    global const ushort* row_in = X_bf16 + (long)m * K;
    float local_max = 1e-12f;

    for (int blk = sg; blk < nblocks; blk += 16) {
        const int base = blk << 7;
        ushort8 raw = intel_sub_group_block_read_us8(row_in + base);
        float v[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int idx = base + j * 16 + lane;
            v[j] = intel_convert_as_bfloat16_float(raw[j]) * (float)Signs[idx];
        }
        fwht128_sg16(v, lane);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            local_max = fmax(local_max, fabs(v[j]));
        }
    }

    float sgm = sub_group_reduce_max(local_max);
    if (lane == 0) sg_max[sg] = sgm;
    barrier(CLK_LOCAL_MEM_FENCE);

    float rmax = sub_group_reduce_max(sg_max[lane]);
    const float scale = rmax / 127.0f;
    const float inv_scale = 127.0f / rmax;
    if (lid == 0) Sx[m] = scale;

    global char* row_out = X_i8 + (long)m * K;
    // Pass 2: re-read the bf16 input and re-run the exact in-register FP32
    // butterfly.  No FP16 staging exists anywhere in ACTQ: large finite values
    // cannot overflow to inf, ties reach rint in FP32, and the kernel needs no
    // 34-68 KiB row SLM allocation at all.
    for (int blk = sg; blk < nblocks; blk += 16) {
        const int base = blk << 7;
        ushort8 rawq = intel_sub_group_block_read_us8(row_in + base);
        float vq[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int idx = base + j * 16 + lane;
            vq[j] = intel_convert_as_bfloat16_float(rawq[j]) * (float)Signs[idx];
        }
        fwht128_sg16(vq, lane);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int idx = base + j * 16 + lane;
            float q = clamp(rint(vq[j] * inv_scale), -127.0f, 127.0f);
            row_out[idx] = (char)((int)q);
        }
    }
}

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
#define DEFINE_HF_DECODE_FUSED(NAME, M_VAL)                                                      \
__attribute__((intel_reqd_sub_group_size(16)))                                                   \
kernel void NAME(                                                                                \
    global const ushort* X_raw_bf16,                                                             \
    global const char*   Signs,                                                                  \
    global const uint*   W_packed,                                                               \
    global const ushort* U_packed,                                                               \
    global const float*  D_row,                                                                  \
    global const float*  Bias,                                                                   \
    global ushort*       C_bf16,                                                                 \
    const int N, const int K                                                                     \
) {                                                                                              \
    local float slm_part[8][M_VAL][2][16];                                                       \
    const int n0 = get_group_id(0) * 32;                                                         \
    const int sg = get_sub_group_id();                                                           \
    const int lane = get_sub_group_local_id();                                                   \
    if (n0 >= N) return;                                                                         \
    const bool has1 = (n0 + 16 < N);                                                             \
                                                                                                 \
    const int n_b128 = K >> 7;                                                                   \
    float acc0[M_VAL], acc1[M_VAL];                                                              \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) { acc0[m] = 0.0f; acc1[m] = 0.0f; }        \
                                                                                                 \
    for (int b128 = sg; b128 < n_b128; b128 += 8) {                                              \
        const int base = b128 << 7;                                                              \
        float xv[M_VAL][8];                                                                      \
        float sgn[8];                                                                            \
        _Pragma("unroll") for (int j = 0; j < 8; j++) sgn[j] = (float)Signs[base + j * 16 + lane]; \
        _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                      \
            ushort8 xraw = intel_sub_group_block_read_us8(X_raw_bf16 + (long)m * K + base);      \
            _Pragma("unroll") for (int j = 0; j < 8; j++)                                        \
                xv[m][j] = intel_convert_as_bfloat16_float(xraw[j]) * sgn[j];                    \
            fwht128_sg16(xv[m], lane);                                                           \
        }                                                                                        \
        _Pragma("unroll") for (int h = 0; h < 2; h++) {                                          \
            const int kb = b128 * 2 + h;                                                         \
            uint w0[8], w1[8];                                                                   \
            intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_packed, N*4, K>>3, N*4, (int2)(n0,      kb*8), w0); \
            if (has1) {                                                                          \
                intel_sub_group_2d_block_read_32b_8r16x1c((global void*)W_packed, N*4, K>>3, N*4, (int2)(n0 + 16, kb*8), w1); \
            } else {                                                                             \
                _Pragma("unroll") for (int j = 0; j < 8; j++) w1[j] = 0u;                        \
            }                                                                                    \
            const uint u4_0 = load_u4_safe(U_packed, kb, N, n0, lane);                           \
            const uint u4_1 = has1 ? load_u4_safe(U_packed, kb, N, n0 + 16, lane) : 0u;          \
            _Pragma("unroll") for (int g4 = 0; g4 < 4; g4++) {                                   \
                const float ug0 = (float)(((u4_0 >> (4 * g4)) & 0xFu) + 1u);                     \
                const float ug1 = (float)(((u4_1 >> (4 * g4)) & 0xFu) + 1u);                     \
                const uint wa0 = w0[2 * g4], wb0 = w0[2 * g4 + 1];                               \
                const uint wa1 = w1[2 * g4], wb1 = w1[2 * g4 + 1];                               \
                _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                              \
                    const float xl = xv[m][h * 4 + g4];                                          \
                    const float xsum = sub_group_reduce_add(xl);                                 \
                    float s0 = -8.0f * xsum, s1 = -8.0f * xsum;                                  \
                    float xb;                                                                    \
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
                    acc0[m] = fma(s0, ug0, acc0[m]);                                             \
                    acc1[m] = fma(s1, ug1, acc1[m]);                                             \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                          \
        slm_part[sg][m][0][lane] = acc0[m];                                                      \
        slm_part[sg][m][1][lane] = acc1[m];                                                      \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
    if (sg == 0) {                                                                               \
        const int c0 = n0 + lane, c1 = n0 + 16 + lane;                                           \
        const float d0 = (c0 < N) ? D_row[c0] : 0.0f, b0 = (Bias != 0 && c0 < N) ? Bias[c0] : 0.0f;           \
        const float d1 = (c1 < N) ? D_row[c1] : 0.0f, b1 = (Bias != 0 && c1 < N) ? Bias[c1] : 0.0f;           \
        _Pragma("unroll") for (int m = 0; m < M_VAL; m++) {                                      \
            float t0 = 0.0f, t1 = 0.0f;                                                          \
            _Pragma("unroll") for (int s = 0; s < 8; s++) {                                      \
                t0 += slm_part[s][m][0][lane]; t1 += slm_part[s][m][1][lane];                    \
            }                                                                                    \
            if (c0 < N) C_bf16[(long)m * N + c0] = intel_convert_bfloat16_as_ushort(fma(t0, d0, b0)); \
            if (c1 < N) C_bf16[(long)m * N + c1] = intel_convert_bfloat16_as_ushort(fma(t1, d1, b1)); \
        }                                                                                        \
    }                                                                                            \
}

DEFINE_HF_DECODE_FUSED(hf_decode_fused_m1, 1)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_m2, 2)
DEFINE_HF_DECODE_FUSED(hf_decode_fused_m4, 4)

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
