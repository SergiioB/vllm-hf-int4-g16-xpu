"""Triton XPU reference implementation of HF-INT4-G16.

These kernels are the universal correctness baseline: arbitrary M, arbitrary
N tails, fp16/bf16 activations, and fused fp32 bias/bias-correction epilogue.
The Level-Zero backend is the fast path for supported Xe2 cases; this module
never silently changes the math or layout.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

from .format import had_block_for_k, make_hadamard


@triton.jit
def _decode_w_tile(Q_ptr, U_ptr, rows, cols, N, mask_k, mask_n):
    # Canonical native layout documented in format.py.
    qoff = (rows // 8)[:, None] * N + cols[None, :]
    qraw = tl.load(Q_ptr + qoff, mask=mask_k[:, None] & mask_n[None, :], other=0)
    q32 = qraw.to(tl.uint32, bitcast=True)
    qshift = ((rows % 4) * 8 + ((rows % 8) // 4) * 4)[:, None]
    q = ((q32 >> qshift) & 0xF).to(tl.int16) - 8

    uoff = (rows // 64)[:, None] * N + cols[None, :]
    uraw = tl.load(U_ptr + uoff, mask=mask_k[:, None] & mask_n[None, :], other=0)
    u16 = uraw.to(tl.uint16, bitcast=True).to(tl.uint32)
    ushift = (((rows // 16) % 4) * 4)[:, None]
    u = ((u16 >> ushift) & 0xF).to(tl.int16) + 1
    return q * u


@triton.jit
def hf_int4_g16_w4a16_kernel(
    X_ptr, Signs_ptr, H_ptr, Q_ptr, U_ptr, D_ptr, Bias_ptr, BiasCorr_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_BIAS_CORR: tl.constexpr,
):
    """Fused block-Hadamard activation rotation + W4A16 GEMM.

    BK is the Hadamard block size (128 or 64).  FP32 accumulation is used
    throughout the epilogue.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_k = tl.arange(0, BK)
    h = tl.load(H_ptr + offs_k[:, None] * BK + offs_k[None, :]).to(tl.float32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        rows = k0 + offs_k
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + rows[None, :] * stride_xk,
            mask=mask_m[:, None], other=0.0,
        ).to(tl.float32)
        s = tl.load(Signs_ptr + rows).to(tl.float32)
        xr = tl.dot(x * s[None, :], h, out_dtype=tl.float32)
        mask_k = rows < K
        wi = _decode_w_tile(Q_ptr, U_ptr, rows, offs_n, N, mask_k, mask_n).to(tl.float32)
        acc += tl.dot(xr, wi, out_dtype=tl.float32)

    d = tl.load(D_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = acc * d[None, :]
    if HAS_BIAS:
        b = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = y + b[None, :]
    if HAS_BIAS_CORR:
        bc = tl.load(BiasCorr_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = y + bc[None, :]
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y, mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _bfly_stage(x, BM: tl.constexpr, NB: tl.constexpr, S: tl.constexpr, BK: tl.constexpr):
    """One in-place Sylvester butterfly stage in the native kernel's order."""
    y = tl.reshape(x, (BM, NB, 2, S))
    y = tl.permute(y, (0, 1, 3, 2))
    a, b = tl.split(y)
    z = tl.join(a + b, a - b)
    z = tl.permute(z, (0, 1, 3, 2))
    return tl.reshape(z, (BM, BK))


@triton.jit
def _fwht_exact_fp32(x, BM: tl.constexpr, BK: tl.constexpr):
    """Bit-order-matched FP32 FWHT-64/128 used by the native OpenCL kernel.

    The stage order is 1,2,4,...,BK/2 and only the final butterfly outputs are
    multiplied by 1/sqrt(BK), exactly matching hf_int4_kernels.cl.  This avoids
    both tl.dot reduction-order differences and any FP16 staging.
    """
    x = _bfly_stage(x, BM, BK // 2, 1, BK)
    x = _bfly_stage(x, BM, BK // 4, 2, BK)
    x = _bfly_stage(x, BM, BK // 8, 4, BK)
    x = _bfly_stage(x, BM, BK // 16, 8, BK)
    x = _bfly_stage(x, BM, BK // 32, 16, BK)
    x = _bfly_stage(x, BM, BK // 64, 32, BK)
    if BK == 128:
        x = _bfly_stage(x, BM, 1, 64, BK)
        return x * 0.088388347648
    return x * 0.125


@triton.jit
def fwht_actq_kernel(
    X_ptr, Signs_ptr, OutI8_ptr, Scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BM: tl.constexpr, BK: tl.constexpr,
):
    """Two-pass fused Hadamard + per-token dynamic int8 quantization.

    Rotation and quantization are FP32 end to end.  Rounding is exact IEEE
    round-to-nearest-even through libdevice ``rint``, matching the native
    Level-Zero kernel bit-for-bit for zero rows, ties, and large finite rows.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_k = tl.arange(0, BK)
    row_amax = tl.zeros((BM,), dtype=tl.float32) + 1e-12

    for k0 in range(0, K, BK):
        rows = k0 + offs_k
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + rows[None, :] * stride_xk,
                    mask=mask_m[:, None], other=0.0).to(tl.float32)
        s = tl.load(Signs_ptr + rows).to(tl.float32)
        xr = _fwht_exact_fp32(x * s[None, :], BM, BK)
        row_amax = tl.maximum(row_amax, tl.max(tl.abs(xr), axis=1))

    scale = row_amax * (1.0 / 127.0)
    inv_scale = 127.0 / row_amax
    tl.store(Scale_ptr + offs_m, scale, mask=mask_m)

    for k0 in range(0, K, BK):
        rows = k0 + offs_k
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + rows[None, :] * stride_xk,
                    mask=mask_m[:, None], other=0.0).to(tl.float32)
        s = tl.load(Signs_ptr + rows).to(tl.float32)
        xr = _fwht_exact_fp32(x * s[None, :], BM, BK)
        q = tld.rint(xr * inv_scale[:, None])
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(OutI8_ptr + offs_m[:, None] * stride_om + rows[None, :] * stride_ok,
                 q, mask=mask_m[:, None])


@triton.jit
def hf_int4_g16_moe_pair_kernel(
    X_ptr, Signs_ptr, H_ptr, Ids_ptr, Q_ptr, U_ptr, D_ptr, BiasCorr_ptr, Out_ptr,
    N, K, E, IN_DIV,
    BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS_CORR: tl.constexpr,
):
    """Grouped routed-expert W4A16 GEMM: one output row per (token, slot) pair.

    Pair ``p`` reads input row ``p // IN_DIV`` and multiplies it by expert
    ``Ids[p]`` of the packed 3D weights ``[E, K//8, N]``.  Routing is consumed
    on the device, so the launch shape never depends on which experts were
    picked.  A pair whose expert id is outside ``[0, E)`` writes zeros.
    """
    p = tl.program_id(0)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    eid = tl.load(Ids_ptr + p)
    valid = (eid >= 0) & (eid < E)
    e = tl.where(valid, eid, 0).to(tl.int64)
    row = (p // IN_DIV).to(tl.int64)
    q_base = Q_ptr + e * (K // 8) * N
    u_base = U_ptr + e * (K // 64) * N
    offs_k = tl.arange(0, BK)
    h = tl.load(H_ptr + offs_k[:, None] * BK + offs_k[None, :]).to(tl.float32)
    acc = tl.zeros((BN,), dtype=tl.float32)

    for k0 in range(0, K, BK):
        rows = k0 + offs_k
        x = tl.load(X_ptr + row * K + rows).to(tl.float32)
        s = tl.load(Signs_ptr + rows).to(tl.float32)
        xr = tl.sum((x * s)[:, None] * h, axis=0)
        wi = _decode_w_tile(q_base, u_base, rows, offs_n, N, rows < K, mask_n).to(tl.float32)
        acc += tl.sum(xr[:, None] * wi, axis=0)

    d = tl.load(D_ptr + e * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = acc * d
    if HAS_BIAS_CORR:
        y = y + tl.load(BiasCorr_ptr + e * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = tl.where(valid, y, 0.0)
    tl.store(Out_ptr + p.to(tl.int64) * N + offs_n, y, mask=mask_n)


@triton.jit
def hf_int4_g16_w4a8_kernel(
    XI8_ptr, Q_ptr, U_ptr, SX_ptr, D_ptr, Bias_ptr, BiasCorr_ptr, OutI32_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_BIAS_CORR: tl.constexpr,
    WRITE_I32: tl.constexpr,
):
    """W4A8 GEMM: exact int8 x int8 -> int32 accumulation, fp32 epilogue."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.int32)

    for k0 in range(0, K, BK):
        rows = k0 + tl.arange(0, BK)
        mask_k = rows < K
        x = tl.load(XI8_ptr + offs_m[:, None] * stride_xm + rows[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0)
        w = _decode_w_tile(Q_ptr, U_ptr, rows, offs_n, N, mask_k, mask_n).to(tl.int8)
        acc = tl.dot(x, w, acc)

    if WRITE_I32:
        tl.store(OutI32_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=mask_m[:, None] & mask_n[None, :])

    sx = tl.load(SX_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    d = tl.load(D_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = acc.to(tl.float32) * (sx[:, None] * d[None, :])
    if HAS_BIAS:
        b = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = y + b[None, :]
    if HAS_BIAS_CORR:
        bc = tl.load(BiasCorr_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = y + bc[None, :]
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             y, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def hf_int4_g16_unpack_kernel(Q_ptr, U_ptr, Out_ptr, K, N, BN: tl.constexpr):
    """Exact bit-unpack to int8 [N,K] row-major, primarily for validation."""
    pid_g = tl.program_id(0)  # 16-wide K group
    pid_n = tl.program_id(1)
    cols = pid_n * BN + tl.arange(0, BN)
    mask_n = cols < N
    rows = pid_g * 16 + tl.arange(0, 16)
    mask_k = rows < K
    w = _decode_w_tile(Q_ptr, U_ptr, rows, cols, N, mask_k, mask_n).to(tl.int8)
    # Output logical [N,K]; transposed scatter is intentional for validation only.
    tl.store(Out_ptr + cols[None, :] * K + rows[:, None], w,
             mask=mask_k[:, None] & mask_n[None, :])


def _dummy_f32(device) -> torch.Tensor:
    key = (device.type, device.index)
    if not hasattr(_dummy_f32, "cache"):
        _dummy_f32.cache = {}
    if key not in _dummy_f32.cache:
        _dummy_f32.cache[key] = torch.zeros(1, device=device, dtype=torch.float32)
    return _dummy_f32.cache[key]


def _h32(device, block: int) -> torch.Tensor:
    key = (device.type, device.index, block)
    if not hasattr(_h32, "cache"):
        _h32.cache = {}
    if key not in _h32.cache:
        _h32.cache[key] = make_hadamard(block, device, torch.float32).contiguous()
    return _h32.cache[key]


def _validate_activation_signs(x: torch.Tensor, signs: torch.Tensor) -> tuple[int, int]:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [M,K], got {tuple(x.shape)}")
    m, k = x.shape
    had_block_for_k(k)  # rejects widths that would need out-of-bounds block reads
    if signs.numel() != k:
        raise ValueError(f"signs length {signs.numel()} != K {k}")
    if signs.dtype != torch.int8:
        raise ValueError(f"signs must be int8, got {signs.dtype}")
    return m, k


def fwht_actq(x: torch.Tensor, signs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_activation_signs(x, signs)
    m, k = x.shape
    out = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    fwht_actq_kernel[(triton.cdiv(m, 16),)](
        x, signs, out, scale, m, k,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1), BM=16, BK=had_block_for_k(k),
    )
    return out, scale


def fwht128_actq(x: torch.Tensor, signs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`fwht_actq` restricted to Had128 geometry."""
    if x.shape[-1] % 128:
        raise ValueError(f"fwht128_actq needs K a multiple of 128, got {x.shape[-1]}")
    return fwht_actq(x, signs)


def w4a16_linear(
    x: torch.Tensor, q_pack: torch.Tensor, u_pack: torch.Tensor, d_row: torch.Tensor,
    signs: torch.Tensor, bias: torch.Tensor | None = None, bias_corr: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("w4a16_linear expects a rank-2 activation")
    m, k = _validate_activation_signs(x, signs)
    n = d_row.numel()
    out_dtype = out_dtype or x.dtype
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    db = _dummy_f32(x.device)
    bias_arg = bias if bias is not None else db
    bc_arg = bias_corr if bias_corr is not None else db
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    block = had_block_for_k(k)
    hf_int4_g16_w4a16_kernel[grid](
        x, signs, _h32(x.device, block), q_pack, u_pack, d_row, bias_arg, bc_arg, out,
        m, n, k, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        BM=16, BN=16, BK=block, HAS_BIAS=bias is not None, HAS_BIAS_CORR=bias_corr is not None,
    )
    return out


def w4a8_linear_from_quantized(
    xi8: torch.Tensor, sx: torch.Tensor, q_pack: torch.Tensor, u_pack: torch.Tensor,
    d_row: torch.Tensor, bias: torch.Tensor | None = None, bias_corr: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16, write_i32: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if xi8.ndim != 2 or xi8.dtype != torch.int8:
        raise ValueError("xi8 must be rank-2 int8")
    m, k = xi8.shape
    had_block_for_k(k)
    n = d_row.numel()
    out = torch.empty((m, n), device=xi8.device, dtype=out_dtype)
    out_i32 = torch.empty((m, n), device=xi8.device, dtype=torch.int32) if write_i32 else out.new_empty((1, 1), dtype=torch.int32)
    db = _dummy_f32(xi8.device)
    bias_arg = bias if bias is not None else db
    bc_arg = bias_corr if bias_corr is not None else db
    # 32x32 tiles reduce program count on prefill while preserving arbitrary tails.
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    hf_int4_g16_w4a8_kernel[grid](
        xi8, q_pack, u_pack, sx, d_row, bias_arg, bc_arg, out_i32, out,
        m, n, k, xi8.stride(0), xi8.stride(1), out.stride(0), out.stride(1),
        BM=32, BN=32, BK=64, HAS_BIAS=bias is not None,
        HAS_BIAS_CORR=bias_corr is not None, WRITE_I32=write_i32,
    )
    return (out, out_i32) if write_i32 else out


def w4a8_linear(
    x: torch.Tensor, q_pack: torch.Tensor, u_pack: torch.Tensor, d_row: torch.Tensor,
    signs: torch.Tensor, bias: torch.Tensor | None = None, bias_corr: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    xi8, sx = fwht_actq(x, signs)
    return w4a8_linear_from_quantized(
        xi8, sx, q_pack, u_pack, d_row, bias, bias_corr, out_dtype or x.dtype
    )


def moe_pair_gemm(
    x: torch.Tensor, expert_ids: torch.Tensor, q_pack: torch.Tensor, u_pack: torch.Tensor,
    d_row: torch.Tensor, signs: torch.Tensor, bias_corr: torch.Tensor | None,
    in_div: int, out: torch.Tensor,
) -> torch.Tensor:
    """Run :func:`hf_int4_g16_moe_pair_kernel` into ``out`` ``[P, N]``; operands are pre-validated."""
    num_experts, n = d_row.shape
    if num_experts < 1:
        # The kernel substitutes expert 0 for invalid ids and loads it before masking.
        raise ValueError("empty expert table: there is no expert 0 to read")
    k = x.shape[1]
    pairs = expert_ids.shape[0]
    block = had_block_for_k(k)
    bc_arg = bias_corr if bias_corr is not None else _dummy_f32(x.device)
    hf_int4_g16_moe_pair_kernel[(pairs, triton.cdiv(n, 32))](
        x, signs, _h32(x.device, block), expert_ids, q_pack, u_pack, d_row, bc_arg, out,
        n, k, num_experts, in_div,
        BN=32, BK=block, HAS_BIAS_CORR=bias_corr is not None,
    )
    return out


def unpack_to_int8(q_pack: torch.Tensor, u_pack: torch.Tensor, n: int, k: int) -> torch.Tensor:
    out = torch.empty((n, k), device=q_pack.device, dtype=torch.int8)
    hf_int4_g16_unpack_kernel[(k // 16, triton.cdiv(n, 64))](q_pack, u_pack, out, k, n, BN=64)
    return out
