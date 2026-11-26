"""Routed-expert (MoE) HF-INT4-G16 forward.

Semantics, per token ``t`` with router outputs ``(w_ts, e_ts)``::

    h_ts = act(gate) * up            gate, up = split(bf16(W13[e_ts] x_t + bc13[e_ts]))
    y_t  = bf16( sum_s fp32(w_ts) * (W2[e_ts] h_ts + bc2[e_ts]) )

The gate/up GEMM output is rounded to the activation dtype exactly like a dense
``gate_up_proj``.  The activation is evaluated in FP32 and rounded once.  Every
expert output stays FP32 until the weighted sum, which is cast once.

Routing is consumed on the device in two interchangeable ways.  Small batches
run each GEMM as one (token, slot) pair launch.  From ``GROUPED_MIN_TOKENS``
tokens on, native bf16 batches bucket the pairs by expert on the device, rotate
each activation row once and share an expert's unpacked weight panel across up
to 16 rows.  Both give bit-identical outputs.  Neither syncs with the host or
dispatches per expert, and launch shapes depend only on the token count and the
layer geometry, so a captured XPU graph follows new routing on replay.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import kernels_triton as tk
from . import native_l0
from .format import had_block_for_k

MOE_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda gate: F.gelu(gate, approximate="tanh"),
}

# Bounds the FP32 per-pair temporaries: 1024 tokens x 8 slots x 2816 x 4 B = 92 MB.
DEFAULT_CHUNK_TOKENS = 1024
# Decode and small batches (M <= 8, and 9..15) stay on the pair kernel.  The
# choice depends on the token count only, never on routing read by the host.
GROUPED_MIN_TOKENS = 16
# Activations, routing weights and GEMM outputs.  Integer or FP64 tensors are
# rejected rather than silently converted.
FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@dataclass(frozen=True)
class PackedExperts:
    """One routed projection stack in the packed 3D layout.

    ``q_pack`` int32 ``[E, K//8, N]``, ``u_pack`` int16 ``[E, K//64, N]``,
    ``d_row`` fp32 ``[E, N]``, ``bias_corr`` optional fp32 ``[E, N]`` and
    ``signs`` int8 ``[K]`` (shared by all experts because it depends only on K).
    """

    q_pack: torch.Tensor
    u_pack: torch.Tensor
    d_row: torch.Tensor
    signs: torch.Tensor
    bias_corr: torch.Tensor | None = None

    @property
    def num_experts(self) -> int:
        return int(self.d_row.shape[0])

    @property
    def n(self) -> int:
        return int(self.d_row.shape[1])

    @property
    def k(self) -> int:
        return int(self.signs.shape[0])

    def validate(self, device: torch.device) -> None:
        if self.d_row.ndim != 2 or self.signs.ndim != 1:
            raise ValueError(f"d_row must be [E,N] and signs [K], got {tuple(self.d_row.shape)}, {tuple(self.signs.shape)}")
        e, n, k = self.num_experts, self.n, self.k
        if e < 1 or n < 1:
            # An empty bank has no expert 0 for the out-of-range path to avoid.
            raise ValueError(f"empty expert table: d_row is [E={e}, N={n}]")
        had_block_for_k(k)
        expected = [
            ("q_pack", self.q_pack, (e, k // 8, n), torch.int32),
            ("u_pack", self.u_pack, (e, k // 64, n), torch.int16),
            ("d_row", self.d_row, (e, n), torch.float32),
            ("signs", self.signs, (k,), torch.int8),
        ]
        if self.bias_corr is not None:
            expected.append(("bias_corr", self.bias_corr, (e, n), torch.float32))
        for name, tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
            if tensor.dtype != dtype:
                raise ValueError(f"{name} dtype {tensor.dtype} != {dtype}")
            if tensor.device != device:
                raise ValueError(f"{name} device {tensor.device} != activation device {device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous canonical packed storage")

    def _native_aligned(self) -> bool:
        """USM alignment of everything but the activation; ``q_pack`` is also a
        2D block-io surface, which :func:`native_l0.moe_supported` checks."""
        operands = (self.q_pack, self.u_pack, self.d_row, self.signs, self.bias_corr)
        return all(t is None or t.data_ptr() % 16 == 0 for t in operands)


def moe_pair_gemm(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    experts: PackedExperts,
    in_div: int = 1,
    out_dtype: torch.dtype | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Grouped routed GEMM.  Returns ``[P, N]``; row ``p`` is
    ``W[expert_ids[p]] x[p // in_div] (+ bias_corr)``, zeros for ids outside ``[0, E)``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [rows,K], got {tuple(x.shape)}")
    experts.validate(x.device)
    if x.dtype not in FLOAT_DTYPES or (out_dtype is not None and out_dtype not in FLOAT_DTYPES):
        raise ValueError(f"activation/output dtypes must be bf16, fp16 or fp32, got {x.dtype} -> {out_dtype}")
    if x.shape[1] != experts.k:
        raise ValueError(f"activation K={x.shape[1]} != expert K={experts.k}")
    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != x.device:
        raise ValueError("expert_ids must be a rank-1 int32 tensor on the activation device")
    pairs = expert_ids.shape[0]
    if in_div < 1 or pairs != x.shape[0] * in_div:
        raise ValueError(f"pairs {pairs} != rows {x.shape[0]} * in_div {in_div}")
    if backend not in ("auto", "native", "triton"):
        raise ValueError(f"unknown backend {backend}")
    out_dtype = out_dtype or x.dtype
    if pairs == 0:
        return x.new_empty((0, experts.n), dtype=out_dtype)

    native_ok = _native_ok(x, expert_ids, experts, out_dtype)
    if backend == "native" and not native_ok:
        raise RuntimeError(
            "native HF-INT4-G16 MoE preconditions not met "
            f"(dtype={x.dtype}, out={out_dtype}, contiguous={x.is_contiguous()}, N={experts.n}, K={experts.k})"
        )
    if backend == "native" or (backend == "auto" and native_ok):
        return native_l0.moe_pair_native(
            x, expert_ids, experts.signs, experts.q_pack, experts.u_pack, experts.d_row,
            experts.bias_corr, in_div, out_dtype == torch.float32,
        )
    if not x.is_contiguous() or not expert_ids.is_contiguous():
        raise ValueError("MoE activations and expert ids must be contiguous")
    out = torch.empty((pairs, experts.n), device=x.device, dtype=out_dtype)
    return tk.moe_pair_gemm(
        x, expert_ids, experts.q_pack, experts.u_pack, experts.d_row, experts.signs,
        experts.bias_corr, in_div, out,
    )


def _native_ok(x: torch.Tensor, expert_ids: torch.Tensor, experts: PackedExperts, out_dtype: torch.dtype) -> bool:
    return bool(
        native_l0.moe_supported(x, experts.q_pack)
        and out_dtype in (torch.bfloat16, torch.float32)
        and experts._native_aligned()
        and expert_ids.is_contiguous()
        and expert_ids.data_ptr() % 16 == 0
    )


def moe_grouped_gemm(
    x: torch.Tensor,
    buckets: object,
    experts: PackedExperts,
    in_div: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """:func:`moe_pair_gemm` on pairs bucketed by expert.  Same ``[P, N]`` result.

    ``buckets`` is the opaque descriptor :func:`native_l0.moe_bucket_native`
    built from the pair ids; the two GEMMs of one routed forward share it.
    Native bf16 input only; the output is bf16 or fp32.
    """
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"grouped MoE out_dtype must be torch.bfloat16 or torch.float32, got {out_dtype}")
    return native_l0.moe_tile_native(
        native_l0.moe_rotate_native(x, experts.signs), experts.q_pack, experts.u_pack, experts.d_row,
        experts.bias_corr, buckets, in_div, out_dtype == torch.float32,
    )


def hf_int4_g16_moe(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: PackedExperts,
    w2: PackedExperts,
    activation: str,
    backend: str = "auto",
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    grouped_min_tokens: int = GROUPED_MIN_TOKENS,
) -> torch.Tensor:
    """Routed experts for ``x`` ``[M,H]`` with ``topk_weights``/``topk_ids`` ``[M,T]``.

    Ids outside ``[0, E)`` contribute zero.  Chunks of at least
    ``grouped_min_tokens`` tokens take the grouped path when it is native.
    """
    act = MOE_ACTIVATIONS.get(activation)
    if act is None:
        raise ValueError(f"unsupported MoE activation {activation!r}; supported: {sorted(MOE_ACTIVATIONS)}")
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [M,H], got {tuple(x.shape)}")
    m, hidden = x.shape
    if topk_ids.ndim != 2 or topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != m:
        raise ValueError(
            f"topk_ids/topk_weights must both be [M={m},T], got {tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError(f"topk_ids/topk_weights must be on {x.device}")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")
    if x.dtype not in FLOAT_DTYPES:
        raise ValueError(f"MoE activations must be bf16, fp16 or fp32, got {x.dtype}")
    if topk_weights.dtype not in FLOAT_DTYPES:
        raise ValueError(f"topk_weights must be bf16, fp16 or fp32, got {topk_weights.dtype}")
    if backend not in ("auto", "native", "triton"):
        raise ValueError(f"unknown backend {backend}")
    # Both banks are checked before anything is launched, even for an empty batch.
    w13.validate(x.device)
    w2.validate(x.device)
    if w13.k != hidden or w2.n != hidden:
        raise ValueError(f"hidden size {hidden} != gate_up K {w13.k} / down N {w2.n}")
    if w13.n != 2 * w2.k:
        raise ValueError(f"gate_up N {w13.n} != 2 * down K {w2.k}")
    if w13.num_experts != w2.num_experts:
        raise ValueError(f"expert count mismatch: {w13.num_experts} != {w2.num_experts}")
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    slots = topk_ids.shape[1]
    num_experts = w13.num_experts
    out = torch.empty_like(x)
    for start in range(0, m, chunk_tokens):
        xc = x[start:start + chunk_tokens]
        tokens = xc.shape[0]
        raw = topk_ids[start:start + tokens].reshape(-1)
        if raw.dtype == torch.int64:
            # Mask in int64: narrowing first would alias 2**32 + 1 onto expert 1.
            raw = torch.where((raw >= 0) & (raw < num_experts), raw, -1)
        # An int32 id cannot alias and the kernels already zero out-of-range
        # ids, so vLLM's int32 routes pay for this one copy only (decode is
        # launch-bound).  The copy is contiguous and aligned for the device.
        ids = raw.to(torch.int32, copy=True)
        grouped = (
            backend != "triton" and slots > 0 and tokens >= grouped_min_tokens
            and _native_ok(xc, ids, w13, x.dtype) and x.dtype == torch.bfloat16
            and w2._native_aligned() and native_l0.packed_weight_surface_error(w2.q_pack) is None
        )
        if grouped:
            buckets = native_l0.moe_bucket_native(ids, num_experts)
            gate_up = moe_grouped_gemm(xc, buckets, w13, slots, x.dtype)
        else:
            gate_up = moe_pair_gemm(xc, ids, w13, in_div=slots, out_dtype=x.dtype, backend=backend)
        gate, up = gate_up.chunk(2, dim=-1)
        inner = (act(gate.float()) * up.float()).to(x.dtype)
        if grouped:
            y = moe_grouped_gemm(inner, buckets, w2, 1, torch.float32)
        else:
            y = moe_pair_gemm(inner, ids, w2, in_div=1, out_dtype=torch.float32, backend=backend)
        # y is indexed by the original pair id token * slots + slot, so the sum
        # runs in slot order on both paths.
        weights = topk_weights[start:start + tokens].to(torch.float32).reshape(tokens, slots, 1)
        out[start:start + tokens] = (y.reshape(tokens, slots, hidden) * weights).sum(dim=1).to(x.dtype)
    return out
