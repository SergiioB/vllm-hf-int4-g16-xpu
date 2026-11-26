"""Unified HF-INT4-G16 linear operator dispatch.

Backends:
* ``native``: zero-copy hardened Xe2 XMX DPAS / fused decode kernels through
  Level-Zero.  Currently bf16-only and conservative about 2D-block-io alignment.
* ``triton``: universal fp16/bf16 baseline for every legal shape.
* ``auto``: native when its exact preconditions hold, otherwise Triton.

A caller may force ``backend='native'``; unsupported shapes then raise instead
of silently falling back.

The rank-2 entry point is also registered as a PyTorch custom operator.  This
is essential in vLLM's default fullgraph compilation mode: Dynamo treats the
complete dispatch/kernels as an opaque accelerator operation instead of trying
to trace Python runtime decisions, Triton launch machinery, or the Level-Zero
extension loader.
"""
from __future__ import annotations

import torch

from . import kernels_triton as tk
from . import native_l0
from .format import had_block_for_k


def _validate_linear_inputs(
    x: torch.Tensor,
    q_pack: torch.Tensor,
    u_pack: torch.Tensor,
    d_row: torch.Tensor,
    signs: torch.Tensor,
    bias: torch.Tensor | None = None,
    bias_corr: torch.Tensor | None = None,
) -> tuple[int, int, int]:
    """Validate the packed operator contract before any backend is selected.

    Geometry, dtype, device and packed-layout violations are invalid input in
    every backend.  In contrast, an aligned/native-only fast-path property such
    as bf16 storage, a 16-byte base address or absence of a model bias is not
    judged here; those cases may legitimately continue to Triton under ``auto``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [M,K], got shape {tuple(x.shape)}")
    if q_pack.ndim != 2:
        raise ValueError(f"q_pack must be rank-2 [K//8,N], got shape {tuple(q_pack.shape)}")
    if u_pack.ndim != 2:
        raise ValueError(f"u_pack must be rank-2 [K//64,N], got shape {tuple(u_pack.shape)}")
    if d_row.ndim != 1:
        raise ValueError(f"d_row must be rank-1 [N], got shape {tuple(d_row.shape)}")
    if signs.ndim != 1:
        raise ValueError(f"signs must be rank-1 [K], got shape {tuple(signs.shape)}")

    m, k = x.shape
    had_block_for_k(k)
    if q_pack.shape[0] * 8 != k:
        raise ValueError(f"q_pack K mismatch: shape {tuple(q_pack.shape)} for activation K={k}")
    if u_pack.shape[0] * 64 != k:
        raise ValueError(f"u_pack K mismatch: shape {tuple(u_pack.shape)} for activation K={k}")
    if q_pack.shape[1] != u_pack.shape[1]:
        raise ValueError(f"q_pack/u_pack N mismatch: {q_pack.shape[1]} != {u_pack.shape[1]}")
    n = int(q_pack.shape[1])
    if d_row.shape != (n,):
        raise ValueError(f"d_row shape {tuple(d_row.shape)} != ({n},)")
    if signs.shape != (k,):
        raise ValueError(f"signs shape {tuple(signs.shape)} != ({k},)")

    tensors = [("x", x), ("q_pack", q_pack), ("u_pack", u_pack), ("d_row", d_row), ("signs", signs)]
    if bias is not None:
        tensors.append(("bias", bias))
    if bias_corr is not None:
        tensors.append(("bias_corr", bias_corr))
    for name, t in tensors[1:]:
        if t.device != x.device:
            raise ValueError(f"{name} device {t.device} != x device {x.device}")

    if q_pack.dtype != torch.int32:
        raise ValueError(f"q_pack must be int32, got {q_pack.dtype}")
    if u_pack.dtype != torch.int16:
        raise ValueError(f"u_pack must be int16, got {u_pack.dtype}")
    if d_row.dtype != torch.float32:
        raise ValueError(f"d_row must be float32, got {d_row.dtype}")
    if signs.dtype != torch.int8:
        raise ValueError(f"signs must be int8, got {signs.dtype}")
    if bias is not None and bias.shape != (n,):
        raise ValueError(f"bias shape {tuple(bias.shape)} != ({n},)")
    if bias_corr is not None:
        if bias_corr.shape != (n,):
            raise ValueError(f"bias_corr shape {tuple(bias_corr.shape)} != ({n},)")
        if bias_corr.dtype != torch.float32:
            raise ValueError(f"bias_corr must be float32, got {bias_corr.dtype}")

    # Packed bit containers are defined as dense canonical storage.  Triton
    # indexes that layout directly and must not silently interpret a strided view.
    for name, t in (("q_pack", q_pack), ("u_pack", u_pack), ("d_row", d_row), ("signs", signs)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous canonical packed storage")
    if bias is not None and not bias.is_contiguous():
        raise ValueError("bias must be contiguous")
    if bias_corr is not None and not bias_corr.is_contiguous():
        raise ValueError("bias_corr must be contiguous")
    return int(m), int(k), n


def _native_ok(x2: torch.Tensor, q_pack: torch.Tensor, u_pack: torch.Tensor,
               d_row: torch.Tensor, signs: torch.Tensor, n: int,
               model_bias: torch.Tensor | None, bias_corr: torch.Tensor | None, mode: str) -> bool:
    return native_l0.native_supported(
        x2, n, bias_present=model_bias is not None,
        signs=signs, q_pack=q_pack, u_pack=u_pack, d_row=d_row, bias_corr=bias_corr, dpas=mode == "w4a8",
    )


def _linear2d_impl(
    x: torch.Tensor,
    q_pack: torch.Tensor,
    u_pack: torch.Tensor,
    d_row: torch.Tensor,
    signs: torch.Tensor,
    bias: torch.Tensor | None = None,
    bias_corr: torch.Tensor | None = None,
    backend: str = "auto",
    mode: str = "auto",
) -> torch.Tensor:
    """Compute one quantized linear for rank-2 ``x``.

    The Hadamard block follows from the activation width (``had_block_for_k``).
    A caller that passes a row shard must keep the shard width a multiple of the
    block chosen for the global K; the vLLM plugin enforces that.
    """
    m, k, n = _validate_linear_inputs(x, q_pack, u_pack, d_row, signs, bias, bias_corr)
    if m == 0:
        return torch.empty((0, n), device=x.device, dtype=x.dtype)
    if mode == "auto":
        mode = "w4a16" if m in (1, 2, 3, 4) else "w4a8"
    if mode not in ("w4a16", "w4a8"):
        raise ValueError(f"unknown mode {mode}")
    if mode == "w4a8" and k > 17408:
        raise ValueError(f"K={k} exceeds the safe W4A8 INT32 accumulation bound 17408")
    if backend not in ("auto", "native", "triton"):
        raise ValueError(f"unknown backend {backend}")

    native_ok = _native_ok(x, q_pack, u_pack, d_row, signs, n, bias, bias_corr, mode)
    use_native = backend == "native" or (backend == "auto" and native_ok)
    if backend == "native" and not native_ok:
        raise RuntimeError(
            "native HF-INT4-G16 preconditions not met "
            f"(dtype={x.dtype}, contiguous={x.is_contiguous()}, M={m}, N={n}, K={k}, "
            f"model_bias={bias is not None}); choose backend='triton' explicitly"
        )

    if use_native:
        if mode == "w4a16" and m in (1, 2, 3, 4):
            return native_l0.decode_native(x, signs, q_pack, u_pack, d_row, bias_corr)
        # The native prefill chain is W4A8.  W4A16 forced mode for other M is
        # deliberately not approximated as W4A8.
        if mode == "w4a8":
            return native_l0.w4a8_native(x, signs, q_pack, u_pack, d_row, bias_corr)
        if backend == "native":
            raise RuntimeError(f"native backend has no W4A16 kernel for M={m}")

    if mode == "w4a16":
        return tk.w4a16_linear(x, q_pack, u_pack, d_row, signs, bias, bias_corr, x.dtype)
    return tk.w4a8_linear(x, q_pack, u_pack, d_row, signs, bias, bias_corr, x.dtype)


@torch.library.custom_op("hf_int4_g16_xpu::linear2d", mutates_args=())
def _linear2d_custom(
    x: torch.Tensor,
    q_pack: torch.Tensor,
    u_pack: torch.Tensor,
    d_row: torch.Tensor,
    signs: torch.Tensor,
    bias: torch.Tensor | None,
    bias_corr: torch.Tensor | None,
    backend: str,
    mode: str,
) -> torch.Tensor:
    return _linear2d_impl(x, q_pack, u_pack, d_row, signs, bias, bias_corr, backend, mode)


@_linear2d_custom.register_fake
def _linear2d_fake(x, q_pack, u_pack, d_row, signs, bias, bias_corr, backend, mode):
    if x.ndim != 2:
        raise ValueError(f"expected rank-2 input, got {tuple(x.shape)}")
    return torch.empty((x.shape[0], d_row.shape[0]), device=x.device, dtype=x.dtype)


def hf_int4_g16_linear2d(
    x: torch.Tensor,
    q_pack: torch.Tensor,
    u_pack: torch.Tensor,
    d_row: torch.Tensor,
    signs: torch.Tensor,
    bias: torch.Tensor | None = None,
    bias_corr: torch.Tensor | None = None,
    backend: str = "auto",
    mode: str = "auto",
) -> torch.Tensor:
    """Compute one quantized linear for rank-2 ``x``.

    ``mode='auto'`` uses fused W4A16 decode for M in {1,2,3,4} and W4A8 for
    larger batches.  ``mode='w4a16'`` or ``'w4a8'`` forces the math path.
    """
    if torch.compiler.is_compiling():
        return _linear2d_custom(x, q_pack, u_pack, d_row, signs, bias, bias_corr, backend, mode)
    # Eager decode is latency-critical.  Python custom-op dispatch costs roughly
    # 90 us/call in this container; use the same implementation directly in
    # eager mode and reserve the opaque op for Dynamo/fullgraph tracing.
    return _linear2d_impl(x, q_pack, u_pack, d_row, signs, bias, bias_corr, backend, mode)


def hf_int4_g16_linear(
    x: torch.Tensor,
    q_pack: torch.Tensor,
    u_pack: torch.Tensor,
    d_row: torch.Tensor,
    signs: torch.Tensor,
    bias: torch.Tensor | None = None,
    bias_corr: torch.Tensor | None = None,
    backend: str = "auto",
    mode: str = "auto",
) -> torch.Tensor:
    """N-dimensional input wrapper matching ``torch.nn.functional.linear`` semantics."""
    sh = x.shape
    x2 = x.reshape(-1, sh[-1])
    y = hf_int4_g16_linear2d(
        x2, q_pack, u_pack, d_row, signs, bias, bias_corr, backend=backend, mode=mode
    )
    return y.reshape(*sh[:-1], y.shape[-1])
