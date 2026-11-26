"""Exact HF-INT4-G16 bit layout and quantization math.

Canonical production layout (Xe2 DPAS native, little-endian):

* ``q_pack``: int32 bit container, shape ``[K // 8, N]``.  The uint32 word at
  ``[r, n]`` describes eight consecutive input positions ``base = 8*r`` for
  output row ``n``.  Byte ``b`` (bits ``8*b+7:8*b``) has low nibble
  ``q(base+b)+8`` and high nibble ``q(base+4+b)+8``.  This nibble interleaving
  is exactly what the Xe2 XMX DPAS kernels consume; it is not the naive
  adjacent-pair layout.
* ``u_pack``: int16 bit container, shape ``[K // 64, N]``.  The uint16 word at
  ``[r, n]`` contains ``u-1`` nibbles for groups ``4*r+j`` at bits ``4*j``.
* ``d_row``: fp32 ``[N]`` row scale.  Reconstruction before bias correction is
  ``D[n] * (q[n,k] * u[n,k//16])``.
* ``bias_corr``: optional fp32 ``[N]``, added once in the fp32 epilogue.

Rotation: contiguous non-permuted blockwise Sylvester Hadamard.  Both weights
and activations are transformed as ``z = (x * signs) @ H_B`` per block, where
signs are generated on CPU with torch generator seed ``had_seed * 100003 + K``
and values in {-1,+1}.  The converter reference and this loader use seed 101.

The block size ``B`` is a pure function of the projection's global ``K``
(:func:`had_block_for_k`): 128 when ``K % 128 == 0``, else 64 when
``K % 64 == 0`` (every block of that projection is 64 wide), else the
projection is not representable.  The packed containers are the same for both,
because ``u_pack`` already groups 64 positions per word.

Format versions: ``hf_int4_g16/1`` holds dense Had128 modules only.
``hf_int4_g16/2`` is required as soon as a checkpoint holds a Had64 projection
or packed MoE experts.  Expert tensors keep this bit layout but use the names
in :data:`EXPERT_TENSOR_SUFFIXES`.

Disk-contract note: ``xe2_dpas_native_v1`` is a distinct Xe2-native contract,
not a portable GPTQ/AWQ-style adjacent-nibble interchange format.  Loaders
must reject unknown ``layout`` or ``format_version`` values rather than trying
to reinterpret the bit containers.
"""
from __future__ import annotations

import hashlib
import math

import torch

FORMAT_VERSION = "hf_int4_g16/1"
FORMAT_VERSION_V2 = "hf_int4_g16/2"
ROTATION_V1 = "Had128NoPerm"
ROTATION_V2 = "HadAutoNoPerm"
# The only block schedule hf_int4_g16/2 defines.  A different schedule (for
# example Had128 blocks plus a Had64 tail) needs a new format version.
HAD_BLOCK_RULE_V2 = "per projection: 128 if K % 128 == 0 else 64"
GROUP_SIZE = 16
HADAMARD_BLOCK = 128
HADAMARD_BLOCK_SMALL = 64
HAD_SEED = 101

# MoE expert tensors: logical packed role -> on-disk suffix.  vLLM derives the
# RoutedExperts parameter name from the suffix ("w13_" / "w2_" + suffix), and its
# GraniteMoeHybrid loader only finds names that match its fixed expert suffix
# list (".w13_qweight", ".w13_scale", ".w13_bias", ...).  The row scale is
# therefore "scale_super": "super_scales" would not be remapped.
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERT_TENSOR_SUFFIXES = {
    "q_pack": "qweight",
    "u_pack": "scales",
    "d_row": "scale_super",
    "bias_corr": "bias_corr",
}


def is_json_int(value) -> bool:
    """A JSON integer.  ``16.0`` and ``true`` compare equal to ints but are not."""
    return isinstance(value, int) and not isinstance(value, bool)


def had_block_for_k(k: int) -> int:
    """Hadamard block size of a projection with global input width ``k``."""
    k = int(k)
    if k > 0 and k % HADAMARD_BLOCK == 0:
        return HADAMARD_BLOCK
    if k > 0 and k % HADAMARD_BLOCK_SMALL == 0:
        return HADAMARD_BLOCK_SMALL
    raise ValueError(f"K must be a multiple of 128 (Had128) or of 64 (Had64); got K={k}")


def check_payload(role: str, tensor: torch.Tensor, what: str) -> None:
    """Value checks that header-only admission cannot make."""
    if role not in ("d_row", "bias_corr"):
        return
    if not torch.isfinite(tensor).all():
        raise ValueError(f"hf_int4_g16: {what} contains NaN or Inf")
    if role == "d_row" and not (tensor > 0).all():
        raise ValueError(f"hf_int4_g16: {what} row scales must be positive")


def make_hadamard(block: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix of size ``block`` (64 or 128)."""
    if block not in (HADAMARD_BLOCK_SMALL, HADAMARD_BLOCK):
        raise ValueError(f"unsupported Hadamard block {block}")
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(float(block))


def make_had128(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return make_hadamard(HADAMARD_BLOCK, device, dtype)


def signs_for_k(k: int, seed: int = HAD_SEED, device: torch.device | str = "cpu") -> torch.Tensor:
    """Deterministic sign vector.  The exact seed arithmetic is part of the format."""
    had_block_for_k(k)
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 100003 + int(k))
    return (torch.randint(0, 2, (int(k),), generator=g) * 2 - 1).to(device=device, dtype=torch.int8)




def sign_sha256_for_k(k: int, seed: int = HAD_SEED) -> str:
    """SHA-256 of the raw little-endian int8 sign bytes for one global K."""
    return hashlib.sha256(signs_for_k(k, seed, "cpu").numpy().tobytes()).hexdigest()


def rotate_had_noperm(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply the canonical rotation to a tensor whose last dimension is the global K."""
    if x.shape[-1] != signs.numel():
        raise ValueError(f"K mismatch: x {tuple(x.shape)} signs {tuple(signs.shape)}")
    block = had_block_for_k(x.shape[-1])
    h = make_hadamard(block, x.device, torch.float32)
    xs = x.float() * signs.to(device=x.device, dtype=torch.float32)
    return (xs.reshape(-1, block) @ h).reshape(xs.shape)


def rotate_had128_noperm(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """:func:`rotate_had_noperm` restricted to Had128 geometry."""
    if x.shape[-1] % HADAMARD_BLOCK:
        raise ValueError(f"Had128 rotation needs K a multiple of 128, got {x.shape[-1]}")
    return rotate_had_noperm(x, signs)


def _q_u_from_logical(w_rot: torch.Tensor, ratios=(0.76, 0.82, 0.88, 0.94, 1.00, 1.06), chunk_n: int | None = None):
    """Progressive G16 quantizer exactly matching the exp8b reference."""
    n, k = w_rot.shape
    had_block_for_k(k)
    ng = k // GROUP_SIZE
    wg = w_rot.reshape(n, ng, GROUP_SIZE)
    row_amax = wg.abs().amax(dim=(1, 2)).clamp(min=1e-12)
    u_grid = torch.arange(1, 17, device=w_rot.device, dtype=torch.float32)
    best_row_err = torch.full((n,), 1e30, device=w_rot.device)
    best_d = torch.zeros(n, device=w_rot.device, dtype=torch.float32)
    best_q = torch.zeros((n, k), device=w_rot.device, dtype=torch.uint8)
    best_u = torch.zeros((n, ng), device=w_rot.device, dtype=torch.uint8)
    if chunk_n is None:
        chunk_n = 512 if k > 8192 else 1024
    for r in ratios:
        dc = (row_amax * float(r)) / 112.0
        re = torch.zeros(n, device=w_rot.device, dtype=torch.float32)
        qc = torch.empty((n, k), device=w_rot.device, dtype=torch.uint8)
        uc = torch.empty((n, ng), device=w_rot.device, dtype=torch.uint8)
        for s0 in range(0, n, chunk_n):
            s1 = min(s0 + chunk_n, n)
            wc = wg[s0:s1]
            sc = dc[s0:s1, None, None, None] * u_grid[None, None, None, :]
            qa = torch.round(wc.unsqueeze(-1) / sc).clamp(-8.0, 7.0)
            err = (wc.unsqueeze(-1) - qa * sc).pow(2).sum(dim=2)
            bi = err.argmin(dim=-1)
            re[s0:s1] = err.gather(-1, bi.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            uc[s0:s1] = bi.to(torch.uint8)
            qw = qa.gather(-1, bi.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, GROUP_SIZE, 1)).squeeze(-1)
            qc[s0:s1] = (qw + 8.0).to(torch.uint8).reshape(s1 - s0, k)
        m = re < best_row_err
        best_row_err = torch.where(m, re, best_row_err)
        best_d = torch.where(m, dc, best_d)
        best_q[m] = qc[m]
        best_u[m] = uc[m]
    return best_q, best_u, best_d


def pack_native_words(q_plus8: torch.Tensor, u_minus1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack logical arrays into the canonical Xe2-native int32/int16 bit containers.

    ``q_plus8`` is uint8 ``[N,K]`` with values q+8. ``u_minus1`` is uint8
    ``[N,K//16]`` with values u-1.  This implementation uses NumPy only for the
    final bit-container cast, making the little-endian bit layout explicit.
    """
    n, k = q_plus8.shape
    had_block_for_k(k)
    qcpu = q_plus8.detach().cpu().numpy().astype("uint32", copy=False)
    ucpu = u_minus1.detach().cpu().numpy().astype("uint32", copy=False)
    import numpy as np

    q64 = qcpu.reshape(n, k // 8, 8)
    words = np.zeros((n, k // 8), dtype=np.uint32)
    for b in range(4):
        words |= q64[:, :, b] << (8 * b)
        words |= q64[:, :, 4 + b] << (8 * b + 4)
    # Native layout is [K//8,N]; bytes inside each uint32 are little-endian.
    q_out = words.T.copy().view(np.int32)

    ng = k // GROUP_SIZE
    u64 = ucpu.reshape(n, k // 64, 4)
    uwords = np.zeros((n, k // 64), dtype=np.uint32)
    for j in range(4):
        uwords |= u64[:, :, j] << (4 * j)
    u_out = uwords.astype(np.uint16).T.copy().view(np.int16)
    return torch.from_numpy(q_out), torch.from_numpy(u_out)


@torch.no_grad()
def quantize_pack_hf_int4(
    weight: torch.Tensor,
    seed: int = HAD_SEED,
    device: torch.device | str | None = None,
    ratios=(0.76, 0.82, 0.88, 0.94, 1.00, 1.06),
) -> dict[str, torch.Tensor]:
    """Quantize a BF16/FP16/FP32 ``[N,K]`` weight and return canonical tensors.

    Returned tensors are CPU tensors suitable for safetensors serialization:
    ``q_pack`` int32 ``[K//8,N]``, ``u_pack`` int16 ``[K//64,N]`` and
    ``d_row`` fp32 ``[N]``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2 [N,K], got {tuple(weight.shape)}")
    n, k = weight.shape
    had_block_for_k(k)
    dev = torch.device(device) if device is not None else weight.device
    w = weight.to(device=dev, dtype=torch.float32)
    signs = signs_for_k(k, seed, dev).float()
    wr = rotate_had_noperm(w, signs)
    q, u, d = _q_u_from_logical(wr, ratios=ratios)
    qp, up = pack_native_words(q, u)
    return {"q_pack": qp, "u_pack": up, "d_row": d.detach().cpu().float().contiguous()}


def unpack_hf_int4_reference(q_pack: torch.Tensor, u_pack: torch.Tensor) -> torch.Tensor:
    """Decode canonical packed tensors to exact int8 ``[N,K]`` (q*u).

    This is deliberately straightforward PyTorch indexing and is the bit-exact
    oracle used by unit tests; it is not a performance path.
    """
    if q_pack.ndim != 2 or u_pack.ndim != 2:
        raise ValueError("q_pack and u_pack must be rank-2")
    k8, n = q_pack.shape
    k64, nu = u_pack.shape
    k = k8 * 8
    if k64 != k // 64 or nu != n:
        raise ValueError(f"packed shape mismatch q{tuple(q_pack.shape)} u{tuple(u_pack.shape)}")
    rows = torch.arange(k, device=q_pack.device)
    q32 = q_pack.to(torch.int64)[rows // 8, :].transpose(0, 1).contiguous()  # [N,K]
    q32 = q32 & 0xFFFFFFFF
    shifts = ((rows % 4) * 8 + ((rows % 8) // 4) * 4).to(q_pack.device)
    q = ((q32 >> shifts) & 0xF).to(torch.int16) - 8
    u16 = u_pack.to(torch.int64)[rows // 64, :].transpose(0, 1).contiguous() & 0xFFFF
    ushifts = (((rows // 16) % 4) * 4).to(q_pack.device)
    u = ((u16 >> ushifts) & 0xF).to(torch.int16) + 1
    return (q * u).to(torch.int8)
