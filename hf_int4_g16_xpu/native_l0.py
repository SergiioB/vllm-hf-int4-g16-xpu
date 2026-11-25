"""Runtime builder and Python interface for the zero-copy Level-Zero backend."""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None
_BUILD_ERROR: Exception | None = None
_KERNEL_BIN = str(Path(__file__).resolve().parent / "native" / "hf_int4_kernels_bmg.bin")
_ADMITTED_DIGESTS: dict[tuple[int, str], str] = {}


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
                             expected_x_dtype: torch.dtype = torch.bfloat16) -> None:
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
    if k % 128 or k > 17408:
        raise ValueError(f"native requires K%128==0 and K<=17408, got K={k}")
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
        if q.shape[1] % 8:
            raise ValueError(f"native requires N%8==0 for aligned 2D block-io rows, got N={q.shape[1]}")
    if sx is not None and (sx.dtype != torch.float32 or sx.numel() != m):
        raise ValueError(f"sx must be contiguous fp32 with M={m} elements")
    for name, t in [("x", x), ("signs", signs), ("q_pack", q), ("u_pack", u), ("d_row", d), ("sx", sx), ("bias_corr", bias_corr)]:
        if t is not None and not _aligned16(t):
            raise ValueError(f"{name} base pointer is not 16-byte aligned")


def native_supported(x: torch.Tensor, n: int, bias_present: bool = False,
                     signs: torch.Tensor | None = None, q_pack: torch.Tensor | None = None,
                     u_pack: torch.Tensor | None = None, d_row: torch.Tensor | None = None,
                     bias_corr: torch.Tensor | None = None) -> bool:
    """Conservative fast-path gate.  All other cases use the explicit Triton baseline."""
    if not (x.is_xpu and x.is_contiguous()):
        return False
    if x.dtype != torch.bfloat16:
        return False
    if x.ndim != 2 or x.shape[1] % 128 or x.shape[1] > 17408:
        return False
    # The Xe2 2D block-io pitch contract used by the hardened DPAS kernels
    # requires a conservative aligned output pitch.  N=300 intentionally falls
    # to the universal Triton kernel rather than risking a misaligned read.
    if n % 8:
        return False
    if signs is not None and q_pack is not None and u_pack is not None and d_row is not None:
        try:
            validate_native_operands(x, signs, q_pack, u_pack, d_row, bias_corr=bias_corr)
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
    validate_native_operands(x, signs, q, u, d, bias_corr=_optional_bias(bias_corr))
    ext = build_extension()
    return ext.fwht_dpas_w4a8(x, signs, q, u, d, _optional_bias(bias_corr), _admitted_module(ext, x.device))


def decode_native(x, signs, q, u, d, bias_corr=None):
    validate_native_operands(x, signs, q, u, d, bias_corr=_optional_bias(bias_corr))
    ext = build_extension()
    return ext.decode_fused(x, signs, q, u, d, _optional_bias(bias_corr), _admitted_module(ext, x.device))


def dpas_native_fixed(xi8, sx, q, u, d, bias_corr=None, write_i32=False):
    # Fixed-int8 input is already quantized but obeys the same device/packed contract.
    validate_native_operands(xi8, None, q, u, d, sx=sx, bias_corr=_optional_bias(bias_corr),
                             expected_x_dtype=torch.int8)
    ext = build_extension()
    ret = ext.dpas_w4a8(xi8, sx, q, u, d, _optional_bias(bias_corr), bool(write_i32), _admitted_module(ext, xi8.device))
    return ret
