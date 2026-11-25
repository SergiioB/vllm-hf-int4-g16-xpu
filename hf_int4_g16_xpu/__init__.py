"""HF-INT4-G16 XPU quantization plugin.

Canonical on-disk format: Had128NoPerm rotation, progressive INT4 x UINT4
group-16 quantization, Xe2-DPAS-native packed little-endian words.
"""

from .format import (
    FORMAT_VERSION,
    GROUP_SIZE,
    HADAMARD_BLOCK,
    HAD_SEED,
    make_had128,
    signs_for_k,
    quantize_pack_hf_int4,
    unpack_hf_int4_reference,
)

__all__ = [
    "FORMAT_VERSION",
    "GROUP_SIZE",
    "HADAMARD_BLOCK",
    "HAD_SEED",
    "make_had128",
    "signs_for_k",
    "quantize_pack_hf_int4",
    "unpack_hf_int4_reference",
]
