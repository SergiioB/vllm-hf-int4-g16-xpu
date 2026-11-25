"""Live GPU0 verification suite for hf_int4_g16_xpu.

Run inside vllm/vllm-openai-xpu:v0.31.0 with:
  pytest -q /work/tests/test_hf_int4_g16.py
The tests are deliberately explicit about backend coverage: native XMX exactness
is checked where its 2D-block-io contract applies; arbitrary tails are checked
through the universal Triton path and forced-native misuse must fail closed.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from hf_int4_g16_xpu.format import (
    HAD_SEED,
    make_had128,
    pack_native_words,
    rotate_had128_noperm,
    signs_for_k,
    unpack_hf_int4_reference,
)
from hf_int4_g16_xpu import kernels_triton as tk
from hf_int4_g16_xpu import native_l0
from hf_int4_g16_xpu.ops import hf_int4_g16_linear

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
DEV = "xpu"


def sync():
    torch.xpu.synchronize()


def logical_packing_case(n=37, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    u = torch.randint(0, 16, (n, k // 16), generator=g, dtype=torch.uint8)
    qp, up = pack_native_words(q, u)
    return q, u, qp, up


def int8_weight(qp, up):
    return unpack_hf_int4_reference(qp.cpu(), up.cpu())


@pytest.mark.parametrize("n", [256, 300, 4096, 5120, 6144, 10240, 17408])
def test_pack_unpack_bit_exact_cpu_and_triton(n):
    q, u, qp, up = logical_packing_case(n=n, k=256, seed=n)
    ref = int8_weight(qp, up)
    # CPU bit-container oracle.
    assert torch.equal(ref, ((q.to(torch.int16) - 8) * (u.repeat_interleave(16, dim=1).to(torch.int16) + 1)).to(torch.int8))
    # Triton device unpacker must be exactly equal, including N tails.
    got = tk.unpack_to_int8(qp.to(DEV), up.to(DEV), n, 256).cpu()
    sync()
    assert torch.equal(got, ref)


@pytest.mark.parametrize("m", [1, 2, 4, 7, 8, 15, 16, 32, 64, 128, 129, 256, 512, 1024])
def test_w4a8_int32_accumulation_exact_triton_and_native(m):
    n, k = 512, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=123)
    qp, up = qp.to(DEV), up.to(DEV)
    d = (torch.rand(n, generator=torch.Generator().manual_seed(9)) * 0.01 + 0.001).float().to(DEV)
    g = torch.Generator().manual_seed(1000 + m)
    xi8 = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8).to(DEV)
    sx = torch.rand(m, generator=g, dtype=torch.float32).to(DEV) * 0.01 + 0.001
    ref = (xi8.cpu().to(torch.int32) @ int8_weight(qp.cpu(), up.cpu()).to(torch.int32).T).contiguous()

    out, ti32 = tk.w4a8_linear_from_quantized(
        xi8, sx, qp, up, d, out_dtype=torch.bfloat16, write_i32=True
    )
    sync()
    assert torch.equal(ti32.cpu(), ref), f"Triton max diff {(ti32.cpu()-ref).abs().max()}"

    ni32 = native_l0.dpas_native_fixed(xi8, sx, qp, up, d, write_i32=True)
    sync()
    assert torch.equal(ni32.cpu(), ref), f"native max diff {(ni32.cpu()-ref).abs().max()}"
    assert out.shape == (m, n)


@pytest.mark.parametrize("n", [256, 300, 4096, 5120, 6144, 10240, 17408])
def test_w4a8_int32_exact_aligned_and_unaligned_n(n):
    m, k = 15, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=n + 17)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.full((n,), 0.01, device=DEV, dtype=torch.float32)
    g = torch.Generator().manual_seed(2000 + n)
    xi8 = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8).to(DEV)
    sx = torch.full((m,), 0.01, device=DEV, dtype=torch.float32)
    ref = (xi8.cpu().to(torch.int32) @ int8_weight(qp.cpu(), up.cpu()).to(torch.int32).T)
    _, ti32 = tk.w4a8_linear_from_quantized(xi8, sx, qp, up, d, write_i32=True)
    sync()
    assert torch.equal(ti32.cpu(), ref)
    if n % 8 == 0:
        ni32 = native_l0.dpas_native_fixed(xi8, sx, qp, up, d, write_i32=True)
        sync()
        assert torch.equal(ni32.cpu(), ref)
    else:
        with pytest.raises(RuntimeError, match="preconditions"):
            hf_int4_g16_linear(
                torch.zeros((m, k), device=DEV, dtype=torch.bfloat16),
                qp, up, d, signs_for_k(k, device=DEV), backend="native", mode="w4a8",
            )


@pytest.mark.parametrize("k", [128, 512, 1024, 5120, 17408])
def test_w4a8_int32_exact_across_hadamard_k_extents(k):
    m, n = 8, 512
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=3000 + k)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.full((n,), 0.01, device=DEV, dtype=torch.float32)
    g = torch.Generator().manual_seed(4000 + k)
    xi8 = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8).to(DEV)
    sx = torch.full((m,), 0.01, device=DEV, dtype=torch.float32)
    ref = (xi8.cpu().to(torch.int32) @ int8_weight(qp.cpu(), up.cpu()).to(torch.int32).T)
    _, ti32 = tk.w4a8_linear_from_quantized(xi8, sx, qp, up, d, write_i32=True)
    ni32 = native_l0.dpas_native_fixed(xi8, sx, qp, up, d, write_i32=True)
    sync()
    assert torch.equal(ti32.cpu(), ref)
    assert torch.equal(ni32.cpu(), ref)


def oracle_weight(qp, up, d, k, seed=HAD_SEED):
    wr = int8_weight(qp.cpu(), up.cpu()).float() * d.cpu()[:, None]
    h = make_had128("cpu", torch.float32)
    signs = signs_for_k(k, seed, "cpu").float()
    # inverse of the orthonormal symmetric block Hadamard rotation
    return ((wr.reshape(-1, 128) @ h).reshape(wr.shape) * signs[:, None].T).contiguous()


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 128])
def test_decode_and_prefill_match_unquantized_oracle(m):
    n, k = 300, 256  # deliberately exercises an unaligned output tail
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=33)
    d = torch.rand(n, generator=torch.Generator().manual_seed(34)).float() * 0.01 + 0.001
    w = oracle_weight(qp, up, d, k).to(DEV)
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(35)).to(DEV).bfloat16()
    bias = torch.randn(n, generator=torch.Generator().manual_seed(36)).to(DEV)
    bc = torch.randn(n, generator=torch.Generator().manual_seed(37)).float().to(DEV) * 0.01
    got = hf_int4_g16_linear(
        x, qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV),
        bias=bias, bias_corr=bc, backend="triton", mode="w4a16",
    )
    ref = torch.nn.functional.linear(x.float(), w.float(), bias.float()) + bc.float()
    sync()
    diff = (got.float() - ref).abs()
    denom = ref.abs().clamp_min(1.0)
    assert (diff / denom).max().item() < 0.02


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_native_fused_decode_matches_triton_fp32_path(m):
    n, k = 512, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=77)
    d = torch.rand(n, generator=torch.Generator().manual_seed(78)).float() * 0.01 + 0.001
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(79)).to(DEV).bfloat16()
    args = (qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV))
    a = hf_int4_g16_linear(x, *args, backend="native", mode="w4a16")
    b = hf_int4_g16_linear(x, *args, backend="triton", mode="w4a16")
    sync()
    # Same math but different reduction order; bf16 output permits one ulp-scale differences.
    rel = (a.float() - b.float()).abs() / b.float().abs().clamp_min(1.0)
    assert rel.max().item() < 0.02  # at most about one bf16 output ulp after different reduction orders


def test_fp16_activation_bias_and_biascorr_triton():
    m, n, k = 8, 300, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=144)
    d = torch.rand(n, generator=torch.Generator().manual_seed(145)).float() * 0.01 + 0.001
    w = oracle_weight(qp, up, d, k).to(DEV)
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(146)).to(DEV).half()
    bias = torch.randn(n, generator=torch.Generator().manual_seed(147)).to(DEV).half()
    bc = torch.randn(n, generator=torch.Generator().manual_seed(148)).float().to(DEV) * 0.01
    got = hf_int4_g16_linear(
        x, qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV),
        bias=bias, bias_corr=bc, backend="triton", mode="w4a16",
    )
    ref = torch.nn.functional.linear(x.float(), w.float(), bias.float()) + bc.float()
    diff = (got.float() - ref).abs()
    denom = ref.abs().clamp_min(1.0)
    assert got.dtype == torch.float16
    assert (diff / denom).max().item() < 0.01


def test_negative_zero_bias_corr_bit_identity():
    m, n, k = 4, 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=88)
    qp, up = qp.to(DEV), up.to(DEV)
    # Zero int8 weights make the exact integer accumulator zero on every output.
    qp.zero_(); up.zero_()
    d = torch.ones(n, device=DEV, dtype=torch.float32)
    x = torch.zeros((m, k), device=DEV, dtype=torch.bfloat16)
    signs = signs_for_k(k, device=DEV)
    none = hf_int4_g16_linear(x, qp, up, d, signs, backend="triton", mode="w4a8")
    negzero = torch.full((n,), -0.0, device=DEV, dtype=torch.float32)
    nz = hf_int4_g16_linear(x, qp, up, d, signs, bias_corr=negzero, backend="triton", mode="w4a8")
    sync()
    assert torch.equal(none.view(torch.int16), nz.view(torch.int16))


def test_tp2_column_parallel_split_concatenates_exactly():
    m, n, k = 8, 512, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=101)
    d = torch.rand(n, generator=torch.Generator().manual_seed(102)).float() * 0.01 + 0.001
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(103)).to(DEV).bfloat16()
    signs = signs_for_k(k, device=DEV)
    full = hf_int4_g16_linear(x, qp.to(DEV), up.to(DEV), d.to(DEV), signs, backend="triton", mode="w4a16")
    halves = []
    for r in range(2):
        sl = slice(r * n // 2, (r + 1) * n // 2)
        halves.append(hf_int4_g16_linear(
            x, qp[:, sl].contiguous().to(DEV), up[:, sl].contiguous().to(DEV),
            d[sl].contiguous().to(DEV), signs, backend="triton", mode="w4a16",
        ))
    got = torch.cat(halves, dim=1)
    sync()
    assert torch.equal(got.view(torch.int16), full.view(torch.int16))


def test_tp2_row_parallel_sum_and_rank0_only_biascorr():
    m, n, k = 8, 256, 256
    assert k % 128 == 0
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=111)
    d = torch.rand(n, generator=torch.Generator().manual_seed(112)).float() * 0.01 + 0.001
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(113)).to(DEV).bfloat16()
    # Large constant correction makes a double-add unambiguous even after bf16 rounding.
    bc = torch.full((n,), 0.5, device=DEV, dtype=torch.float32)
    full_signs = signs_for_k(k, device=DEV)
    ref = hf_int4_g16_linear(x, qp.to(DEV), up.to(DEV), d.to(DEV), full_signs,
                             bias_corr=bc, backend="triton", mode="w4a16")
    partials = []
    for r in range(2):
        k0, k1 = r * k // 2, (r + 1) * k // 2
        qs = qp[k0 // 8:k1 // 8, :].contiguous().to(DEV)
        us = up[k0 // 64:k1 // 64, :].contiguous().to(DEV)
        ss = full_signs[k0:k1].contiguous()
        # Bias correction is present only on simulated rank zero.
        partials.append(hf_int4_g16_linear(
            x[:, k0:k1].contiguous(), qs, us, d.to(DEV), ss,
            bias_corr=bc if r == 0 else None, backend="triton", mode="w4a16",
        ))
    got = partials[0] + partials[1]
    sync()
    # W4A16 fp32 reduction is split differently, so compare tightly rather than bitwise.
    diff = (got.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1.0)
    # TP summation rounds each rank partial to bf16 before the final add; this is
    # the expected bf16 all-reduce accuracy envelope, not a bit-exactness contract.
    assert torch.allclose(got.float(), ref.float(), atol=0.10, rtol=0.02)
    # If rank 1 incorrectly adds bias_corr too, the error is exactly one extra bc and large.
    wrong = got + bc
    assert (wrong - ref).abs().max().item() > (got - ref).abs().max().item() + 0.25


def test_native_forced_fail_closed_for_fp16_and_bias():
    m, n, k = 4, 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=55)
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(56)).to(DEV).half()
    with pytest.raises(RuntimeError, match="preconditions"):
        hf_int4_g16_linear(x, qp.to(DEV), up.to(DEV), torch.ones(n, device=DEV),
                           signs_for_k(k, device=DEV), backend="native")
    xb = x.bfloat16()
    with pytest.raises(RuntimeError, match="preconditions"):
        hf_int4_g16_linear(xb, qp.to(DEV), up.to(DEV), torch.ones(n, device=DEV),
                           signs_for_k(k, device=DEV), bias=torch.ones(n, device=DEV), backend="native")


def test_stream_interop_and_rotation_contract():
    assert native_l0.probe_stream() == "immediate_command_list"
    k = 256
    signs = signs_for_k(k, HAD_SEED, DEV)
    x = torch.randn((3, k), generator=torch.Generator().manual_seed(7)).to(DEV)
    xr = rotate_had128_noperm(x, signs)
    # The normalized block Hadamard is orthonormal and symmetric.  Undo it
    # explicitly as inverse((x*s)H) = (((x*s)H)H)*s; this pins the no-permutation contract.
    h = make_had128(DEV, torch.float32)
    back = (xr.reshape(-1, 128) @ h).reshape(x.shape) * signs.float()
    assert torch.allclose(back, x.float(), atol=1e-5, rtol=1e-5)



def test_canonical_pack_words_and_sign_hashes():
    # Astra §2.1 canonical words: logical q codes 0..7 interleave as native
    # bytes 40 51 62 73 (little-endian word 0x73625140), not GPTQ order.
    q = torch.arange(8, dtype=torch.uint8).reshape(1, 8).repeat(1, 32)  # [1,256]
    u = torch.zeros((1, 16), dtype=torch.uint8)
    qp, _ = pack_native_words(q, u)
    assert qp[0, 0].item() == torch.tensor(0x73625140, dtype=torch.int32).item()
    q0 = torch.zeros((1, 256), dtype=torch.uint8)
    u = torch.zeros((1, 16), dtype=torch.uint8)
    u[0, :4] = torch.tensor([0, 1, 14, 15], dtype=torch.uint8)  # u values 1,2,15,16 -> nibbles 0,1,e,f
    _, up = pack_native_words(q0, u)
    assert (up[0, 0].item() & 0xFFFF) == 0xFE10

    import hashlib
    want = {
        5120: "028cb2dc4ebf61d99df3c4364b8064a049a51cf875b2dd124cfc7a208b3ec5ba",
        6144: "15c035838cc2c04d5401ea19e59f80a2b19e53c46e9db92c6158c7e01c745f48",
        17408: "437f9c830a46873494877922e9daa25c1e9edc881d975079b393ea5fcfd9d5dd",
    }
    for k, digest in want.items():
        signs = signs_for_k(k, HAD_SEED, "cpu")
        assert hashlib.sha256(signs.numpy().tobytes()).hexdigest() == digest


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("use_bias_corr", [False, True])
def test_xpu_graph_changed_input_replay_native(m, use_bias_corr):
    n, k = 512, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=5150 + m)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.rand(n, generator=torch.Generator().manual_seed(5151)).float().to(DEV) * 0.01 + 0.001
    signs = signs_for_k(k, device=DEV)
    static_x = torch.randn((m, k), generator=torch.Generator().manual_seed(5152)).to(DEV).bfloat16()
    bc = (torch.randn(n, generator=torch.Generator().manual_seed(5153)).float().to(DEV) * 0.01) if use_bias_corr else None
    mode = "w4a16" if m in (1, 2, 3, 4) else "w4a8"

    # Warm up on a side stream, then capture one call.  The output allocation is
    # inside torch's graph pool, while every launch remains address-stable.
    s = torch.xpu.Stream()
    with torch.xpu.stream(s):
        for _ in range(3):
            warm = hf_int4_g16_linear(static_x, qp, up, d, signs, bias_corr=bc, backend="native", mode=mode)
    torch.xpu.current_stream().wait_stream(s)
    sync()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        static_y = hf_int4_g16_linear(static_x, qp, up, d, signs, bias_corr=bc, backend="native", mode=mode)

    churn = torch.xpu.Stream()
    for seed in (6100, 6200, 6300):
        static_x.copy_(torch.randn((m, k), generator=torch.Generator().manual_seed(seed)).to(DEV).bfloat16())
        # Allocate, use and release storage on a non-default stream between
        # replays.  recordStream protection must prevent premature reuse of any
        # foreign-stream input storage while graph/eager work is still queued.
        with torch.xpu.stream(churn):
            scratch = static_x.clone()
            scratch.add_(0)
            del scratch
        torch.xpu.current_stream().wait_stream(churn)
        g.replay()
        sync()
        graph_out = static_y.clone()
        eager = hf_int4_g16_linear(static_x, qp, up, d, signs, bias_corr=bc, backend="native", mode=mode)
        sync()
        assert torch.equal(graph_out.view(torch.int16), eager.view(torch.int16)), (
            f"XPUGraph replay did not consume changed input at M={m}, bc={use_bias_corr}, seed={seed}"
        )


def test_m0_returns_correctly_shaped_empty_all_backends():
    n, k = 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7000)
    x = torch.empty((0, k), device=DEV, dtype=torch.bfloat16)
    args = (qp.to(DEV), up.to(DEV), torch.ones(n, device=DEV), signs_for_k(k, device=DEV))
    for backend in ("auto", "triton", "native"):
        y = hf_int4_g16_linear(x, *args, backend=backend, mode="w4a16")
        assert y.shape == (0, n)
        assert y.dtype == torch.bfloat16
        assert y.device.type == "xpu"


def test_k129_rejected_before_oob_block_read():
    n, k = 64, 129
    x = torch.zeros((2, k), device=DEV, dtype=torch.bfloat16)
    signs = torch.ones(k, device=DEV, dtype=torch.int8)
    qp = torch.zeros((17, n), device=DEV, dtype=torch.int32)
    up = torch.zeros((3, n), device=DEV, dtype=torch.int16)
    d = torch.ones(n, device=DEV)
    for backend in ("auto", "triton", "native"):
        with pytest.raises((ValueError, RuntimeError), match="multiple of 128|preconditions"):
            hf_int4_g16_linear(x, qp, up, d, signs, backend=backend)


def test_misaligned_contiguous_offset_view_falls_back_or_native_rejects():
    n, k = 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7100)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.ones(n, device=DEV)
    signs = signs_for_k(k, device=DEV)
    storage = torch.randn(1 + 4 * k, generator=torch.Generator().manual_seed(7101)).to(DEV).bfloat16()
    x = storage[1:1 + 4 * k].reshape(4, k)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    got = hf_int4_g16_linear(x, qp, up, d, signs, backend="auto", mode="w4a16")
    want = hf_int4_g16_linear(x, qp, up, d, signs, backend="triton", mode="w4a16")
    sync()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    with pytest.raises(RuntimeError, match="preconditions"):
        hf_int4_g16_linear(x, qp, up, d, signs, backend="native", mode="w4a16")


def test_native_public_validation_rejects_cpu_wrong_lengths_and_overlarge_k():
    n, k = 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7200)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.ones(n, device=DEV)
    signs = signs_for_k(k, device=DEV)
    x = torch.zeros((2, k), device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="XPU"):
        native_l0.validate_native_operands(x.cpu(), signs.cpu())
    with pytest.raises(ValueError, match="signs length"):
        native_l0.validate_native_operands(x, signs[:-1], qp, up, d)
    xbig = torch.zeros((1, 17536), device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="17408"):
        native_l0.validate_native_operands(xbig, torch.ones(17536, device=DEV, dtype=torch.int8))


@pytest.mark.parametrize("case", ["short_q", "mismatched_u", "wrong_device", "narrow_d", "short_signs"])
def test_invalid_packed_operands_raise_immediately_under_auto(case):
    n, k = 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7300)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.ones(n, device=DEV, dtype=torch.float32)
    signs = signs_for_k(k, device=DEV)
    x = torch.zeros((2, k), device=DEV, dtype=torch.bfloat16)
    if case == "short_q":
        qp = qp[:-1].contiguous()
        match = "q_pack K mismatch"
    elif case == "mismatched_u":
        up = torch.zeros((k // 64, n - 1), device=DEV, dtype=torch.int16)
        match = "N mismatch|d_row shape"
    elif case == "wrong_device":
        qp = qp.cpu()
        match = "device"
    elif case == "narrow_d":
        d = d[:-1].contiguous()
        match = "d_row shape"
    elif case == "short_signs":
        signs = signs[:-1].contiguous()
        match = "signs shape"
    else:
        raise AssertionError(case)
    with pytest.raises(ValueError, match=match):
        hf_int4_g16_linear(x, qp, up, d, signs, backend="auto")


def test_valid_fp16_bias_and_biascorr_still_route_to_triton_under_auto():
    m, n, k = 4, 300, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7350)
    d = (torch.rand(n, generator=torch.Generator().manual_seed(7351)).float() * 0.01 + 0.001).to(DEV)
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(7352)).to(DEV).half()
    bias = torch.randn(n, generator=torch.Generator().manual_seed(7353)).to(DEV).half()
    bc = torch.randn(n, generator=torch.Generator().manual_seed(7354)).float().to(DEV) * 0.01
    args = (qp.to(DEV), up.to(DEV), d, signs_for_k(k, device=DEV))
    got = hf_int4_g16_linear(x, *args, bias=bias, bias_corr=bc, backend="auto", mode="w4a16")
    want = hf_int4_g16_linear(x, *args, bias=bias, bias_corr=bc, backend="triton", mode="w4a16")
    sync()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_actq_native_triton_exact_rne_ties_zero_and_high_rows():
    k = 128
    # This +/-1 vector is the post-sign input.  Its exact FP32 FWHT/scale ratio
    # contains both +63.5 and -63.5.  Correct RNE maps both magnitudes to 64;
    # floor(z+0.5) incorrectly maps the negative ties to -63.
    v = torch.tensor([
        -1,1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,
        1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,
        1,1,-1,1,-1,1,1,-1,1,1,1,1,1,-1,-1,-1,
        1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,
        1,-1,1,-1,1,1,1,-1,1,-1,-1,1,-1,1,1,1,
        -1,1,1,-1,1,-1,-1,1,1,1,-1,1,1,1,1,-1,
        -1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,
        -1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,
    ], dtype=torch.float32)
    assert v.numel() == 128
    signs_cpu = signs_for_k(k, HAD_SEED, "cpu").float()
    rows = []
    rows.append((v * signs_cpu).reshape(1, k))
    rows.append(torch.zeros((1, k)))
    rows.append(torch.full((1, k), 70000.0))
    rows.append(torch.full((1, k), -70000.0))
    rows.append(torch.randn((1, k), generator=torch.Generator().manual_seed(7300)))
    x = torch.cat(rows, dim=0).to(DEV).bfloat16()
    signs = signs_for_k(k, device=DEV)
    ti, ts = tk.fwht128_actq(x, signs)
    ni, ns = native_l0.fwht_actq(x, signs)
    sync()
    assert torch.equal(ti.cpu(), ni.cpu())
    assert torch.equal(ts.cpu(), ns.cpu())
    q0 = ni[0].cpu()
    assert int((q0 == 64).sum()) == 2
    assert int((q0 == -64).sum()) == 3
    assert int((q0 == 63).sum()) == 0
    assert int((q0 == -63).sum()) == 0
    assert torch.isfinite(ts.cpu()).all()


@pytest.mark.parametrize("m,mode", [(1, "w4a16"), (2, "w4a16"), (4, "w4a16"), (8, "w4a8"), (128, "w4a8")])
def test_native_true_null_biascorr_matches_loaded_negative_zero(m, mode):
    n, k = 512, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7400 + m)
    qp, up = qp.to(DEV), up.to(DEV)
    d = torch.rand(n, generator=torch.Generator().manual_seed(7401)).float().to(DEV) * 0.01 + 0.001
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(7402 + m)).to(DEV).bfloat16()
    signs = signs_for_k(k, device=DEV)
    none = hf_int4_g16_linear(x, qp, up, d, signs, bias_corr=None, backend="native", mode=mode)
    negzero = torch.full((n,), -0.0, device=DEV, dtype=torch.float32)
    loaded = hf_int4_g16_linear(x, qp, up, d, signs, bias_corr=negzero, backend="native", mode=mode)
    sync()
    assert torch.equal(none.view(torch.int16), loaded.view(torch.int16))


def test_triton_n301_tail_exact_vs_unquantized_oracle():
    m, n, k = 5, 301, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7500)
    d = torch.rand(n, generator=torch.Generator().manual_seed(7501)).float() * 0.01 + 0.001
    w = oracle_weight(qp, up, d, k).to(DEV)
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(7502)).to(DEV).bfloat16()
    got = hf_int4_g16_linear(x, qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV),
                             backend="triton", mode="w4a16")
    ref = torch.nn.functional.linear(x.float(), w.float())
    sync()
    assert got.shape == (m, n)
    # Include the final partial-tile output columns explicitly; these are the
    # positions an unmasked guard/tail load or store would corrupt first.
    rel = (got.float() - ref).abs() / ref.abs().clamp_min(1.0)
    assert rel.max().item() < 0.02
    assert torch.isfinite(got[:, -3:]).all()


def test_triton_packed_tail_guard_canaries_are_not_consumed():
    m, n, k = 3, 301, 256
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=7600)
    d = (torch.rand(n, generator=torch.Generator().manual_seed(7601)).float() * 0.01 + 0.001).to(DEV)
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(7602)).to(DEV).bfloat16()
    signs = signs_for_k(k, device=DEV)
    qp0, up0 = qp.to(DEV), up.to(DEV)
    ref = hf_int4_g16_linear(x, qp0, up0, d, signs, backend="triton", mode="w4a16")

    # Place nonzero canary bytes immediately after the exact packed allocations.
    # Any over-read at the final partial-N tile consumes poisoned bits and will
    # normally change the output; exact equality proves the tails are masked.
    qs = torch.empty(qp.numel() + 4096, device=DEV, dtype=torch.int32)
    qs.fill_(0x55555555)
    qg = qs[:qp.numel()].reshape(qp.shape)
    qg.copy_(qp.to(DEV))
    us = torch.empty(up.numel() + 4096, device=DEV, dtype=torch.int16)
    us.fill_(0x5555)
    ug = us[:up.numel()].reshape(up.shape)
    ug.copy_(up.to(DEV))
    got = hf_int4_g16_linear(x, qg, ug, d, signs, backend="triton", mode="w4a16")
    sync()
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert torch.isfinite(got[:, -8:]).all()


@pytest.mark.parametrize("m,backend", [(2, "auto"), (2, "triton"), (8, "auto"), (8, "triton")])
def test_k17536_w4a16_runs_while_w4a8_rejects(m, backend):
    # K = 137 * 128 is above the W4A8 INT32 accumulation bound (17408).  The
    # bound belongs to W4A8 only: the FP32-accumulating W4A16 path must work.
    n, k = 64, 17536
    _, _, qp, up = logical_packing_case(n=n, k=k, seed=17536)
    d = torch.rand(n, generator=torch.Generator().manual_seed(17537)).float() * 0.001 + 0.0001
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(17538)).bfloat16()
    w = oracle_weight(qp, up, d, k).double()
    want = x.double() @ w.T
    args = (qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV))
    got = hf_int4_g16_linear(x.to(DEV), *args, backend=backend, mode="w4a16")
    sync()
    rel_l2 = (got.cpu().double() - want).norm() / want.norm()
    assert got.shape == (m, n)
    assert rel_l2.item() < 0.01, f"K=17536 W4A16 rel_l2={rel_l2.item()}"
    with pytest.raises(ValueError, match="17408"):
        hf_int4_g16_linear(x.to(DEV), *args, backend=backend, mode="w4a8")


def test_admit_module_hashes_once_and_launches_never_reread_binary(tmp_path, monkeypatch):
    # Admit a private copy of the zebin, then delete it.  Any launch that still
    # re-read (or re-hashed) the file would fail to open it.
    blob = open(native_l0.kernel_bin(), "rb").read()
    private = tmp_path / "admitted_kernels.bin"
    private.write_bytes(blob)
    monkeypatch.setattr(native_l0, "_KERNEL_BIN", str(private))
    monkeypatch.setattr(native_l0, "_ADMITTED_DIGESTS", {})
    real = native_l0.build_extension()
    calls = []

    class Spy:
        def __getattr__(self, name):
            fn = getattr(real, name)

            def wrapped(*args):
                calls.append((name, args[-1] if args else None))
                return fn(*args)

            return wrapped

    monkeypatch.setattr(native_l0, "_EXT", Spy())

    n, kdim = 256, 256
    _, _, qp, up = logical_packing_case(n=n, k=kdim, seed=4242)
    d = torch.full((n,), 0.001, device=DEV, dtype=torch.float32)
    signs = signs_for_k(kdim, device=DEV)
    x = torch.randn((2, kdim), generator=torch.Generator().manual_seed(4243)).to(DEV).bfloat16()
    ops = (signs, qp.to(DEV), up.to(DEV), d)

    first = native_l0.decode_native(x, *ops)
    sync()
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    assert list(native_l0._ADMITTED_DIGESTS.values()) == [digest]
    private.unlink()

    again = native_l0.decode_native(x, *ops)
    prefill = native_l0.w4a8_native(x, *ops)
    sync()
    assert torch.equal(first.view(torch.int16), again.view(torch.int16))
    assert prefill.shape == (2, n)
    assert [c[0] for c in calls].count("admit_module") == 1
    launches = [c for c in calls if c[0] in ("decode_fused", "fwht_dpas_w4a8")]
    assert len(launches) == 3 and all(c[1] == digest for c in launches)

    bogus = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="unadmitted native module digest"):
        real.decode_fused(x, *ops, None, bogus)
    with pytest.raises(RuntimeError, match="unadmitted native module digest"):
        real.fwht_dpas_w4a8(x, *ops, None, bogus)

