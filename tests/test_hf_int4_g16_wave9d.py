"""Fail-closed regressions from the v0.2.0 reviews, and grouped-by-expert prefill.

Each blocker test starts from a valid case, breaks exactly one thing, and
requires either a hard error or the exact result of the safe path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from hf_int4_g16_xpu import kernels_triton as tk
from hf_int4_g16_xpu import moe as moe_module
from hf_int4_g16_xpu import native_l0
from hf_int4_g16_xpu.converter import DEFAULT_SKIP, convert, dry_run_layer_mapping
from hf_int4_g16_xpu.format import HAD_BLOCK_RULE_V2, sign_sha256_for_k, signs_for_k
from hf_int4_g16_xpu.moe import GROUPED_MIN_TOKENS, PackedExperts, hf_int4_g16_moe, moe_grouped_gemm, moe_pair_gemm
from hf_int4_g16_xpu.moe_admission import FullPrecisionBanks, PackedCoverage
from hf_int4_g16_xpu.ops import hf_int4_g16_linear

from test_hf_int4_g16_had64 import _fake_tp, _gemma4_architecture, _oracle, _packed, _x
from test_hf_int4_g16_moe import (
    E,
    H,
    I,
    LAYOUTS,
    TOPK,
    _admit,
    _bits,
    _build_moe,
    _convert_layout,
    _expert_items,
    _forward,
    _gen,
    _moe_case,
    _oracle_moe,
    _random_stack,
    _read_checkpoint,
    _rel,
    _rewrite,
    _routing,
    _run,
    _source_tensors,
    _source_weights,
    _tiny_hf_moe,
    _to_device,
    _write_checkpoint,
    vllm_distributed,  # noqa: F401  (fixture)
    vllm_env,  # noqa: F401  (fixture)
)

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
DEV = "xpu"


def _offset_view(t: torch.Tensor, offset_bytes: int) -> torch.Tensor:
    """A contiguous view of ``t``'s values that starts ``offset_bytes`` into new storage."""
    skip = offset_bytes // t.element_size()
    storage = torch.zeros(t.numel() + skip, dtype=t.dtype, device=t.device)
    view = storage[skip:].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous() and view.storage_offset() == skip
    assert view.data_ptr() % 64 == offset_bytes % 64
    return view


# ===========================================================================
# 1. Xe2 2D block-io surfaces (cl_intel_subgroup_2d_block_io 6.13.X.6)
# ===========================================================================
@pytest.mark.parametrize("address,width,height,pitch,match", [
    (16, 64, 1, 64, "64-byte aligned"),
    (0, 32, 1, 32, "surface width 32"),
    (0, (1 << 24) + 4, 1, (1 << 24) + 16, "surface width"),
    (0, 66, 1, 80, "multiple of 4"),
    (0, 64, 0, 64, "surface height 0"),
    (0, 64, (1 << 24) + 1, 64, "surface height"),
    (0, 128, 1, 64, "pitch 64"),
    (0, 68, 1, 72, "multiple of 16"),
])
def test_block2d_surface_rules(address, width, height, pitch, match):
    assert match in native_l0.block2d_surface_error("s", address, width, height, pitch)
    assert native_l0.block2d_surface_error("s", 64, 64, 1, 64) is None
    assert native_l0.block2d_surface_error("s", 0, 1 << 24, 1 << 24, 1 << 24) is None


def _dense_args(n, k, seed):
    _, _, qp, up, d = _packed(n, k, seed=seed)
    return (qp, up, d), (qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV))


@pytest.mark.parametrize("m,mode", [(1, "w4a16"), (4, "w4a16"), (8, "w4a8"), (40, "w4a8")])
@pytest.mark.parametrize("offset", [16, 32, 48])
def test_dense_q_offset_view_is_rejected_native_and_safe_under_auto(m, mode, offset):
    """A 16-byte-aligned but not cache-line-aligned weight view passes every
    allocator-aligned test and is still an illegal 2D surface."""
    n, k = 256, 256
    _, args = _dense_args(n, k, seed=offset + m)
    q = _offset_view(args[0], offset)
    assert q.data_ptr() % 16 == 0
    x = _x(m, k, 3).to(DEV)
    with pytest.raises(RuntimeError, match="preconditions"):
        hf_int4_g16_linear(x, q, *args[1:], backend="native", mode=mode)
    with pytest.raises(ValueError, match="64-byte aligned"):
        native_l0.validate_native_operands(x, args[3], q, args[1], args[2])
    got = hf_int4_g16_linear(x, q, *args[1:], backend="auto", mode=mode)
    want = hf_int4_g16_linear(x, *args, backend="triton", mode=mode)
    assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("n", [8, 24, 40, 264])
@pytest.mark.parametrize("m,mode", [(2, "w4a16"), (8, "w4a8")])
def test_dense_narrow_or_ragged_surface_is_rejected_native_and_safe_under_auto(n, m, mode):
    k = 256
    (qp, up, d), args = _dense_args(n, k, seed=n + m)
    x = _x(m, k, 5)
    with pytest.raises(RuntimeError, match="preconditions"):
        hf_int4_g16_linear(x.to(DEV), *args, backend="native", mode=mode)
    with pytest.raises(ValueError, match="N%16==0"):
        native_l0.validate_native_operands(x.to(DEV), args[3], *args[:3])
    got = hf_int4_g16_linear(x.to(DEV), *args, backend="auto", mode=mode)
    assert torch.equal(_bits(got), _bits(hf_int4_g16_linear(x.to(DEV), *args, backend="triton", mode=mode)))
    if mode == "w4a16":
        assert _rel(got.cpu().float(), _oracle(x, qp, up, d)) < 4e-3


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_n16_is_legal_for_decode_but_not_for_the_dpas_output_surface(m):
    """N=16 gives a 64 B weight surface (legal) and a 32 B BF16 output surface,
    which only the DPAS kernels 2D-write."""
    n, k = 16, 256
    (qp, up, d), args = _dense_args(n, k, seed=m)
    x = _x(m, k, 6)
    got = hf_int4_g16_linear(x.to(DEV), *args, backend="native", mode="w4a16").cpu()
    assert _rel(got.float(), _oracle(x, qp, up, d)) < 4e-3
    with pytest.raises(RuntimeError, match="preconditions"):
        hf_int4_g16_linear(x.to(DEV), *args, backend="native", mode="w4a8")
    auto = hf_int4_g16_linear(x.to(DEV), *args, backend="auto", mode="w4a8")
    assert torch.equal(_bits(auto), _bits(hf_int4_g16_linear(x.to(DEV), *args, backend="triton", mode="w4a8")))
    xi8 = torch.zeros((m, k), dtype=torch.int8, device=DEV)
    with pytest.raises(ValueError, match="bf16 output"):
        native_l0.dpas_native_fixed(xi8, torch.ones(m, device=DEV), *args[:3])


def test_dpas_int8_activation_view_must_be_cache_line_aligned():
    n, k, m = 256, 256, 4
    _, args = _dense_args(n, k, seed=9)
    xi8 = _offset_view(torch.ones((m, k), dtype=torch.int8, device=DEV), 16)
    with pytest.raises(ValueError, match="int8 activation effective base"):
        native_l0.dpas_native_fixed(xi8, torch.ones(m, device=DEV), *args[:3])


def test_cpp_enforces_the_same_surface_rules_without_the_python_gate():
    """Call the extension directly: the C++ side must not rely on Python."""
    ext = native_l0.build_extension()
    n, k = 256, 256
    _, (q, u, d, signs) = _dense_args(n, k, seed=11)
    x = _x(2, k, 12).to(DEV)
    digest = native_l0._admitted_module(ext, x.device)
    with pytest.raises(RuntimeError, match="q_pack effective base address is not 64-byte aligned"):
        ext.decode_fused(x, signs, _offset_view(q, 16), u, d, None, digest)
    with pytest.raises(RuntimeError, match="N>=16 and N%16==0"):
        ext.decode_fused(x, signs, q[:, :8].contiguous(), u[:, :8].contiguous(), d[:8].contiguous(), None, digest)
    xi8 = torch.zeros((4, k), dtype=torch.int8, device=DEV)
    sx = torch.ones(4, device=DEV)
    with pytest.raises(RuntimeError, match="int8 activation effective base"):
        ext.dpas_w4a8(_offset_view(xi8, 16), sx, q, u, d, None, False, digest)
    with pytest.raises(RuntimeError, match="bf16 output 2D block-io surface width 32"):
        ext.dpas_w4a8(xi8, sx, q[:, :16].contiguous(), u[:, :16].contiguous(), d[:16].contiguous(),
                      None, False, digest)

    raw, _ = _random_stack(3, 64, 256, seed=13, bias=False)
    experts = _to_device(raw, 256)
    ids = torch.zeros(2, dtype=torch.int32, device=DEV)
    shifted = _offset_view(experts.q_pack, 16)
    with pytest.raises(RuntimeError, match="q_pack effective base address is not 64-byte aligned"):
        ext.moe_pair(x, experts.signs, shifted, experts.u_pack, experts.d_row, None, ids, 1, False, digest)
    narrow = (experts.q_pack[:, :, :8].contiguous(), experts.u_pack[:, :, :8].contiguous(),
              experts.d_row[:, :8].contiguous())
    with pytest.raises(RuntimeError, match="N>=16 and N%16==0"):
        ext.moe_pair(x, experts.signs, *narrow, None, ids, 1, False, digest)
    buckets = native_l0.moe_bucket_native(ids, 3)
    rotated = native_l0.moe_rotate_native(x, experts.signs)
    with pytest.raises(RuntimeError, match="q_pack effective base address"):
        ext.moe_tile(rotated, shifted, experts.u_pack, experts.d_row, None, buckets, 1, False, digest)
    with pytest.raises(RuntimeError, match="N>=16 and N%16==0"):
        ext.moe_tile(rotated, *narrow, None, buckets, 1, False, digest)


@pytest.mark.parametrize("case", ["offset16", "n8", "n24"])
@pytest.mark.parametrize("m", [4, 32])
def test_moe_illegal_surface_rejected_native_and_safe_under_auto(case, m):
    experts, slots, hidden = 8, 4, 256
    if case == "offset16":
        raw13, _ = _random_stack(experts, 384, hidden, seed=21, bias=True)
        raw2, _ = _random_stack(experts, hidden, 192, seed=22, bias=True)
        w13, w2 = _to_device(raw13, hidden), _to_device(raw2, 192)
        w13 = PackedExperts(_offset_view(w13.q_pack, 16), w13.u_pack, w13.d_row, w13.signs, w13.bias_corr)
    else:
        n = int(case[1:])
        raw13, _ = _random_stack(experts, n, hidden, seed=23, bias=False)
        w13 = _to_device(raw13, hidden)
    x = torch.randn((m, hidden), generator=_gen(24)).bfloat16().to(DEV)
    ids = torch.randint(0, experts, (m * slots,), generator=_gen(25), dtype=torch.int32).to(DEV)
    with pytest.raises(RuntimeError, match="preconditions"):
        moe_pair_gemm(x, ids, w13, slots, torch.bfloat16, "native")
    auto = moe_pair_gemm(x, ids, w13, slots, torch.bfloat16, "auto")
    assert torch.equal(_bits(auto), _bits(moe_pair_gemm(x, ids, w13, slots, torch.bfloat16, "triton")))
    if case == "offset16":
        weights, topk = _routing(m, experts, slots, 26)
        arguments = (x, weights.to(DEV), topk.to(DEV), w13, w2, "silu")
        with pytest.raises(RuntimeError, match="preconditions"):
            hf_int4_g16_moe(*arguments, backend="native")
        assert torch.equal(_bits(hf_int4_g16_moe(*arguments, backend="auto")),
                           _bits(hf_int4_g16_moe(*arguments, backend="triton")))


# ===========================================================================
# 2. Empty expert tables
# ===========================================================================
def _empty_stack(n, k):
    return PackedExperts(
        torch.zeros((0, k // 8, n), dtype=torch.int32, device=DEV),
        torch.zeros((0, k // 64, n), dtype=torch.int16, device=DEV),
        torch.zeros((0, n), dtype=torch.float32, device=DEV), signs_for_k(k, device=DEV))


@pytest.mark.parametrize("backend", ["auto", "native", "triton"])
@pytest.mark.parametrize("m", [0, 1, 32])
def test_empty_expert_table_is_rejected_before_dispatch(backend, m):
    w13, w2 = _empty_stack(384, 256), _empty_stack(256, 192)
    x = torch.zeros((m, 256), dtype=torch.bfloat16, device=DEV)
    ids = torch.zeros((m, 2), dtype=torch.int64, device=DEV)
    with pytest.raises(ValueError, match="empty expert table"):
        hf_int4_g16_moe(x, torch.ones((m, 2), device=DEV), ids, w13, w2, "silu", backend=backend)
    with pytest.raises(ValueError, match="empty expert table"):
        moe_pair_gemm(x, ids.reshape(-1).int(), w13, 2, backend=backend)


def test_empty_expert_table_is_rejected_by_raw_launchers():
    w13 = _empty_stack(384, 256)
    x = torch.zeros((2, 256), dtype=torch.bfloat16, device=DEV)
    ids = torch.zeros(2, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="empty expert table"):
        tk.moe_pair_gemm(x, ids, w13.q_pack, w13.u_pack, w13.d_row, w13.signs, None, 1,
                         torch.zeros((2, 384), device=DEV, dtype=torch.bfloat16))
    ext = native_l0.build_extension()
    digest = native_l0._admitted_module(ext, x.device)
    with pytest.raises(RuntimeError, match="empty expert table"):
        ext.moe_pair(x, w13.signs, w13.q_pack, w13.u_pack, w13.d_row, None, ids, 1, False, digest)
    with pytest.raises(RuntimeError, match="empty expert table"):
        native_l0.moe_bucket_native(ids, 0)


# ===========================================================================
# 3. int64 route ids and dtype validation
# ===========================================================================
@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("m", [4, 32])
def test_int64_ids_are_masked_before_narrowing(backend, m):
    """2**32 + e narrows to expert e.  It must contribute zero instead."""
    experts, slots = 8, 3
    inputs, stacks, reference = _moe_case(m, experts, slots, 256, 192, True, seed=1700 + m)
    x, weights, ids = inputs
    aliased = ids.clone()
    aliased[0, 0] = 2**32          # would alias expert 0
    aliased[1, 1] = 2**32 + 5      # would alias expert 5
    aliased[2, 2] = -(2**32) + 2   # would alias expert 2
    aliased[3] = torch.tensor([2**63 - 1, -(2**63), 2**31])
    masked = torch.where((aliased >= 0) & (aliased < experts), aliased, -1)
    assert torch.equal(aliased[0, :1].to(torch.int32), torch.zeros(1, dtype=torch.int32))  # the hazard is real
    got = _run((x, weights, aliased), stacks, "silu", backend=backend)
    assert torch.equal(_bits(got), _bits(_run((x, weights, masked), stacks, "silu", backend=backend)))
    assert torch.equal(got[3], torch.zeros_like(got[3]))
    assert _rel(got.float(), _oracle_moe(x, weights, masked, *reference, "silu").double()) < 4e-3


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("m", [1, 4, 32])
def test_int32_out_of_range_ids_contribute_zero(backend, m):
    """int32 routes (vLLM's router) skip the host-side mask: the kernels, pair
    and grouped, must zero every out-of-range id themselves."""
    experts, slots = 8, 4
    inputs, stacks, reference = _moe_case(m, experts, slots, 256, 192, True, seed=1750 + m)
    x, weights, ids = inputs
    ids = ids.to(torch.int32)
    ids[0] = torch.tensor([-1, experts, 2**31 - 1, -(2**31)], dtype=torch.int32)
    got = _run((x, weights, ids), stacks, "silu", backend=backend)
    assert torch.equal(got[0], torch.zeros_like(got[0]))
    assert torch.equal(_bits(got), _bits(_run((x, weights, ids.to(torch.int64)), stacks, "silu", backend=backend)))
    if m > 1:
        assert _rel(got.float(), _oracle_moe(x, weights, ids, *reference, "silu").double()) < 4e-3


@pytest.mark.parametrize("backend", ["auto", "triton"])
@pytest.mark.parametrize("field,dtype", [
    ("x", torch.int32), ("x", torch.int8), ("x", torch.int16), ("x", torch.float64),
    ("topk_weights", torch.int64), ("topk_weights", torch.int32), ("topk_weights", torch.bool),
    ("topk_weights", torch.float64),
])
def test_unsupported_activation_and_weight_dtypes_are_rejected(backend, field, dtype):
    inputs, (w13, w2), _ = _moe_case(2, 4, 2, 256, 192, False, seed=1800)
    arguments = dict(zip(("x", "topk_weights", "topk_ids"), (t.to(DEV) for t in inputs)))
    arguments[field] = arguments[field].to(dtype)
    with pytest.raises(ValueError, match="bf16, fp16 or fp32"):
        hf_int4_g16_moe(**arguments, w13=w13, w2=w2, activation="silu", backend=backend)


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_pair_gemm_rejects_integer_activations_and_outputs(backend):
    inputs, (w13, _), _ = _moe_case(2, 4, 2, 256, 192, False, seed=1801)
    x, ids = inputs[0].to(DEV), inputs[2].reshape(-1).int().to(DEV)
    with pytest.raises(ValueError, match="bf16, fp16 or fp32"):
        moe_pair_gemm(x.to(torch.int32), ids, w13, 2, backend=backend)
    with pytest.raises(ValueError, match="bf16, fp16 or fp32"):
        moe_pair_gemm(x, ids, w13, 2, torch.int32, backend)


# ===========================================================================
# 4. Expert opt-out applies to a whole logical bank
# ===========================================================================
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_experts_opt_out_keeps_every_layout_in_bf16(tmp_path, layout):
    """GraniteMoE's legacy names do not contain "experts"; the opt-out must still apply."""
    out, report, (gate_up, down) = _convert_layout(tmp_path, layout, skip_patterns=DEFAULT_SKIP + ",experts")
    source = _read_checkpoint_dir(tmp_path / "src")
    converted = _read_checkpoint(out)
    bank = _source_tensors(layout, gate_up, down)
    for key in bank:
        assert torch.equal(converted[key], source[key]) and converted[key].dtype == torch.bfloat16, key
        assert not any(other.startswith(key.removesuffix(".weight") + ".") and other != key for other in converted)
    manifest = json.loads((out / "quantization_config.json").read_text())
    assert "moe_modules" not in manifest
    assert report["stats"]["expert_namespaces"] == 0 and report["stats"]["quantized_expert_matrices"] == 0
    assert report["stats"]["quantized_linears"] == 1  # the dense linear beside the bank


def _read_checkpoint_dir(path):
    from safetensors import safe_open

    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_opted_out_bank_loads_through_vllms_unquantized_method(tmp_path, vllm_env, layout):
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

    out, _, _ = _convert_layout(tmp_path, layout, skip_patterns=DEFAULT_SKIP + ",experts")
    moe = _build_moe(_admit(out), LAYOUTS[layout][1], LAYOUTS[layout][2])
    assert type(moe.routed_experts.quant_method) is UnquantizedFusedMoEMethod


def test_a_pattern_matching_one_expert_key_opts_out_the_whole_bank(tmp_path):
    out, report, (gate_up, down) = _convert_layout(
        tmp_path, "per_expert_qwen3moe", skip_patterns=DEFAULT_SKIP + ",experts.3.down_proj")
    converted = _read_checkpoint(out)
    for key, tensor in _source_tensors("per_expert_qwen3moe", gate_up, down).items():
        assert torch.equal(converted[key], tensor), key
    assert not any(".experts." in key and key.endswith(".q_pack") for key in converted)
    assert report["stats"]["expert_namespaces"] == 0 and report["stats"]["quantized_linears"] == 1


@pytest.mark.parametrize("layout", ["fused_granite", "per_expert_qwen3moe"])
def test_opted_out_bank_is_not_forced_onto_the_hadamard_grid(tmp_path, layout):
    gate_up, down = _source_weights(hidden=H, inter=200)  # K=200 fits no block
    _write_checkpoint(tmp_path / "src", {
        **_source_tensors(layout, gate_up, down),
        "model.layers.0.self_attn.o_proj.weight": torch.randn((H, H), generator=_gen(1)).bfloat16(),
    }, {"num_experts": E})
    with pytest.raises(ValueError, match="multiple of 128"):
        convert(tmp_path / "src", tmp_path / "packed", device="xpu:0")
    skip = DEFAULT_SKIP + ",experts"
    assert dry_run_layer_mapping(tmp_path / "src", skip)["expert_namespaces"] == []
    report = convert(tmp_path / "src", tmp_path / "out", device="xpu:0", skip_patterns=skip)
    assert report["stats"]["quantized_linears"] == 1 and report["stats"]["expert_namespaces"] == 0


def test_calibration_discovers_a_skipped_off_grid_bank(tmp_path):
    """BiasCorr needs the BF16 experts for the forward even though they cannot be packed."""
    pytest.importorskip("transformers")
    import transformers

    from test_converter_streaming import _tiny_tokenizer

    config = transformers.OlmoeConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        head_dim=64, max_position_embeddings=512, tie_word_embeddings=False, intermediate_size=200,
        num_experts=E, num_experts_per_tok=TOPK)
    torch.manual_seed(5)
    model = transformers.OlmoeForCausalLM(config).to(torch.bfloat16).eval()
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__.endswith("Experts") and hasattr(module, "gate_up_proj"):
                module.gate_up_proj.normal_(0.0, 0.05)
                module.down_proj.normal_(0.0, 0.05)
    src = tmp_path / "src"
    model.save_pretrained(src, safe_serialization=True)
    _tiny_tokenizer(src)
    corpus = src / "corpus.txt"
    corpus.write_text(" ".join("alpha beta gamma delta epsilon zeta eta theta".split() * 20))
    report = convert(src, tmp_path / "out", device="xpu:0", skip_patterns=DEFAULT_SKIP + ",experts",
                     biascorr_scope="shared_only", biascorr_corpus=str(corpus), biascorr_tokens=64)
    assert report["stats"]["expert_namespaces"] == 0
    assert report["stats"]["biascorr_linears"] == report["stats"]["quantized_linears"] > 0


# ===========================================================================
# 5. Expert bank coverage, unindexed members, processed == admitted
# ===========================================================================
NS0 = "model.layers.0.mlp.experts"
NS1 = "model.layers.1.mlp.experts"


@pytest.fixture(scope="module")
def two_layer(tmp_path_factory):
    """Two packed expert layers plus one packed dense linear."""
    root = tmp_path_factory.mktemp("two_layer")
    gate_up, down = _source_weights()
    tensors = _source_tensors("per_expert_qwen3moe", gate_up, down)
    tensors.update({k.replace(NS0, NS1): v.clone() for k, v in list(tensors.items())})
    tensors["model.layers.0.shared_mlp.output_linear.weight"] = torch.randn((H, I), generator=_gen(78)).bfloat16()
    _write_checkpoint(root / "src", tensors, {"num_experts": E, "hidden_size": H})
    report = convert(root / "src", root / "out", device="xpu:0")
    assert report["stats"]["expert_namespaces"] == 2
    return (_read_checkpoint(root / "out"), json.loads((root / "out" / "quantization_config.json").read_text()),
            tensors)


def _checkpoint(path, two_layer):
    tensors, manifest = dict(two_layer[0]), json.loads(json.dumps(two_layer[1]))
    return tensors, manifest, lambda: (_rewrite(path, tensors, manifest), _admit(path))[1]


def test_removing_a_whole_expert_layer_fails_when_that_layer_is_built(tmp_path, vllm_env, two_layer):
    tensors, manifest, admit = _checkpoint(tmp_path, two_layer)
    for key in [k for k in tensors if k.startswith(NS1 + ".")]:
        del tensors[key]
    del manifest["moe_modules"]["layers.1.mlp.experts"]
    config = admit()  # the other layer still admits the checkpoint
    _build_moe(config, NS0)
    with pytest.raises(ValueError, match="no packed moe_modules entry and no full-precision weights"):
        _build_moe(config, NS1)


def test_mixed_packed_and_full_precision_layers_are_accepted(tmp_path, vllm_env, two_layer):
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

    from hf_int4_g16_xpu.vllm_moe import HFInt4G16MoEMethod

    tensors, manifest, admit = _checkpoint(tmp_path, two_layer)
    for key in [k for k in tensors if k.startswith(NS1 + ".")]:
        del tensors[key]
    tensors.update({k: v for k, v in two_layer[2].items() if k.startswith(NS1 + ".")})
    del manifest["moe_modules"]["layers.1.mlp.experts"]
    config = admit()
    assert isinstance(_build_moe(config, NS0).routed_experts.quant_method, HFInt4G16MoEMethod)
    assert type(_build_moe(config, NS1).routed_experts.quant_method) is UnquantizedFusedMoEMethod


@pytest.mark.parametrize("damage,match", [
    ("missing_one", r"missing 3\.up_proj\.weight"),
    ("missing_last_expert", rf"missing {E - 1}\.gate_proj\.weight"),
    ("wrong_shape", r"2\.down_proj\.weight is \[256, 128\]"),
    ("integer_dtype", r"0\.gate_proj\.weight is .* I32"),
])
def test_incomplete_full_precision_bank_is_rejected(tmp_path, vllm_env, two_layer, damage, match):
    tensors, manifest, admit = _checkpoint(tmp_path, two_layer)
    for key in [k for k in tensors if k.startswith(NS1 + ".")]:
        del tensors[key]
    tensors.update({k: v for k, v in two_layer[2].items() if k.startswith(NS1 + ".")})
    del manifest["moe_modules"]["layers.1.mlp.experts"]
    if damage == "missing_one":
        del tensors[f"{NS1}.3.up_proj.weight"]
    elif damage == "missing_last_expert":
        for key in [k for k in tensors if k.startswith(f"{NS1}.{E - 1}.")]:
            del tensors[key]
    elif damage == "wrong_shape":
        tensors[f"{NS1}.2.down_proj.weight"] = torch.zeros((H, 128), dtype=torch.bfloat16)
    else:
        tensors[f"{NS1}.0.gate_proj.weight"] = torch.zeros((I, H), dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        _build_moe(admit(), NS1)


def _banks(entries):
    weight_map = dict.fromkeys(entries, "s")
    headers = {"s": {k: {"shape": list(shape), "dtype": dtype} for k, (shape, dtype) in entries.items()}}
    from hf_int4_g16_xpu.vllm_plugin import _norm_key

    return FullPrecisionBanks(weight_map, headers, _norm_key)


def _layout_entries(kind, e=3, h=256, i=192, dtype="BF16"):
    ns = "model.layers.0.mlp.experts"
    if kind in ("fused", "fused_weight"):
        suffix = ".weight" if kind == "fused_weight" else ""
        return {f"{ns}.gate_up_proj{suffix}": ((e, 2 * i, h), dtype), f"{ns}.down_proj{suffix}": ((e, h, i), dtype)}
    if kind == "granite":
        base = "model.layers.0.block_sparse_moe"
        return {f"{base}.input_linear.weight": ((e, 2 * i, h), dtype),
                f"{base}.output_linear.weight": ((e, h, i), dtype)}
    if kind == "mixtral":
        base = "model.layers.0.block_sparse_moe.experts"
        return {f"{base}.{x}.{name}.weight": (((h, i) if name == "w2" else (i, h)), dtype)
                for x in range(e) for name in ("w1", "w2", "w3")}
    return {f"{ns}.{x}.{name}.weight": (((h, i) if name == "down_proj" else (i, h)), dtype)
            for x in range(e) for name in ("gate_proj", "up_proj", "down_proj")}


@pytest.mark.parametrize("kind", ["per_expert", "fused", "fused_weight", "granite", "mixtral"])
@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
def test_every_full_precision_layout_is_verified(kind, dtype):
    pytest.importorskip("vllm")
    prefix = {"granite": "model.layers.0.block_sparse_moe",
              "mixtral": "model.layers.0.block_sparse_moe.experts"}.get(kind, "model.layers.0.mlp.experts")
    entries = _layout_entries(kind, dtype=dtype)
    _banks(entries).require(prefix, 3, 256, 192, True)
    first = sorted(entries)[0]
    for broken in ({k: v for k, v in entries.items() if k != first},
                   {**entries, first: (entries[first][0], "I8")},
                   {**entries, first: ((1, 1), dtype)}):
        with pytest.raises(ValueError, match="incomplete|unrecognized"):
            _banks(broken).require(prefix, 3, 256, 192, True)
    with pytest.raises(ValueError, match="no full-precision weights"):
        _banks(entries).require("model.layers.1.mlp.experts", 3, 256, 192, True)


def test_vision_tower_bank_cannot_stand_in_for_a_text_layer():
    pytest.importorskip("vllm")
    vision = {k.replace("model.layers.0.", "model.visual.layers.0."): v
              for k, v in _layout_entries("per_expert").items()}
    banks = _banks(vision)
    banks.require("model.visual.layers.0.mlp.experts", 3, 256, 192, True)
    with pytest.raises(ValueError, match="no full-precision weights"):
        banks.require("model.layers.0.mlp.experts", 3, 256, 192, True)


@pytest.mark.parametrize("name", [
    f"{NS0}.0.gate_proj.weight",
    f"{NS0}.gate_up_proj",
    f"{NS0}.7.junk",
    "model.layers.0.block_sparse_moe.input_linear.weight",
    "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
])
def test_unindexed_member_of_a_packed_bank_is_rejected(tmp_path, two_layer, name):
    """vLLM reads every tensor in a shard, so one the index never mentions still loads."""
    pytest.importorskip("vllm")
    tensors, manifest, admit = _checkpoint(tmp_path, two_layer)
    if "block_sparse_moe" in name:
        tensors = {k.replace(".mlp.experts.", ".block_sparse_moe.experts."): v for k, v in tensors.items()}
        manifest["moe_modules"] = {k.replace(".mlp.", ".block_sparse_moe."): v
                                   for k, v in manifest["moe_modules"].items()}
    _rewrite(tmp_path, tensors, manifest)
    _admit(tmp_path)  # control
    tensors[name] = torch.zeros((I, H), dtype=torch.bfloat16)
    _rewrite(tmp_path, tensors, manifest)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"][name]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unpacked tensors inside a packed expert namespace"):
        _admit(tmp_path)


def _loaded_layers(tmp_path, two_layer, prefixes=(NS0, NS1), process=(True, True)):
    """Admit the two-layer checkpoint and build, load and process what is asked."""
    from test_hf_int4_g16_moe import _dense_linear, _load_dense, _process

    _, _, admit = _checkpoint(tmp_path, two_layer)
    config = admit()
    dense = _load_dense(_dense_linear(config, tmp_path), tmp_path)
    layers = []
    for index, prefix in enumerate(prefixes):
        moe = _build_moe(config, prefix)
        items = [(k.removeprefix(f"model.layers.{index}.mlp.experts."), v)
                 for k, v in two_layer[0].items() if k.startswith(f"model.layers.{index}.mlp.experts.")]
        list(moe.load_weights(items))
        if process[index]:
            _process(moe)
        layers.append(moe)
    return config, dense, layers


def _moe_inputs():
    return torch.randn((3, H), generator=_gen(90)).bfloat16(), torch.randn((3, E), generator=_gen(91)).bfloat16()


def test_complete_coverage_runs_every_layer(tmp_path, vllm_env, two_layer):
    config, dense, layers = _loaded_layers(tmp_path, two_layer)
    for moe in layers:
        _forward(moe, *_moe_inputs(), vllm_env)
    dense.quant_method.apply(dense, _x(3, I, 92).to(DEV))
    assert config.coverage.verified


def test_dropped_module_attachment_fails_before_first_forward(tmp_path, vllm_env, two_layer):
    """Layer 1 asked for its method but never reached process_weights_after_loading."""
    _, dense, layers = _loaded_layers(tmp_path, two_layer, process=(True, False))
    with pytest.raises(ValueError, match="claimed but never processed.*layers.1.mlp.experts"):
        _forward(layers[0], *_moe_inputs(), vllm_env)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        dense.quant_method.apply(dense, _x(3, I, 93).to(DEV))


def test_a_layer_the_model_never_builds_fails_before_first_forward(tmp_path, vllm_env, two_layer):
    from test_hf_int4_g16_moe import _dense_linear, _load_dense, _process

    _, _, admit = _checkpoint(tmp_path, two_layer)
    config = admit()
    dense = _load_dense(_dense_linear(config, tmp_path), tmp_path)
    moe = _build_moe(config, NS0)
    list(moe.load_weights([(k.removeprefix(NS0 + "."), v) for k, v in two_layer[0].items()
                           if k.startswith(NS0 + ".")]))
    _process(moe)
    with pytest.raises(ValueError, match="no model layer asked for.*layers.1.mlp.experts"):
        dense.quant_method.apply(dense, _x(3, I, 94).to(DEV))


def test_renamed_expert_prefix_is_a_hard_error(tmp_path, vllm_env, two_layer):
    _, _, admit = _checkpoint(tmp_path, two_layer)
    with pytest.raises(ValueError, match="no packed moe_modules entry and no full-precision weights"):
        _build_moe(admit(), "model.layers.1.mlp.moe_experts")


def _unquantized_linear(config, prefix):
    import vllm.model_executor.layers.linear as linear_mod

    with _fake_tp(1), torch.device(DEV):
        layer = linear_mod.RowParallelLinear(I, H, bias=False, input_is_parallel=True, quant_config=config,
                                             prefix=prefix, params_dtype=torch.bfloat16)
    layer.weight.data.normal_(0.0, 0.01)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def test_renamed_dense_prefix_trips_the_unquantized_path(tmp_path, vllm_env, two_layer):
    """Every packed dense layer renamed: only unquantized linears remain, and
    the first one to run still refuses."""
    _, _, admit = _checkpoint(tmp_path, two_layer)
    config = admit()
    renamed = _unquantized_linear(config, "model.layers.0.renamed_mlp.output_linear")
    with pytest.raises(ValueError, match="linear modules no model layer asked for"):
        renamed.quant_method.apply(renamed, _x(2, I, 95).to(DEV))


@pytest.mark.parametrize("complete", [True, False])
def test_coverage_check_survives_fullgraph_compile(tmp_path, vllm_env, two_layer, complete):
    from test_hf_int4_g16_moe import _dense_linear, _load_dense

    config, dense, _ = _loaded_layers(tmp_path, two_layer, process=(True, complete))
    x = _x(5, I, 96).to(DEV)

    def forward(value):
        return dense.quant_method.apply(dense, value) + 0

    compiled = torch.compile(forward, fullgraph=True, backend="eager")
    if complete:
        assert torch.equal(_bits(compiled(x)), _bits(forward(x)))
    else:
        with pytest.raises(Exception, match="coverage is incomplete"):
            compiled(x)


def test_coverage_object_reports_each_gap():
    coverage = PackedCoverage()
    coverage.claimed_linear.update({"a", "b"})
    coverage.processed_linear.add("a")
    with pytest.raises(ValueError) as error:
        coverage.verify({"a", "b", "c"}, {"e"})
    message = str(error.value)
    assert "no model layer asked for (1): ['c']" in message and "claimed but never processed (1): ['b']" in message
    assert "expert modules no model layer asked for (1): ['e']" in message and not coverage.verified
    coverage.processed_linear.update({"b", "c"})
    coverage.claimed_linear.add("c")
    coverage.claimed_moe.add("e")
    coverage.processed_moe.add("e")
    coverage.verify({"a", "b", "c"}, {"e"})
    assert coverage.verified


def test_gemma4_kv_shared_layers_are_the_only_exempt_tensors():
    """vLLM's Gemma-4 mapper drops K/V of KV-shared layers; nothing else is exempt."""
    pytest.importorskip("vllm")
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    names = [f"layers.{i}.self_attn.{p}_proj" for i in range(6) for p in "qkv"]
    config = HFInt4G16Config(quantized_modules={n: {"shape": [64, 256], "bias_corr": False} for n in names})
    config.bind_architecture(_gemma4_architecture(k_eq_v=False, kv_shared=2))
    assert config.model_dropped_modules == {f"layers.{i}.self_attn.{p}_proj" for i in (4, 5) for p in "kv"}
    used = set(names) - config.model_dropped_modules
    config.coverage.claimed_linear.update(used - {"layers.0.self_attn.q_proj"})
    config.coverage.processed_linear.update(used - {"layers.0.self_attn.q_proj"})
    with pytest.raises(ValueError, match=r"\(1\): \['layers.0.self_attn.q_proj'\]"):
        config.require_complete_coverage()


# ===========================================================================
# 6. hf_int4_g16/2 format tightening and name-alias scoping
# ===========================================================================
def _edit(path, valid, mutate):
    tensors, manifest = dict(valid[0]), json.loads(json.dumps(valid[1]))
    _rewrite(path, tensors, manifest)
    _admit(path)  # control: the unmodified checkpoint is admitted
    mutate(manifest)
    _rewrite(path, tensors, manifest)
    return _admit(path)


def _set(*path, value):
    def mutate(manifest):
        node = manifest
        for part in path[:-1]:
            node = node[part]
        if value is _DELETE:
            del node[path[-1]]
        else:
            node[path[-1]] = value
    return mutate


_DELETE = object()
DENSE = "layers.0.shared_mlp.output_linear"
EXPERTS = "layers.0.mlp.experts"


def test_converter_writes_the_frozen_block_rule(two_layer):
    manifest = two_layer[1]
    assert manifest["had_block_rule"] == HAD_BLOCK_RULE_V2 and "hadamard_block" not in manifest
    assert manifest["quantized_modules"][DENSE]["had_block"] == 64
    assert "hadamard_block" not in manifest["quantized_modules"][DENSE]


@pytest.mark.parametrize("mutate,match", [
    pytest.param(_set("had_block_rule", value=_DELETE), "requires had_block_rule", id="rule-missing"),
    pytest.param(_set("had_block_rule", value="128 blocks plus a Had64 tail"), "requires had_block_rule",
                 id="rule-unknown-mixed-schedule"),
    pytest.param(_set("had_block_rule", value=HAD_BLOCK_RULE_V2 + " "), "requires had_block_rule",
                 id="rule-near-miss"),
    pytest.param(_set("hadamard_block", value=128), "top-level hadamard_block", id="top-level-legacy-field"),
    pytest.param(_set("quantized_modules", DENSE, "hadamard_block", value=128), "legacy hadamard_block=128 conflicts",
                 id="module-contradictory-legacy-block"),
    pytest.param(_set("quantized_modules", DENSE, "hadamard_block", value=64), "legacy hadamard_block=64 conflicts",
                 id="module-redundant-legacy-block"),
    pytest.param(_set("quantized_modules", DENSE, "had_block", value=64.0), "had_block must be a JSON integer",
                 id="dense-had-block-float"),
    pytest.param(_set("quantized_modules", DENSE, "had_block", value=True), "had_block must be a JSON integer",
                 id="dense-had-block-bool"),
    pytest.param(_set("quantized_modules", DENSE, "shape", value=[256.0, 192]), "two positive integers",
                 id="dense-shape-float"),
    pytest.param(_set("moe_modules", EXPERTS, "projections", "down_proj", "had_block", value=64.0),
                 "had_block 64.0 does not match", id="moe-had-block-float"),
    pytest.param(_set("moe_modules", EXPERTS, "num_experts", value=float(E)), "must be a positive integer",
                 id="moe-experts-float"),
    pytest.param(_set("moe_modules", EXPERTS, "num_experts", value=True), "must be a positive integer",
                 id="moe-experts-bool"),
    pytest.param(_set("group_size", value=16.0), "group_size must be a JSON integer", id="group-size-float"),
    pytest.param(_set("had_seed", value=101.0), "had_seed must be a JSON integer", id="had-seed-float"),
    pytest.param(_set("had_seed", value="101"), "had_seed must be a JSON integer", id="had-seed-string"),
])
def test_v2_admission_rejects(tmp_path, two_layer, mutate, match):
    pytest.importorskip("vllm")
    with pytest.raises(ValueError, match=match):
        _edit(tmp_path, two_layer, mutate)


@pytest.mark.parametrize("field,value", [("had_block", 128.0), ("hadamard_block", 128.0), ("had_block", True)])
def test_v1_admission_rejects_non_integer_blocks(field, value):
    pytest.importorskip("vllm")
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    assert "JSON integer" in HFInt4G16Config()._module_block_error({field: value}, 256)


@pytest.mark.parametrize("field,value", [("group_size", 16.5), ("hadamard_block", 128.0), ("had_seed", True)])
def test_config_constructor_does_not_coerce_integers(field, value):
    pytest.importorskip("vllm")
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    with pytest.raises(ValueError, match="JSON integer"):
        HFInt4G16Config.from_config({field: value})


def _qkv_config(layer, members, architecture):
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    config = HFInt4G16Config(quantized_modules={
        f"layers.{layer}.self_attn.{m}_proj": {"shape": [64, 256], "bias_corr": False} for m in members})
    config.bind_architecture(architecture)
    return config


@pytest.mark.parametrize("layer,architecture", [
    pytest.param(4, _gemma4_architecture(), id="sliding-layer-of-a-k_eq_v-model"),
    pytest.param(5, _gemma4_architecture(k_eq_v=False), id="k_eq_v-not-declared"),
    pytest.param(5, _gemma4_architecture(kv_shared=1), id="kv-shared-layer"),
    pytest.param(5, None, id="no-model-config"),
    pytest.param(5, SimpleNamespace(model_type="qwen3_moe"), id="other-architecture"),
])
def test_k_proj_doubling_only_for_declared_layers(layer, architecture):
    pytest.importorskip("vllm")
    config = _qkv_config(layer, "qk", architecture)
    with pytest.raises(ValueError, match="partial fused quantization"):
        config._members_for(f"model.layers.{layer}.self_attn.qkv_proj")


def test_declared_k_eq_v_layer_must_hold_exactly_q_and_k():
    pytest.importorskip("vllm")
    prefix = "model.layers.5.self_attn.qkv_proj"
    assert _qkv_config(5, "qk", _gemma4_architecture())._members_for(prefix) == [
        f"layers.5.self_attn.{m}_proj" for m in "qkk"]
    for members in ("qkv", "q", "qv"):
        with pytest.raises(ValueError, match="declared attention_k_eq_v layer"):
            _qkv_config(5, members, _gemma4_architecture())._members_for(prefix)
    assert len(_qkv_config(4, "qkv", _gemma4_architecture())._members_for(prefix.replace(".5.", ".4."))) == 3


@pytest.mark.parametrize("nested", [False, True])
def test_architecture_is_read_from_real_transformers_configs(nested):
    pytest.importorskip("vllm")
    import transformers

    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    text = transformers.Gemma4TextConfig(num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"],
                                         attention_k_eq_v=True)
    config = HFInt4G16Config()
    config.bind_architecture(SimpleNamespace(text_config=text) if nested else text)
    assert config.k_eq_v_layers == {5} and not config.mamba_registered_as_mixer
    granite = HFInt4G16Config()
    granite.bind_architecture(transformers.GraniteMoeHybridConfig())
    assert granite.mamba_registered_as_mixer and not granite.k_eq_v_layers


def _mixer_config(names, model_type):
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    config = HFInt4G16Config(quantized_modules={
        f"layers.0.{n}.in_proj": {"shape": [1064, 256], "bias_corr": False} for n in names})
    config.bind_architecture(SimpleNamespace(model_type=model_type))
    return config


def test_mamba_alias_is_scoped_to_granite_and_ambiguity_is_rejected():
    pytest.importorskip("vllm")
    prefix = "model.layers.0.mixer.in_proj"
    for model_type in ("nemotron_h", "bamba", "mamba2"):
        assert _mixer_config(["mamba"], model_type)._members_for(prefix) is None
        assert _mixer_config(["mixer", "mamba"], model_type)._members_for(prefix) == ["layers.0.mixer.in_proj"]
    with pytest.raises(ValueError, match="ambiguous checkpoint names"):
        _mixer_config(["mixer", "mamba"], "granitemoehybrid")._members_for(prefix)


# ===========================================================================
# 7. Grouped-by-expert prefill: bit parity with the pair path
# ===========================================================================
def _routes(kind: str, pairs: int, slots: int, experts: int, seed: int) -> torch.Tensor:
    g = _gen(seed)
    ids = torch.randint(0, experts, (pairs,), generator=g, dtype=torch.int32)
    if kind == "hot":                      # every pair on one expert
        ids[:] = experts // 2
    elif kind == "skew":                   # most pairs on two experts, a long tail
        mask = torch.rand(pairs, generator=g) < 0.85
        ids[mask] = (torch.rand(int(mask.sum()), generator=g) < 0.7).to(torch.int32) * (experts - 1)
    elif kind == "empty_experts":          # only two experts of the table are used
        ids = (torch.randint(0, 2, (pairs,), generator=g, dtype=torch.int32) * (experts - 1))
    elif kind == "repeated_in_token":      # every slot of a token picks the same expert
        ids = ids[::slots].repeat_interleave(slots)[:pairs]
    elif kind == "invalid_mix":
        bad = torch.rand(pairs, generator=g) < 0.4
        ids[bad] = torch.tensor([-1, experts, experts + 7, -(2**31)], dtype=torch.int32)[
            torch.randint(0, 4, (int(bad.sum()),), generator=g)]
    elif kind == "all_invalid":
        ids[:] = -1
    elif kind == "one_each":               # as many one-row tiles as the table allows
        ids = (torch.arange(pairs) % experts).to(torch.int32)
    return ids


ROUTE_KINDS = ["uniform", "hot", "skew", "empty_experts", "repeated_in_token", "invalid_mix", "all_invalid",
               "one_each"]
GROUP_SHAPES = [  # (experts, N, K): Had64 tails (K%128==64), Had128, N%32==16 tails
    (16, 48, 192), (7, 272, 704), (64, 384, 256), (1, 80, 128), (12, 16, 2112), (32, 256, 576), (128, 1408, 2816),
]


def _with_bias(stack: PackedExperts, mode: str, seed: int) -> PackedExperts:
    if mode == "null":
        bias = None
    elif mode == "neg_zero":
        bias = torch.full((stack.num_experts, stack.n), -0.0, device=DEV)
    else:
        bias = (torch.randn((stack.num_experts, stack.n), generator=_gen(seed)) * 0.01).to(DEV)
    return PackedExperts(stack.q_pack, stack.u_pack, stack.d_row, stack.signs, bias)


def _compare_gemm(x, ids, stack, slots, label):
    buckets = native_l0.moe_bucket_native(ids, stack.num_experts)
    for dtype in (torch.float32, torch.bfloat16):
        pair = moe_pair_gemm(x, ids, stack, slots, dtype, "native")
        grouped = moe_grouped_gemm(x, buckets, stack, slots, dtype)
        torch.xpu.synchronize()
        differing = int((_bits(pair) != _bits(grouped)).sum())
        assert differing == 0, f"{label} {dtype}: {differing}/{pair.numel()} words differ"


@pytest.mark.parametrize("experts,n,k", GROUP_SHAPES)
@pytest.mark.parametrize("kind", ROUTE_KINDS)
def test_grouped_gemm_is_bit_identical_to_pair(experts, n, k, kind):
    slots, tokens = 8, 40
    raw, _ = _random_stack(experts, n, k, seed=n + k, bias=False)
    stack = _to_device(raw, k)
    x = torch.randn((tokens, k), generator=_gen(k + experts)).bfloat16()
    x[3] = 0  # an exact-zero row keeps its signed zeros
    ids = _routes(kind, tokens * slots, slots, experts, seed=n).to(DEV)
    for bias in ("null", "neg_zero", "random"):
        _compare_gemm(x.to(DEV), ids, _with_bias(stack, bias, 5), slots, f"{kind} bias={bias}")


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32, 33, 48])
def test_grouped_gemm_every_tile_length(count):
    """One expert with ``count`` pairs walks the 1/2/4/8/16-row bodies and their padding."""
    raw, _ = _random_stack(3, 80, 192, seed=count, bias=True)
    stack = _to_device(raw, 192)
    x = torch.randn((count, 192), generator=_gen(count)).bfloat16().to(DEV)
    _compare_gemm(x, torch.ones(count, dtype=torch.int32, device=DEV), stack, 1, f"len={count}")


def test_bucket_descriptors_partition_the_pairs():
    experts, pairs = 37, 1000
    ids = _routes("invalid_mix", pairs, 1, experts, seed=3)
    buckets = native_l0.moe_bucket_native(ids.to(DEV), experts)
    perm, tiles, ntiles = (t.cpu() for t in native_l0.moe_bucket_snapshot(buckets))
    live = tiles[:int(ntiles)]
    assert tiles.shape[0] == min(pairs, pairs // 16 + experts + 1) and int(ntiles) <= tiles.shape[0]
    assert sorted(perm.tolist()) == list(range(pairs))
    cursor = 0
    for bucket, start, length in live.tolist():
        assert start == cursor and 1 <= length <= 16
        member = ids[perm[start:start + length]]
        assert torch.all(member == bucket) if bucket < experts else torch.all((member < 0) | (member >= experts))
        cursor += length
    assert cursor == pairs
    assert torch.equal(live[:, 0], live[:, 0].sort().values)


@pytest.mark.parametrize("m", [1, 8, 15, 16, 17, 64, 512])
@pytest.mark.parametrize("hidden,inter,experts,slots", [(256, 192, 16, 8), (192, 256, 64, 4), (2816, 704, 32, 8)])
def test_grouped_forward_is_bit_identical_to_pair(m, hidden, inter, experts, slots):
    inputs, stacks, reference = _moe_case(m, experts, slots, hidden, inter, True, seed=2000 + m + hidden)
    pair = _run(inputs, stacks, "gelu_tanh", backend="native", grouped_min_tokens=1 << 30)
    forced = _run(inputs, stacks, "gelu_tanh", backend="native", grouped_min_tokens=1)
    default = _run(inputs, stacks, "gelu_tanh", backend="native")
    assert torch.equal(_bits(pair), _bits(forced))
    assert torch.equal(_bits(pair), _bits(default))
    if hidden < 1000 and m <= 64:
        assert _rel(forced.float(), _oracle_moe(*inputs, *reference, "gelu_tanh").double()) < 4e-3


@pytest.mark.parametrize("kind", ROUTE_KINDS)
def test_grouped_forward_route_distributions_and_chunk_tails(kind):
    """M=1030 crosses the 1024-token chunk: the 6-token tail takes the pair path."""
    experts, slots, m = 64, 8, 1030
    inputs, stacks, _ = _moe_case(m, experts, slots, 256, 192, True, seed=2100)
    ids = _routes(kind, m * slots, slots, experts, seed=7).reshape(m, slots).to(torch.int64)
    case = (inputs[0], inputs[1], ids)
    assert torch.equal(_bits(_run(case, stacks, "silu", backend="native")),
                       _bits(_run(case, stacks, "silu", backend="native", grouped_min_tokens=1 << 30)))


def test_grouped_path_is_repeatable():
    """Integer atomics may reorder rows inside a bucket from run to run; outputs must not move."""
    inputs, stacks, _ = _moe_case(128, 8, 8, 256, 192, True, seed=2200)
    first = _bits(_run(inputs, stacks, "silu", backend="native"))
    for _ in range(20):
        assert torch.equal(first, _bits(_run(inputs, stacks, "silu", backend="native")))


@pytest.mark.parametrize("m,backend,grouped_calls", [
    (1, "auto", 0), (8, "auto", 0), (15, "auto", 0), (16, "auto", 1), (64, "native", 1), (64, "triton", 0),
    (2050, "auto", 2),
])
def test_dispatch_depends_on_token_count_only(monkeypatch, m, backend, grouped_calls):
    calls = []
    real = native_l0.moe_bucket_native
    monkeypatch.setattr(moe_module.native_l0, "moe_bucket_native",
                        lambda ids, experts: calls.append(ids.shape[0]) or real(ids, experts))
    inputs, stacks, _ = _moe_case(m, 8, 2, 256, 192, False, seed=2300)
    _run(inputs, stacks, "silu", backend=backend)
    assert GROUPED_MIN_TOKENS == 16 and len(calls) == grouped_calls


def test_fp16_prefill_stays_on_triton(monkeypatch):
    monkeypatch.setattr(moe_module.native_l0, "moe_bucket_native",
                        lambda *a: pytest.fail("grouped path needs native bf16"))
    inputs, stacks, reference = _moe_case(32, 8, 2, 256, 192, True, seed=2400)
    got = _run((inputs[0].to(torch.float16), *inputs[1:]), stacks, "silu")
    assert _rel(got.float(), _oracle_moe(*inputs, *reference, "silu").double()) < 1e-2


@pytest.mark.parametrize("m", [16, 17, 64, 512])
def test_grouped_graph_replay_follows_changing_routes(m):
    """Capture once, then change activations, weights and routing in place,
    including a single hot expert and routes that are entirely out of range."""
    experts, slots = 24, 8
    inputs, stacks, _ = _moe_case(m, experts, slots, 256, 192, True, seed=2500 + m)
    static = [t.to(DEV) for t in inputs]

    def run(**kwargs):
        return hf_int4_g16_moe(static[0], static[1], static[2], *stacks, activation="silu", backend="native", **kwargs)

    for _ in range(2):
        run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = run()
    seen = []
    for step, kind in enumerate(["uniform", "hot", "skew", "all_invalid", "one_each", "invalid_mix", "uniform"]):
        static[0].copy_(torch.randn((m, 256), generator=_gen(2600 + step)).bfloat16().to(DEV))
        static[1].copy_(_routing(m, experts, slots, 2700 + step)[0].to(DEV))
        static[2].copy_(_routes(kind, m * slots, slots, experts, 2800 + step).reshape(m, slots).to(DEV))
        graph.replay()
        torch.xpu.synchronize()
        replayed = _bits(captured).clone()
        assert torch.equal(replayed, _bits(run())), f"stale grouped replay: {kind}"
        assert torch.equal(replayed, _bits(run(grouped_min_tokens=1 << 30))), f"replay != pair path: {kind}"
        seen.append(replayed)
    assert any(not torch.equal(a, b) for a, b in zip(seen, seen[1:]))


def test_grouped_graph_scratch_is_private_to_the_capture():
    """Bucket and rotation scratch is allocated during capture.  Tensors made
    afterwards must not land on it, or a replay would overwrite them."""
    m, experts, slots = 64, 16, 8
    inputs, stacks, _ = _moe_case(m, experts, slots, 256, 192, True, seed=2900)
    static = [t.to(DEV) for t in inputs]
    call = lambda: hf_int4_g16_moe(*static, *stacks, activation="silu", backend="native")
    call()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = call()
    graph.replay()
    torch.xpu.synchronize()
    reference = _bits(captured).clone()
    pairs = m * slots
    sentinels = [torch.full((size,), 0x5A5A5A5A, dtype=torch.int32, device=DEV)
                 for size in (1, experts + 1, 3 * (pairs // 16 + experts + 1), pairs, m * 256, pairs * 192,
                              pairs * 256, pairs * 384)
                 for _ in range(4)]
    for _ in range(3):
        graph.replay()
    torch.xpu.synchronize()
    assert torch.equal(_bits(captured), reference)
    assert all(bool((s == 0x5A5A5A5A).all()) for s in sentinels)


def test_grouped_forward_under_vllm_moe_runner(tmp_path, vllm_env):
    """A prefill-sized batch through vLLM's router and moe_forward op."""
    from test_hf_int4_g16_moe import _dense_linear, _load_dense, _load_into, _process

    layout = "per_expert_qwen3moe"
    out, _, _ = _convert_layout(tmp_path, layout)
    config = _admit(out)
    _load_dense(_dense_linear(config, out), out)
    moe = _build_moe(config, LAYOUTS[layout][1], LAYOUTS[layout][2])
    _load_into(moe, out, layout)
    _process(moe)
    x = torch.randn((48, H), generator=_gen(97)).bfloat16()
    logits = torch.randn((48, E), generator=_gen(98)).bfloat16()
    grouped = _forward(moe, x, logits, vllm_env)
    pieces = torch.cat([_forward(moe, x[i:i + 8], logits[i:i + 8], vllm_env) for i in range(0, 48, 8)])
    assert torch.equal(_bits(grouped), _bits(pieces))
