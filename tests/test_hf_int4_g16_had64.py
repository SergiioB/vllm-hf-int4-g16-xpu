"""Had64 (block-64) geometry: format, packing, Triton/native parity, fail-closed."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from hf_int4_g16_xpu import kernels_triton as tk
from hf_int4_g16_xpu import native_l0
from hf_int4_g16_xpu.format import (
    HAD_BLOCK_RULE_V2,
    had_block_for_k,
    make_hadamard,
    pack_native_words,
    quantize_pack_hf_int4,
    rotate_had128_noperm,
    rotate_had_noperm,
    signs_for_k,
    unpack_hf_int4_reference,
)
from hf_int4_g16_xpu.ops import hf_int4_g16_linear

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
DEV = "xpu"
HAD64_KS = [704, 1856, 2112]  # Gemma-4 expert down, NemotronH expert down, Gemma-4 shared down


def _packed(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    u = torch.randint(0, 16, (n, k // 16), generator=g, dtype=torch.uint8)
    qp, up = pack_native_words(q, u)
    d = torch.rand(n, generator=g) * 0.0002 + 0.00005
    return q, u, qp, up, d


def _oracle_rotate(x64: torch.Tensor, k: int) -> torch.Tensor:
    """Independent block-by-block FP64 rotation (does not call rotate_had_noperm)."""
    block = 128 if k % 128 == 0 else 64
    h = make_hadamard(block, "cpu", torch.float64)
    xs = x64 * signs_for_k(k).to(torch.float64)
    return torch.cat([xs[:, i:i + block] @ h for i in range(0, k, block)], dim=1)


def _oracle(x, qp, up, d, bias=None):
    k = x.shape[1]
    w = unpack_hf_int4_reference(qp, up).double() * d.double()[:, None]
    y = _oracle_rotate(x.double(), k) @ w.T
    return y if bias is None else y + bias.double()


def _x(m, k, seed):
    return torch.randn((m, k), generator=torch.Generator().manual_seed(seed)).bfloat16()


@pytest.mark.parametrize("block", [64, 128])
def test_hadamard_is_exactly_orthonormal_and_symmetric(block):
    h = make_hadamard(block, "cpu", torch.float64)
    # 1/sqrt(64) = 1/8 is exact, so Had64 is orthonormal to the last bit.
    tolerance = 0.0 if block == 64 else 1e-14
    assert (h @ h.T - torch.eye(block, dtype=torch.float64)).abs().max().item() <= tolerance
    assert torch.equal(h, h.T)
    assert torch.equal(h.abs(), torch.full_like(h, h[0, 0].item()))


def test_block_rule():
    assert [had_block_for_k(k) for k in (128, 256, 2816, 17408)] == [128] * 4
    assert [had_block_for_k(k) for k in (64, 192, 704, 1856, 2112)] == [64] * 5


@pytest.mark.parametrize("k", [0, 32, 96, 100, 129, 656, 17440])
def test_block_rule_fails_closed_off_the_64_grid(k):
    with pytest.raises(ValueError, match="multiple of 128 .Had128. or of 64"):
        had_block_for_k(k)


@pytest.mark.parametrize("k", HAD64_KS)
def test_rotation_is_blockwise_64_and_norm_preserving(k):
    x = torch.randn((5, k), generator=torch.Generator().manual_seed(k), dtype=torch.float32)
    rotated = rotate_had_noperm(x, signs_for_k(k))
    assert torch.allclose(rotated.double(), _oracle_rotate(x.double(), k), atol=2e-6, rtol=0)
    assert torch.allclose(rotated.norm(dim=1), x.norm(dim=1), rtol=1e-6)
    # Every 64-block is closed under the rotation: nothing leaks across blocks.
    only_first = torch.zeros_like(x)
    only_first[:, :64] = x[:, :64]
    assert rotate_had_noperm(only_first, signs_for_k(k))[:, 64:].abs().max().item() == 0.0


def test_had128_entry_point_refuses_had64_width():
    with pytest.raises(ValueError, match="Had128 rotation"):
        rotate_had128_noperm(torch.zeros(1, 704), signs_for_k(704))
    with pytest.raises(ValueError, match="fwht128_actq"):
        tk.fwht128_actq(torch.zeros((1, 704), device=DEV).bfloat16(), signs_for_k(704, device=DEV))


@pytest.mark.parametrize("k", HAD64_KS)
@pytest.mark.parametrize("n", [264, 300])
def test_pack_unpack_bit_exact_cpu_and_triton(k, n):
    q, u, qp, up, _ = _packed(n, k, seed=k + n)
    expected = ((q.to(torch.int16) - 8) * (u.repeat_interleave(16, dim=1).to(torch.int16) + 1)).to(torch.int8)
    assert torch.equal(unpack_hf_int4_reference(qp, up), expected)
    assert torch.equal(tk.unpack_to_int8(qp.to(DEV), up.to(DEV), n, k).cpu(), expected)


@pytest.mark.parametrize("k", HAD64_KS)
def test_quantizer_approximates_the_unrotated_weight(k):
    weight = torch.randn((96, k), generator=torch.Generator().manual_seed(k + 1)) * 0.02
    packed = quantize_pack_hf_int4(weight.bfloat16(), device=DEV)
    x = _x(64, k, k + 2)
    want = x.double() @ weight.bfloat16().double().T
    got = _oracle(x, packed["q_pack"], packed["u_pack"], packed["d_row"])
    # INT4 x UINT4/G16 error on Gaussian weights; a wrong rotation gives ~1.4.
    assert ((got - want).norm() / want.norm()).item() < 0.12


@pytest.mark.parametrize("k", HAD64_KS)
# Native W4A16 is the fused decode kernel, which exists for M 1..4 only.
@pytest.mark.parametrize("backend,m", [("native", m) for m in (1, 2, 3, 4)]
                         + [("triton", m) for m in (1, 2, 3, 4, 8, 65)])
def test_w4a16_matches_fp64_oracle(k, m, backend):
    n = 272  # native needs N%16==0
    _, _, qp, up, d = _packed(n, k, seed=3 * k + m)
    bc = torch.randn(n, generator=torch.Generator().manual_seed(k + m)) * 0.01
    x = _x(m, k, 5 * k + m)
    got = hf_int4_g16_linear(
        x.to(DEV), qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV),
        bias_corr=bc.to(DEV), backend=backend, mode="w4a16",
    ).cpu()
    want = _oracle(x, qp, up, d, bc)
    assert ((got.double() - want).norm() / want.norm()).item() < 4e-3  # one BF16 rounding


@pytest.mark.parametrize("k", HAD64_KS)
@pytest.mark.parametrize("m", [1, 7, 33])
def test_native_actq_is_bit_exact_with_triton(k, m):
    x = _x(m, k, 7 * k + m)
    x[0] = 0  # zero row
    signs = signs_for_k(k, device=DEV)
    native_i8, native_sx = native_l0.fwht_actq(x.to(DEV), signs)
    triton_i8, triton_sx = tk.fwht_actq(x.to(DEV), signs)
    torch.xpu.synchronize()
    assert torch.equal(native_i8.cpu(), triton_i8.cpu())
    assert torch.equal(native_sx.cpu().view(torch.int32), triton_sx.cpu().view(torch.int32))
    rotated = _oracle_rotate(x.double(), k)
    reference_scale = rotated.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    assert torch.allclose(native_sx.cpu().double(), reference_scale, rtol=1e-6, atol=0)


@pytest.mark.parametrize("k", HAD64_KS)
@pytest.mark.parametrize("m", [5, 16, 32, 64, 129, 512])
def test_native_dpas_int32_exact_and_w4a8_matches_triton(k, m):
    n = 272
    _, _, qp, up, d = _packed(n, k, seed=11 * k + m)
    g = torch.Generator().manual_seed(13 * k + m)
    xi8 = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8)
    sx = torch.rand(m, generator=g) * 0.001 + 0.0001
    got_i32 = native_l0.dpas_native_fixed(
        xi8.to(DEV), sx.to(DEV), qp.to(DEV), up.to(DEV), d.to(DEV), write_i32=True
    ).cpu()
    want_i32 = xi8.double() @ unpack_hf_int4_reference(qp, up).double().T
    assert torch.equal(got_i32.double(), want_i32)

    x = _x(m, k, 17 * k + m)
    args = (qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV))
    native = hf_int4_g16_linear(x.to(DEV), *args, backend="native", mode="w4a8").cpu()
    triton = hf_int4_g16_linear(x.to(DEV), *args, backend="triton", mode="w4a8").cpu()
    assert torch.allclose(native.float(), triton.float(), rtol=8e-3, atol=1e-5)


@pytest.mark.parametrize("k", [656, 100])
@pytest.mark.parametrize("backend", ["auto", "triton", "native"])
def test_non_multiple_of_64_rejected_before_any_block_read(k, backend):
    n = 64
    x = torch.zeros((2, k), device=DEV, dtype=torch.bfloat16)
    qp = torch.zeros((k // 8, n), device=DEV, dtype=torch.int32)
    up = torch.zeros((max(1, k // 64), n), device=DEV, dtype=torch.int16)
    d = torch.ones(n, device=DEV)
    signs = torch.ones(k, device=DEV, dtype=torch.int8)
    with pytest.raises(ValueError, match="multiple of 128"):
        hf_int4_g16_linear(x, qp, up, d, signs, backend=backend)
    with pytest.raises(ValueError, match="multiple of 128"):
        native_l0.validate_native_operands(x, signs)
    with pytest.raises(ValueError, match="multiple of 128"):
        tk.fwht_actq(x, signs)
    with pytest.raises(ValueError, match="multiple of 128"):
        quantize_pack_hf_int4(torch.zeros(4, k), device="cpu")


@pytest.mark.parametrize("m", [1, 4, 8])
def test_xpu_graph_replay_had64_native(m):
    n, k = 272, 704
    _, _, qp, up, d = _packed(n, k, seed=90 + m)
    args = (qp.to(DEV), up.to(DEV), d.to(DEV), signs_for_k(k, device=DEV))
    mode = "w4a16" if m <= 4 else "w4a8"
    static_x = _x(m, k, 91).to(DEV)
    for _ in range(2):
        hf_int4_g16_linear(static_x, *args, backend="native", mode=mode)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        static_y = hf_int4_g16_linear(static_x, *args, backend="native", mode=mode)
    for seed in (92, 93):
        static_x.copy_(_x(m, k, seed).to(DEV))
        graph.replay()
        torch.xpu.synchronize()
        eager = hf_int4_g16_linear(static_x, *args, backend="native", mode=mode)
        torch.xpu.synchronize()
        assert torch.equal(static_y.view(torch.int16), eager.view(torch.int16))


# ---------------------------------------------------------------------------
# vLLM dense layers with Had64 inputs
# ---------------------------------------------------------------------------
@contextmanager
def _fake_tp(size: int):
    """vLLM linear layers outside an engine: fix the tensor-parallel geometry."""
    import vllm.model_executor.layers.linear as linear_mod

    with patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: size), \
         patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: 0):
        yield


def test_dense_row_parallel_had64_layer_loads_and_matches_oracle():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.linear import RowParallelLinear

    from hf_int4_g16_xpu.format import FORMAT_VERSION_V2
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    n, k = 2816, 2112  # Gemma-4 shared mlp.down_proj
    _, _, qp, up, d = _packed(n, k, seed=2112)
    config = HFInt4G16Config(
        format_version=FORMAT_VERSION_V2,
        quantized_modules={"layers.0.mlp.down_proj": {"shape": [n, k], "bias_corr": False, "had_block": 64}},
    )
    config.validate_format()
    with _fake_tp(1):
        layer = RowParallelLinear(k, n, bias=False, input_is_parallel=True, quant_config=config,
                                  prefix="model.layers.0.mlp.down_proj")
    for role, tensor in (("q_pack", qp), ("u_pack", up), ("d_row", d)):
        parameter = getattr(layer, role)
        parameter.weight_loader(parameter, tensor, 0)
    layer = layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    x = _x(3, k, 4224)
    got = layer.quant_method.apply(layer, x.to(DEV)).cpu()
    want = _oracle(x, qp, up, d)
    assert ((got.double() - want).norm() / want.norm()).item() < 4e-3


def test_row_shard_must_align_to_the_global_block():
    """A 64-wide shard of a Had128 projection must not be silently rotated as Had64."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.linear import RowParallelLinear

    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    n, k = 256, 256
    config = HFInt4G16Config(quantized_modules={"layers.0.mlp.down_proj": {"shape": [n, k], "bias_corr": False}})
    with _fake_tp(4):
        with pytest.raises(ValueError, match="must align to the Had128 blocks"):
            RowParallelLinear(k, n, bias=False, input_is_parallel=True, quant_config=config,
                              prefix="model.layers.0.mlp.down_proj")


@pytest.mark.parametrize("version,had_block,match", [
    ("hf_int4_g16/2", 128, "had_block 128 does not match K=704"),
    ("hf_int4_g16/2", None, "had_block None does not match K=704"),
    ("hf_int4_g16/1", 64, "hf_int4_g16/1 is Had128 only"),
])
def test_dense_admission_rejects_block_disagreement(tmp_path, version, had_block, match):
    pytest.importorskip("vllm")
    import json

    from safetensors.torch import save_file

    from hf_int4_g16_xpu.format import sign_sha256_for_k
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    n, k, base = 264, 704, "layers.0.mlp.down_proj"
    entry = {"shape": [n, k], "bias_corr": False}
    if had_block is not None:
        entry["had_block"] = had_block
    manifest = {
        "group_size": 16, "had_seed": 101, "layout": "xe2_dpas_native_v1", "format_version": version,
        "rotation": "HadAutoNoPerm" if version.endswith("2") else "Had128NoPerm",
        "sign_sha256_by_k": {str(k): sign_sha256_for_k(k)}, "quantized_modules": {base: entry},
    }
    if version.endswith("2"):
        manifest["had_block_rule"] = HAD_BLOCK_RULE_V2
    tensors = {
        base + ".q_pack": torch.zeros((k // 8, n), dtype=torch.int32),
        base + ".u_pack": torch.zeros((k // 64, n), dtype=torch.int16),
        base + ".d_row": torch.ones(n),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")}))
    (tmp_path / "quantization_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        HFInt4G16Config.from_config(manifest).maybe_update_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Gemma-4 attention_k_eq_v: full-attention layers ship q_proj and k_proj only,
# and vLLM's weight mapper delivers k_proj once with shard_id ["k", "v"].
# ---------------------------------------------------------------------------
def _gemma4_architecture(k_eq_v=True, kv_shared=0):
    """The text-config fields vLLM's Gemma-4 weight mapper reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model_type="gemma4_text", num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"],
        attention_k_eq_v=k_eq_v, num_kv_shared_layers=kv_shared)


def _k_eq_v_layer(k_eq_v=True):
    from vllm.model_executor.layers.linear import QKVParallelLinear

    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    k, head, heads, kv_heads = 256, 64, 4, 2
    widths = {"q_proj": head * heads, "k_proj": head * kv_heads}
    config = HFInt4G16Config(quantized_modules={
        f"layers.5.self_attn.{name}": {"shape": [n, k], "bias_corr": True} for name, n in widths.items()
    })
    config.bind_architecture(_gemma4_architecture(k_eq_v))
    with _fake_tp(1):
        layer = QKVParallelLinear(k, head, heads, kv_heads, bias=False, quant_config=config,
                                  prefix="model.layers.5.self_attn.qkv_proj")
    payloads = {}
    for index, (name, n) in enumerate(widths.items()):
        _, _, qp, up, d = _packed(n, k, seed=500 + index)
        correction = torch.arange(n, dtype=torch.float32) * 2.0 ** -9 + index
        payloads[name] = {"q_pack": qp, "u_pack": up, "d_row": d, "bias_corr": correction}
    return layer, payloads, k


def _load_through_gemma4_mapper(layer, payloads, k_eq_v=True, members=("q_proj", "k_proj")):
    """Feed checkpoint names through vLLM's own Gemma-4 WeightsMapper, then call
    the loaders the way AutoWeightsLoader does (shard id as a tensor attribute)."""
    from types import SimpleNamespace

    from vllm.model_executor.models.gemma4 import _gemma4_layer_weights_mapper

    mapper = _gemma4_layer_weights_mapper(SimpleNamespace(
        num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"],
        attention_k_eq_v=k_eq_v, num_kv_shared_layers=0))
    weights = [(f"layers.5.self_attn.{name}.{role}", tensor.clone())
               for name in members for role, tensor in payloads[name].items()]
    seen = []
    for name, tensor in mapper.apply(weights):
        module, role = name.rsplit(".", 1)
        assert module == "layers.5.self_attn.qkv_proj"
        parameter = getattr(layer, role)
        seen.append((role, tensor.shard_id))
        parameter.weight_loader(parameter, tensor)
    return seen


def test_k_eq_v_loads_k_proj_into_both_k_and_v():
    pytest.importorskip("vllm")
    layer, payloads, k = _k_eq_v_layer()
    seen = _load_through_gemma4_mapper(layer, payloads)
    assert [shard for role, shard in seen if role == "q_pack"] == ["q", "k", "v"]
    layer = layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    x = _x(3, k, 501)
    got = layer.quant_method.apply(layer, x.to(DEV)).cpu().double()
    expected = [
        _oracle(x, p["q_pack"], p["u_pack"], p["d_row"], p["bias_corr"])
        for p in (payloads["q_proj"], payloads["k_proj"], payloads["k_proj"])
    ]
    want = torch.cat(expected, dim=1)
    assert got.shape == want.shape
    assert (got - want).abs().max().item() < 1e-2
    k_columns, v_columns = got[:, 256:384], got[:, 384:]
    assert torch.equal(k_columns, v_columns)


def test_lost_v_proj_without_k_eq_v_fails_closed():
    """The same two-member checkpoint under a model that does expect v_proj is
    rejected when the layer is built, before any weight is loaded."""
    pytest.importorskip("vllm")
    with pytest.raises(ValueError, match="partial fused quantization"):
        _k_eq_v_layer(k_eq_v=False)


def test_k_eq_v_incomplete_load_rejected():
    pytest.importorskip("vllm")
    layer, payloads, _ = _k_eq_v_layer()
    _load_through_gemma4_mapper(layer, payloads, members=("q_proj",))
    layer = layer.to(DEV)
    with pytest.raises(ValueError, match="incomplete fused suffix loading"):
        layer.quant_method.process_weights_after_loading(layer)
