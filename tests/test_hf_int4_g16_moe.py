"""Packed routed experts: GEMM and forward parity, graph replay, converter -> vLLM.

The numerical oracle is a deliberately naive per-(token, slot) FP64 loop.  It
shares only the format's bit decoder with the code under test.  The end-to-end
tests run converter output through vLLM's own ``RoutedExperts.load_weights``,
router and ``moe_forward`` op.
"""
from __future__ import annotations

import json
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from hf_int4_g16_xpu.converter import convert, dry_run_layer_mapping
from hf_int4_g16_xpu.format import (
    EXPERT_PROJECTIONS,
    EXPERT_TENSOR_SUFFIXES,
    make_hadamard,
    pack_native_words,
    signs_for_k,
    unpack_hf_int4_reference,
)
from hf_int4_g16_xpu.moe import PackedExperts, hf_int4_g16_moe, moe_pair_gemm

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
DEV = "xpu"
ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda g: F.gelu(g, approximate="tanh"),
}


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _bits(t):
    return t.cpu().contiguous().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _rel(got, want):
    return ((got.double() - want).norm() / want.norm()).item()


def _rotate64(x: torch.Tensor) -> torch.Tensor:
    """FP64 blockwise rotation along the last dim, written independently of the package."""
    k = x.shape[-1]
    block = 128 if k % 128 == 0 else 64
    h = make_hadamard(block, "cpu", torch.float64)
    xs = x.double() * signs_for_k(k).double()
    return (xs.reshape(-1, block) @ h).reshape(xs.shape)


def _random_stack(experts: int, n: int, k: int, seed: int, bias: bool):
    """Random packed 3D experts plus their FP64 rotated-basis weights ``[E,N,K]``."""
    g = _gen(seed)
    q = torch.randint(0, 16, (experts * n, k), generator=g, dtype=torch.uint8)
    u = torch.randint(0, 16, (experts * n, k // 16), generator=g, dtype=torch.uint8)
    packed = [pack_native_words(q[e * n:(e + 1) * n], u[e * n:(e + 1) * n]) for e in range(experts)]
    q3 = torch.stack([p[0] for p in packed])
    u3 = torch.stack([p[1] for p in packed])
    d = (torch.rand((experts, n), generator=g) * 0.00006 + 0.00002) * (256.0 / k) ** 0.5
    bc = torch.randn((experts, n), generator=g) * 0.01 if bias else None
    weights = torch.stack([
        unpack_hf_int4_reference(q3[e], u3[e]).double() * d[e].double()[:, None] for e in range(experts)
    ])
    return (q3, u3, d, bc), weights


def _to_device(raw, k) -> PackedExperts:
    q3, u3, d, bc = raw
    return PackedExperts(
        q3.to(DEV), u3.to(DEV), d.to(DEV), signs_for_k(k, device=DEV), None if bc is None else bc.to(DEV)
    )


def _oracle_gemm(x_rows, ids, weights, bias):
    """FP64: row p = W[ids[p]] rot(x_rows[p]) + bias[ids[p]]; zeros for invalid ids."""
    rotated = _rotate64(x_rows.double())
    out = torch.zeros((ids.numel(), weights.shape[1]), dtype=torch.float64)
    for expert in range(weights.shape[0]):
        rows = torch.nonzero(ids == expert)[:, 0]
        if rows.numel():
            y = rotated[rows] @ weights[expert].T
            out[rows] = y if bias is None else y + bias[expert].double()
    return out


def _oracle_moe(x, topk_w, topk_ids, w13, bc13, w2, bc2, activation):
    """The documented semantics: BF16 gate/up, FP32 activation, FP32 weighted sum."""
    tokens, slots = topk_ids.shape
    ids = topk_ids.reshape(-1)
    x_pairs = x.repeat_interleave(slots, dim=0)
    gate_up = _oracle_gemm(x_pairs, ids, w13, bc13).float().bfloat16()
    gate, up = gate_up.chunk(2, dim=-1)
    inner = (ACTIVATIONS[activation](gate.float()) * up.float()).bfloat16()
    y = _oracle_gemm(inner, ids, w2, bc2).float()
    return (y.reshape(tokens, slots, -1) * topk_w.float()[:, :, None]).sum(dim=1).bfloat16()


def _routing(m, experts, slots, seed):
    probs = torch.randn((m, experts), generator=_gen(seed)).softmax(dim=-1)
    weights, ids = probs.topk(slots, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), ids


def _moe_case(m, experts, slots, hidden, inter, bias, seed):
    raw13, w13 = _random_stack(experts, 2 * inter, hidden, seed + 1, bias)
    raw2, w2 = _random_stack(experts, hidden, inter, seed + 2, bias)
    x = torch.randn((m, hidden), generator=_gen(seed + 3)).bfloat16()
    weights, ids = _routing(m, experts, slots, seed + 4)
    return (x, weights, ids), (_to_device(raw13, hidden), _to_device(raw2, inter)), (w13, raw13[3], w2, raw2[3])


def _run(inputs, stacks, activation, **kwargs):
    x, weights, ids = inputs
    out = hf_int4_g16_moe(x.to(DEV), weights.to(DEV), ids.to(DEV), *stacks, activation=activation, **kwargs)
    torch.xpu.synchronize()
    return out.cpu()


# ---------------------------------------------------------------------------
# Grouped GEMM and forward parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(1408, 2816), (2816, 704), (384, 256), (256, 192), (272, 1856)])
@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("bias", [False, True])
def test_pair_gemm_fp32_matches_fp64_oracle(n, k, backend, bias):
    experts, rows, slots = 6, 3, 4
    raw, weights = _random_stack(experts, n, k, seed=n + k, bias=bias)
    x = torch.randn((rows, k), generator=_gen(n * 3 + k)).bfloat16()
    ids = torch.randint(0, experts, (rows * slots,), generator=_gen(k), dtype=torch.int32)
    got = moe_pair_gemm(x.to(DEV), ids.to(DEV), _to_device(raw, k), in_div=slots,
                        out_dtype=torch.float32, backend=backend).cpu()
    want = _oracle_gemm(x.repeat_interleave(slots, dim=0), ids, weights, raw[3])
    assert got.dtype == torch.float32
    assert _rel(got, want) < 3e-6  # FP32 accumulation, no BF16 rounding anywhere
    assert ((got.double() - want).abs().max() / want.abs().max()).item() < 2e-5


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_pair_gemm_n_tail_columns(backend):
    """N % 32 != 0 leaves a partial last work-group (native needs N % 16 == 0)."""
    experts, n, k = 3, 48, 192
    raw, weights = _random_stack(experts, n, k, seed=40, bias=True)
    x = torch.randn((5, k), generator=_gen(41)).bfloat16()
    ids = torch.tensor([2, 0, 1, 2, 0], dtype=torch.int32)
    got = moe_pair_gemm(x.to(DEV), ids.to(DEV), _to_device(raw, k), out_dtype=torch.float32,
                        backend=backend).cpu()
    assert _rel(got, _oracle_gemm(x, ids, weights, raw[3])) < 3e-6


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("experts,slots", [(16, 4), (16, 8), (128, 4), (128, 8)])
@pytest.mark.parametrize("m", [1, 4, 32, 128])
def test_moe_forward_matches_fp32_reference(m, experts, slots, activation, bias):
    inputs, stacks, reference = _moe_case(m, experts, slots, 256, 192, bias, seed=m * 1000 + experts + slots)
    got = _run(inputs, stacks, activation, backend="native")
    want = _oracle_moe(*inputs, *reference, activation)
    assert got.dtype == torch.bfloat16 and got.shape == want.shape
    assert _rel(got.float(), want.double()) < 4e-3


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("hidden,inter", [(256, 192), (192, 256), (2816, 704)])
def test_moe_forward_mixed_had_geometry_and_activations(backend, activation, hidden, inter):
    inputs, stacks, reference = _moe_case(4, 8, 3, hidden, inter, True, seed=hidden + inter)
    got = _run(inputs, stacks, activation, backend=backend)
    want = _oracle_moe(*inputs, *reference, activation)
    assert _rel(got.float(), want.double()) < 4e-3


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_each_activation_is_identified_bit_exactly(activation):
    """erf-GELU and tanh-GELU differ by less than one BF16 rounding, so the tolerance
    tests cannot tell them apart.  Rebuild the pipeline per activation and require
    the forward to equal exactly the requested one."""
    inputs, (w13, w2), _ = _moe_case(64, 8, 3, 256, 192, False, seed=31)
    # Larger scales put the gate pre-activations where the three curves separate.
    w13 = PackedExperts(w13.q_pack, w13.u_pack, w13.d_row * 25, w13.signs)
    x, weights, ids = (t.to(DEV) for t in inputs)
    got = _run(inputs, (w13, w2), activation)
    flat = ids.reshape(-1).to(torch.int32)
    gate, up = moe_pair_gemm(x, flat, w13, 3, torch.bfloat16).chunk(2, dim=-1)
    for name, function in ACTIVATIONS.items():
        inner = (function(gate.float()) * up.float()).bfloat16()
        y = moe_pair_gemm(inner, flat, w2, 1, torch.float32).reshape(64, 3, 256)
        rebuilt = (y * weights.float()[:, :, None]).sum(dim=1).bfloat16()
        assert torch.equal(_bits(got), _bits(rebuilt)) == (name == activation), name


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("m", [1, 4, 32])
def test_bias_corr_none_equals_negative_zero_bit_exact(backend, m):
    inputs, (w13, w2), _ = _moe_case(m, 16, 8, 256, 192, False, seed=700 + m)

    def with_negative_zero(stack):
        zeros = torch.full((stack.num_experts, stack.n), -0.0, device=DEV)
        return PackedExperts(stack.q_pack, stack.u_pack, stack.d_row, stack.signs, zeros)

    negative = (with_negative_zero(w13), with_negative_zero(w2))
    assert torch.equal(_bits(_run(inputs, (w13, w2), "silu", backend=backend)),
                       _bits(_run(inputs, negative, "silu", backend=backend)))
    # The same contract at the FP32 stage, before any rounding can hide a difference.
    x, _, ids = inputs
    flat = ids.reshape(-1).to(torch.int32).to(DEV)
    plain = moe_pair_gemm(x.to(DEV), flat, w13, 8, torch.float32, backend)
    zeroed = moe_pair_gemm(x.to(DEV), flat, negative[0], 8, torch.float32, backend)
    assert torch.equal(_bits(plain), _bits(zeroed))
    # An all-zero input row produces exact zeros whose sign must also survive.
    silent = torch.zeros_like(x).to(DEV)
    assert torch.equal(_bits(moe_pair_gemm(silent, flat, w13, 8, torch.float32, backend)),
                       _bits(moe_pair_gemm(silent, flat, negative[0], 8, torch.float32, backend)))


def test_expert_outputs_stay_fp32_until_one_final_cast():
    inputs, (w13, w2), _ = _moe_case(64, 16, 8, 256, 192, True, seed=808)
    x, weights, ids = (t.to(DEV) for t in inputs)
    got = _run(inputs, (w13, w2), "silu")
    flat = ids.reshape(-1).to(torch.int32)
    gate, up = moe_pair_gemm(x, flat, w13, 8, torch.bfloat16).chunk(2, dim=-1)
    inner = (F.silu(gate.float()) * up.float()).bfloat16()
    y = moe_pair_gemm(inner, flat, w2, 1, torch.float32).reshape(64, 8, 256)
    once = (y * weights.float()[:, :, None]).sum(dim=1).bfloat16()
    per_expert = (y.bfloat16() * weights.bfloat16()[:, :, None]).sum(dim=1, dtype=torch.float32).bfloat16()
    assert torch.equal(_bits(got), _bits(once))
    # The rounding the design avoids really does change bits on this input.
    assert not torch.equal(_bits(got), _bits(per_expert))


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_expert_ids_outside_range_contribute_zero(backend):
    experts = 8
    inputs, stacks, reference = _moe_case(6, experts, 4, 256, 192, True, seed=909)
    x, weights, ids = inputs
    ids = ids.clone()
    ids[0, 1], ids[3, 0], ids[5] = -1, experts, torch.tensor([-1, experts, experts + 100, -7])
    got = _run((x, weights, ids), stacks, "silu", backend=backend)
    want = _oracle_moe(x, weights, ids, *reference, "silu")
    assert torch.equal(got[5], torch.zeros_like(got[5]))
    assert _rel(got.float(), want.double()) < 4e-3
    flat = ids.reshape(-1).to(torch.int32).to(DEV)
    pairs = moe_pair_gemm(x.to(DEV), flat, stacks[0], 4, torch.float32, backend).cpu()
    invalid = (ids.reshape(-1) < 0) | (ids.reshape(-1) >= experts)
    assert pairs[invalid].abs().max().item() == 0.0 and pairs[~invalid].abs().max().item() > 0


@pytest.mark.parametrize("chunk", [1, 5, 32])
def test_token_chunking_is_bit_identical(chunk):
    inputs, stacks, _ = _moe_case(33, 16, 8, 256, 192, True, seed=1001)
    assert torch.equal(_bits(_run(inputs, stacks, "silu")),
                       _bits(_run(inputs, stacks, "silu", chunk_tokens=chunk)))


def test_empty_batch():
    inputs, stacks, _ = _moe_case(1, 4, 2, 256, 192, False, seed=1002)
    x, weights, ids = (t[:0] for t in inputs)
    assert _run((x, weights, ids), stacks, "silu").shape == (0, 256)


@pytest.mark.parametrize("m", [1, 4, 32])
def test_xpu_graph_replay_follows_new_routing(m):
    """Capture once, then change activations, expert ids and weights in place.

    A host-side per-expert dispatch would replay the captured routing.
    """
    experts, slots = 16, 8
    inputs, stacks, _ = _moe_case(m, experts, slots, 256, 192, True, seed=1100 + m)
    static = [t.to(DEV) for t in inputs]

    def run():
        return hf_int4_g16_moe(static[0], static[1], static[2], *stacks, activation="silu", backend="native")

    for _ in range(2):
        run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = run()
    previous = None
    for seed in (1200, 1300, 1400):
        weights, ids = _routing(m, experts, slots, seed)
        static[0].copy_(torch.randn((m, 256), generator=_gen(seed + 1)).bfloat16().to(DEV))
        static[1].copy_(weights.to(DEV))
        static[2].copy_(ids.to(DEV))
        graph.replay()
        torch.xpu.synchronize()
        replayed = captured.clone()
        eager = run()
        torch.xpu.synchronize()
        assert torch.equal(_bits(replayed), _bits(eager)), f"stale replay at M={m} seed={seed}"
        assert previous is None or not torch.equal(_bits(replayed), previous)
        previous = _bits(replayed)


def test_forward_and_gemm_reject_bad_operands():
    inputs, (w13, w2), _ = _moe_case(2, 4, 2, 256, 192, False, seed=1500)
    x, weights, ids = (t.to(DEV) for t in inputs)
    call = lambda **kw: hf_int4_g16_moe(**{"x": x, "topk_weights": weights, "topk_ids": ids,
                                           "w13": w13, "w2": w2, "activation": "silu", **kw})
    with pytest.raises(ValueError, match="unsupported MoE activation"):
        call(activation="relu2")
    with pytest.raises(ValueError, match="topk_ids/topk_weights"):
        call(topk_ids=ids[:1])
    with pytest.raises(ValueError, match="int32 or int64"):
        call(topk_ids=ids.float())
    with pytest.raises(ValueError, match="hidden size"):
        call(w13=w2)
    with pytest.raises(ValueError, match="expert count mismatch"):
        short = PackedExperts(w2.q_pack[:3], w2.u_pack[:3], w2.d_row[:3], w2.signs)
        call(w2=short)
    with pytest.raises(ValueError, match="u_pack shape"):
        call(w13=PackedExperts(w13.q_pack, w13.u_pack[:, :2], w13.d_row, w13.signs))
    flat = ids.reshape(-1)
    with pytest.raises(ValueError, match="int32"):
        moe_pair_gemm(x, flat, w13, 2)
    with pytest.raises(ValueError, match="pairs"):
        moe_pair_gemm(x, flat.to(torch.int32), w13, 3)
    with pytest.raises(RuntimeError, match="preconditions"):
        moe_pair_gemm(x.to(torch.float16), flat.to(torch.int32), w13, 2, backend="native")


def test_fp16_activations_use_triton_under_auto():
    inputs, stacks, reference = _moe_case(3, 8, 2, 256, 192, True, seed=1600)
    x, weights, ids = inputs
    got = _run((x.to(torch.float16), weights, ids), stacks, "silu")
    assert got.dtype == torch.float16
    assert _rel(got.float(), _oracle_moe(*inputs, *reference, "silu").double()) < 1e-2


# ---------------------------------------------------------------------------
# Converter -> admission -> vLLM RoutedExperts load -> MoERunner forward
# ---------------------------------------------------------------------------
E, H, I, TOPK = 8, 256, 192, 3
# layout -> (checkpoint expert namespace, vLLM FusedMoE prefix, activation)
LAYOUTS = {
    "fused_gemma4": ("model.language_model.layers.0.experts", "model.layers.0.experts", "gelu_tanh"),
    "fused_weight_olmoe": ("model.layers.0.mlp.experts", "model.layers.0.mlp.experts", "silu"),
    "fused_granite": ("model.layers.0.block_sparse_moe.experts", "model.layers.0.block_sparse_moe", "silu"),
    "per_expert_qwen3moe": ("model.layers.0.mlp.experts", "model.layers.0.mlp.experts", "silu"),
}


def _source_weights(seed=77, experts=E, hidden=H, inter=I):
    g = _gen(seed)
    gate_up = (torch.randn((experts, 2 * inter, hidden), generator=g) * (1.0 / hidden ** 0.5)).bfloat16()
    down = (torch.randn((experts, hidden, inter), generator=g) * (1.0 / inter ** 0.5)).bfloat16()
    return gate_up, down


def _source_tensors(layout, gate_up, down):
    namespace = LAYOUTS[layout][0]
    experts, two_i, _ = gate_up.shape
    if layout == "fused_gemma4":
        return {f"{namespace}.gate_up_proj": gate_up, f"{namespace}.down_proj": down}
    if layout == "fused_weight_olmoe":
        return {f"{namespace}.gate_up_proj.weight": gate_up, f"{namespace}.down_proj.weight": down}
    if layout == "fused_granite":
        base = namespace.removesuffix(".experts")
        return {f"{base}.input_linear.weight": gate_up, f"{base}.output_linear.weight": down}
    tensors = {}
    for e in range(experts):
        tensors[f"{namespace}.{e}.gate_proj.weight"] = gate_up[e, :two_i // 2].clone()
        tensors[f"{namespace}.{e}.up_proj.weight"] = gate_up[e, two_i // 2:].clone()
        tensors[f"{namespace}.{e}.down_proj.weight"] = down[e].clone()
    return tensors


def _write_checkpoint(path, tensors, config):
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(config))


def _convert_layout(tmp_path, layout, **kwargs):
    gate_up, down = _source_weights()
    dense = torch.randn((H, I), generator=_gen(78)).bfloat16()  # shared down_proj, Had64
    namespace = LAYOUTS[layout][0]
    tensors = _source_tensors(layout, gate_up, down)
    tensors[namespace.rsplit(".layers.", 1)[0] + ".layers.0.shared_mlp.output_linear.weight"] = dense
    router = torch.randn((E, H), generator=_gen(79)).bfloat16()
    tensors["model.layers.0.mlp.gate.weight"] = router
    _write_checkpoint(tmp_path / "src", tensors, {"num_experts": E, "hidden_size": H})
    report = convert(tmp_path / "src", tmp_path / "out", device="xpu:0", **kwargs)
    return tmp_path / "out", report, (gate_up, down)


def _read_checkpoint(path):
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(path / shard), framework="pt", device="cpu") as f:
            tensors.update({k: f.get_tensor(k) for k in f.keys()})
    return tensors


def _admit(path):
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    manifest = json.loads((path / "quantization_config.json").read_text())
    config = HFInt4G16Config.from_config(manifest)
    config.maybe_update_config(str(path))
    return config


def _expert_items(path, layout):
    """``(name relative to the RoutedExperts layer, tensor)``, as vLLM's loader passes them."""
    prefix = LAYOUTS[layout][0] + "."
    return [(k[len(prefix):], v) for k, v in _read_checkpoint(path).items() if k.startswith(prefix)]


def _dequantized(path, layout):
    """Rotated-basis FP64 stacks and corrections decoded from the converted files."""
    tensors, namespace = _read_checkpoint(path), LAYOUTS[layout][0]

    def stack(proj):
        weights, bias = [], []
        for e in range(E):
            base = f"{namespace}.{e}.{proj}."
            q, u, d = (tensors[base + EXPERT_TENSOR_SUFFIXES[r]] for r in ("q_pack", "u_pack", "d_row"))
            weights.append(unpack_hf_int4_reference(q, u).double() * d.double()[:, None])
            bias.append(tensors.get(base + EXPERT_TENSOR_SUFFIXES["bias_corr"]))
        return torch.stack(weights), None if bias[0] is None else torch.stack(bias)

    gate, gate_bc = stack("gate_proj")
    up, up_bc = stack("up_proj")
    down, down_bc = stack("down_proj")
    bc13 = None if gate_bc is None else torch.cat([gate_bc, up_bc], dim=1)
    return torch.cat([gate, up], dim=1), bc13, down, down_bc


@pytest.fixture(scope="module")
def vllm_distributed():
    pytest.importorskip("vllm")
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    with tempfile.TemporaryDirectory() as rendezvous:
        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("xpu"))):
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                         distributed_init_method=f"file://{rendezvous}/init")
            initialize_model_parallel(1, 1)
            init_workspace_manager(torch.device(DEV))
        try:
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.fixture
def vllm_env(vllm_distributed):
    """A fresh VllmConfig per test: vLLM registers MoE layer names per config."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(device_config=DeviceConfig("xpu"))
    with set_current_vllm_config(config):
        yield config


def _build_moe(quant_config, prefix, activation="silu", **kwargs):
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory

    arguments = {"num_experts": E, "top_k": TOPK, "hidden_size": H, "intermediate_size": I,
                 "params_dtype": torch.bfloat16, "prefix": prefix, "activation": activation,
                 "quant_config": quant_config, **kwargs}
    with torch.device(DEV):
        return FusedMoEFactory(**arguments)


def _load_into(moe, path, layout):
    """Feed converted tensors to ``moe`` the way the vLLM model class would."""
    if layout != "fused_granite":
        return list(moe.load_weights(_expert_items(path, layout)))
    # GraniteMoeHybrid does not delegate to MoERunner.load_weights.  Run its
    # real model-level loader on a module tree with the same parameter paths.
    from vllm.model_executor.models.granitemoehybrid import GraniteMoeHybridModel

    class GraniteShell(torch.nn.Module):
        get_expert_mapping = GraniteMoeHybridModel.get_expert_mapping
        config = type("Config", (), {"num_local_experts": E})()

        def __init__(self):
            super().__init__()
            sparse_moe = torch.nn.Module()
            sparse_moe.experts = moe
            layer = torch.nn.Module()
            layer.block_sparse_moe = sparse_moe
            self.layers = torch.nn.ModuleList([layer])

    prefix = LAYOUTS[layout][0].removeprefix("model.")
    weights = [(f"{prefix}.{name}", tensor) for name, tensor in _expert_items(path, layout)]
    GraniteMoeHybridModel.load_weights(GraniteShell(), weights)
    return None  # vLLM's Granite loader does not report quantized expert names


def _dense_linear(config, out, prefix="model.layers.0.shared_mlp.output_linear"):
    """Build, load and process the converter's one dense linear [H, I] as vLLM would."""
    import vllm.model_executor.layers.linear as linear_mod

    with patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1), \
         patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: 0):
        layer = linear_mod.RowParallelLinear(I, H, bias=False, input_is_parallel=True, quant_config=config,
                                             prefix=prefix)
    return layer


def _load_dense(layer, out):
    tensors = {k: v for k, v in _read_checkpoint(out).items() if ".shared_mlp.output_linear." in k}
    for key, tensor in tensors.items():
        role = key.rsplit(".", 1)[1]
        if role in ("q_pack", "u_pack", "d_row"):
            parameter = getattr(layer, role)
            parameter.weight_loader(parameter, tensor, 0)
    layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _process(moe):
    moe.routed_experts.quant_method.process_weights_after_loading(moe.routed_experts)


def _forward(moe, x, logits, env):
    from vllm.forward_context import set_forward_context

    with set_forward_context(None, env), torch.inference_mode():
        # moe_forward declares hidden_states mutable, so hand it a private copy.
        out = moe(x.to(DEV).clone(), logits.to(DEV))
    torch.xpu.synchronize()
    return out.cpu()


def _router_reference(logits):
    weights, ids = logits.float().softmax(dim=-1).topk(TOPK, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), ids


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_converter_to_vllm_forward(tmp_path, vllm_env, layout):
    from hf_int4_g16_xpu.vllm_moe import HFInt4G16MoEMethod

    out, report, (gate_up, down) = _convert_layout(tmp_path, layout)
    assert report["format_version"] == "hf_int4_g16/2"
    assert report["stats"]["expert_namespaces"] == 1
    assert report["stats"]["quantized_expert_matrices"] == 3 * E
    assert report["stats"]["had64_expert_matrices"] == E  # down_proj K=192
    assert report["stats"]["quantized_linears"] == 1 and report["stats"]["had64_linears"] == 1
    keys = set(_read_checkpoint(out))
    namespace = LAYOUTS[layout][0]
    assert {f"{namespace}.{e}.{p}.qweight" for e in range(E) for p in EXPERT_PROJECTIONS} <= keys
    assert "model.layers.0.mlp.gate.weight" in keys  # router stays unquantized

    config = _admit(out)
    _, prefix, activation = LAYOUTS[layout]
    _load_dense(_dense_linear(config, out), out)
    moe = _build_moe(config, prefix, activation)
    assert isinstance(moe.routed_experts.quant_method, HFInt4G16MoEMethod)
    loaded = _load_into(moe, out, layout)
    if loaded is not None:
        assert len(loaded) == E * 3 * 3  # experts x projections x (qweight, scales, scale_super)
        assert set(loaded) == {f"w{g}_{s}" for g in ("13", "2") for s in ("qweight", "scales", "scale_super")}
    assert len(moe.routed_experts.hf_loaded_experts) == E * 3 * 3
    _process(moe)

    x = torch.randn((17, H), generator=_gen(80)).bfloat16()
    logits = torch.randn((17, E), generator=_gen(81)).bfloat16()
    got = _forward(moe, x, logits, vllm_env)
    weights, ids = _router_reference(logits)
    packed_reference = _oracle_moe(x, weights, ids, *_dequantized(out, layout), activation)
    assert _rel(got.float(), packed_reference.double()) < 6e-3

    # Against the original BF16 experts: only INT4 error remains.  Swapped
    # experts, a gate/up swap or a wrong rotation would be off by ~100%.
    original = torch.zeros((17, H), dtype=torch.float64)
    for t in range(17):
        for slot in range(TOPK):
            e = int(ids[t, slot])
            gate, up = (gate_up[e].double() @ x[t].double()).chunk(2)
            original[t] += weights[t, slot].double() * (
                down[e].double() @ (ACTIVATIONS[activation](gate) * up))
    assert _rel(got.float(), original) < 0.2


def test_every_expert_parameter_name_survives_granite_remap():
    """vLLM's GraniteMoeHybrid loader reaches ``routed_experts`` only through this remap."""
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader.weight_utils import maybe_remap_moe_expert_param_name

    base = "model.layers.0.block_sparse_moe.experts."
    for group in ("w13", "w2"):
        for suffix in EXPERT_TENSOR_SUFFIXES.values():
            real = f"{base}routed_experts.{group}_{suffix}"
            assert maybe_remap_moe_expert_param_name(f"{base}{group}_{suffix}", {real: None}) == real


@pytest.mark.parametrize("checkpoint_name", ["mamba", "mixer"])
def test_mamba_linear_resolves_through_vllm_mixer_prefix(checkpoint_name):
    """GraniteMoeHybrid checkpoints say "mamba", NemotronH ones "mixer"; vLLM asks for "mixer" in both."""
    pytest.importorskip("vllm")
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    from types import SimpleNamespace

    stored = f"layers.0.{checkpoint_name}.in_proj"
    config = HFInt4G16Config(quantized_modules={stored: {"shape": [1064, 256], "bias_corr": False}})
    model_type = "granitemoehybrid" if checkpoint_name == "mamba" else "nemotron_h"
    config.bind_architecture(SimpleNamespace(model_type=model_type))
    assert config._members_for("model.layers.0.mixer.in_proj") == [stored]
    assert config._members_for("model.layers.0.mixer.out_proj") is None
    assert config._members_for("model.layers.1.mixer.in_proj") is None


def test_dry_run_reports_expert_namespaces(tmp_path):
    gate_up, down = _source_weights()
    _write_checkpoint(tmp_path / "src", _source_tensors("fused_gemma4", gate_up, down), {"num_experts": E})
    report = dry_run_layer_mapping(tmp_path / "src")
    assert report["decoder_linear_candidates"] == 0
    [namespace] = report["expert_namespaces"]
    assert namespace["layout"] == "fused_3d" and namespace["num_experts"] == E
    assert namespace["had_block"] == {"gate_proj": 128, "up_proj": 128, "down_proj": 64}


@pytest.mark.parametrize("layout", ["fused_gemma4", "per_expert_qwen3moe"])
def test_converted_expert_bytes_are_independent_of_source_layout(tmp_path, layout):
    out, _, _ = _convert_layout(tmp_path / "a", layout)
    reference, _, _ = _convert_layout(tmp_path / "b", "fused_weight_olmoe")

    def experts(path, name):
        prefix = LAYOUTS[name][0] + "."
        return {k[len(prefix):]: v for k, v in _read_checkpoint(path).items() if k.startswith(prefix)}

    got, want = experts(out, layout), experts(reference, "fused_weight_olmoe")
    assert got.keys() == want.keys()
    assert all(torch.equal(got[k], want[k]) for k in got)


def test_skip_experts_keeps_them_bf16_and_dense_is_still_quantized(tmp_path):
    from hf_int4_g16_xpu.converter import DEFAULT_SKIP

    out, report, (gate_up, _) = _convert_layout(tmp_path, "fused_gemma4", skip_patterns=DEFAULT_SKIP + ",experts")
    tensors = _read_checkpoint(out)
    assert report["stats"]["expert_namespaces"] == 0
    assert torch.equal(tensors["model.language_model.layers.0.experts.gate_up_proj"], gate_up)
    assert "moe_modules" not in json.loads((out / "quantization_config.json").read_text())


@pytest.mark.parametrize("mutation,match", [
    ("missing_one", "missing expert tensors"),
    ("truncated_tail", "config.json declares 8"),
    ("wrong_down_shape", "expected"),
    ("expert_bias", "unrecognized routed-expert tensors"),
    ("k_off_grid", "multiple of 128"),
])
def test_converter_fails_closed_on_bad_expert_sources(tmp_path, mutation, match):
    gate_up, down = _source_weights()
    tensors = _source_tensors("per_expert_qwen3moe", gate_up, down)
    namespace = LAYOUTS["per_expert_qwen3moe"][0]
    if mutation == "missing_one":
        del tensors[f"{namespace}.3.up_proj.weight"]
    elif mutation == "truncated_tail":
        tensors = {k: v for k, v in tensors.items() if f".experts.{E - 1}." not in k}
    elif mutation == "wrong_down_shape":
        tensors[f"{namespace}.5.down_proj.weight"] = torch.zeros((H, I + 64), dtype=torch.bfloat16)
    elif mutation == "expert_bias":
        tensors[f"{namespace}.0.gate_proj.bias"] = torch.zeros(I, dtype=torch.bfloat16)
    else:
        gate_up, down = _source_weights(hidden=H, inter=200)
        tensors = _source_tensors("per_expert_qwen3moe", gate_up, down)
    _write_checkpoint(tmp_path / "src", tensors, {"num_experts": E})
    with pytest.raises(ValueError, match=match):
        convert(tmp_path / "src", tmp_path / "out", device="xpu:0")
    assert not (tmp_path / "out").exists()


def test_biascorr_scope_contradiction_rejected(tmp_path):
    with pytest.raises(ValueError, match="contradicts"):
        convert(tmp_path / "src", tmp_path / "out", biascorr=True, biascorr_scope="none")
    with pytest.raises(ValueError, match="biascorr_scope must be"):
        convert(tmp_path / "src", tmp_path / "out", biascorr_scope="everything")


# ---- strict admission: start from a valid conversion, break one thing -------
@pytest.fixture(scope="module")
def valid_checkpoint(tmp_path_factory):
    out, _, _ = _convert_layout(tmp_path_factory.mktemp("valid"), "per_expert_qwen3moe")
    return _read_checkpoint(out), json.loads((out / "quantization_config.json").read_text())


def _rewrite(path, tensors, manifest):
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")}))
    (path / "quantization_config.json").write_text(json.dumps(manifest))


NS = LAYOUTS["per_expert_qwen3moe"][0]
EXPERTS_KEY = "layers.0.mlp.experts"


def _drop(key):
    def mutate(tensors, manifest):
        del tensors[f"{NS}.{key}"]
    return mutate


def _replace(key, tensor):
    def mutate(tensors, manifest):
        tensors[f"{NS}.{key}"] = tensor
    return mutate


def _manifest(path, value):
    def mutate(tensors, manifest):
        node = manifest["moe_modules"][EXPERTS_KEY]
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
    return mutate


def _v1(tensors, manifest):
    manifest.update(format_version="hf_int4_g16/1", rotation="Had128NoPerm")


def _no_moe_manifest(tensors, manifest):
    del manifest["moe_modules"]


def _uncovered_sign_digest(tensors, manifest):
    del manifest["sign_sha256_by_k"][str(I)]


def _no_sign_digests(tensors, manifest):
    del manifest["sign_sha256_by_k"]


def _second_namespace_tensor(tensors, manifest):
    key = f"{NS}.2.up_proj.scales"
    tensors["language_model." + key] = tensors.pop(key)


@pytest.mark.parametrize("mutate,match", [
    pytest.param(_drop("2.up_proj.qweight"), "missing expert tensor expert=2 up_proj.qweight", id="missing-tensor"),
    pytest.param(_drop("7.down_proj.scale_super"), "missing expert tensor expert=7", id="missing-last-scale"),
    pytest.param(_replace("1.down_proj.qweight", torch.zeros((256 // 8, H), dtype=torch.int32)),
                 "wrong K or N", id="wrong-K"),
    pytest.param(_replace("1.gate_proj.scales", torch.zeros((H // 64, I), dtype=torch.int32)),
                 "dtype I32 != I16", id="wrong-dtype"),
    pytest.param(_manifest(["projections", "down_proj", "had_block"], 128),
                 "had_block 128 does not match K=192", id="had-block-128-for-K192"),
    pytest.param(_manifest(["projections", "gate_proj", "had_block"], 64),
                 "had_block 64 does not match K=256", id="had-block-64-for-K256"),
    pytest.param(_manifest(["hidden_size"], 200), "multiple of 128", id="manifest-K-off-grid"),
    pytest.param(_manifest(["num_experts"], E - 1), "unexpected expert tensor", id="extra-expert"),
    pytest.param(_manifest(["num_experts"], E + 1), "missing expert tensor expert=8", id="too-few-experts"),
    pytest.param(_manifest(["projections", "down_proj", "bias_corr"], True),
                 "missing expert tensor expert=0 down_proj.bias_corr", id="declared-bias-absent"),
    pytest.param(_replace("0.down_proj.bias_corr", torch.zeros(H)), "unexpected expert tensor",
                 id="undeclared-bias-present"),
    pytest.param(_manifest(["projections", "gate_proj", "bias_corr"], True), "share one packed tensor",
                 id="gate-up-bias-split"),
    pytest.param(_replace("0.gate_proj.weight", torch.zeros((I, H), dtype=torch.bfloat16)),
                 "unpacked tensors inside a packed expert namespace", id="leftover-bf16"),
    pytest.param(_second_namespace_tensor, "cross-namespace expert tensor mixing", id="cross-namespace"),
    pytest.param(_no_moe_manifest, "missing from moe_modules", id="undeclared-namespace"),
    pytest.param(_v1, "moe_modules require hf_int4_g16/2", id="v1-with-moe"),
    pytest.param(_uncovered_sign_digest, "missing digests for K", id="uncovered-sign-digest"),
    pytest.param(_no_sign_digests, "requires sign_sha256_by_k", id="no-sign-digests"),
])
def test_moe_header_admission_fails_closed(tmp_path, valid_checkpoint, mutate, match):
    pytest.importorskip("vllm")
    tensors, manifest = dict(valid_checkpoint[0]), json.loads(json.dumps(valid_checkpoint[1]))
    _rewrite(tmp_path / "control", tensors, manifest)
    assert list(_admit(tmp_path / "control").moe_modules) == [EXPERTS_KEY]
    mutate(tensors, manifest)
    _rewrite(tmp_path / "bad", tensors, manifest)
    with pytest.raises(ValueError, match=match):
        _admit(tmp_path / "bad")


def test_unindexed_expert_tensor_hidden_in_a_shard_is_rejected(tmp_path, valid_checkpoint):
    pytest.importorskip("vllm")
    tensors, manifest = dict(valid_checkpoint[0]), valid_checkpoint[1]
    _rewrite(tmp_path, tensors, manifest)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"][f"{NS}.0.gate_proj.qweight"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unindexed packed tensors"):
        _admit(tmp_path)


# ---- fail-closed at vLLM layer construction and load ------------------------
@pytest.fixture(scope="module")
def admitted(tmp_path_factory):
    out, _, _ = _convert_layout(tmp_path_factory.mktemp("admitted"), "per_expert_qwen3moe")
    return out


@pytest.mark.parametrize("kwargs,match", [
    ({"activation": "silu_no_mul"}, "unsupported activation"),
    ({"activation": "relu2"}, "unsupported activation"),
    ({"num_experts": 16}, "model has 16 experts"),
    ({"hidden_size": 384}, "model hidden size 384"),
    ({"intermediate_size": 256}, "model intermediate size 256"),
    ({"has_bias": True}, "expert model biases"),
    ({"swiglu_limit": 7.0}, "SwiGLU"),
    ({"apply_router_weight_on_input": True}, "apply_router_weight_on_input"),
])
def test_unsupported_layer_geometry_fails_at_construction(vllm_env, admitted, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build_moe(_admit(admitted), "model.layers.0.mlp.experts", **kwargs)


def test_forced_w4a8_is_rejected_for_experts(vllm_env, admitted, monkeypatch):
    monkeypatch.setenv("HF_INT4_G16_MODE", "w4a8")
    with pytest.raises(ValueError, match="w4a8 is not implemented for MoE"):
        _build_moe(_admit(admitted), "model.layers.0.mlp.experts")


def test_expert_layer_with_no_weights_anywhere_is_rejected(vllm_env, admitted):
    """Layer 1 has neither a packed bank nor BF16 weights: vLLM's unquantized
    method would run uninitialized memory."""
    with pytest.raises(ValueError, match="no packed moe_modules entry and no full-precision weights"):
        _build_moe(_admit(admitted), "model.layers.1.mlp.experts")


@pytest.mark.parametrize("dropped", ["0.gate_proj.qweight", "7.up_proj.scales", "4.down_proj.scale_super"])
def test_incomplete_expert_load_fails_before_first_forward(vllm_env, admitted, dropped):
    moe = _build_moe(_admit(admitted), "model.layers.0.mlp.experts")
    items = [i for i in _expert_items(admitted, "per_expert_qwen3moe") if i[0] != dropped]
    list(moe.load_weights(items))
    with pytest.raises(ValueError, match="incomplete expert loading: 1 missing"):
        _process(moe)
    x = torch.zeros((1, H), device=DEV, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not processed"):
        moe.routed_experts.quant_method.apply(
            moe.routed_experts, x, torch.ones((1, TOPK), device=DEV),
            torch.zeros((1, TOPK), device=DEV, dtype=torch.int32))


@pytest.mark.parametrize("name,tensor,match", [
    ("0.down_proj.scale_super", torch.full((H,), float("nan")), "NaN or Inf"),
    ("0.down_proj.scale_super", torch.zeros(H), "must be positive"),
    ("1.gate_proj.qweight", torch.zeros((H // 8, I), dtype=torch.int64), "refusing silent cast"),
    ("1.gate_proj.qweight", torch.zeros((H // 8, 2 * I), dtype=torch.int32), "shape"),
])
def test_bad_expert_payload_rejected_at_load(vllm_env, admitted, name, tensor, match):
    moe = _build_moe(_admit(admitted), "model.layers.0.mlp.experts")
    with pytest.raises(ValueError, match=match):
        list(moe.load_weights([(name, tensor)]))


def test_duplicate_expert_load_rejected(vllm_env, admitted):
    moe = _build_moe(_admit(admitted), "model.layers.0.mlp.experts")
    item = next(i for i in _expert_items(admitted, "per_expert_qwen3moe") if i[0] == "3.up_proj.qweight")
    list(moe.load_weights([item]))
    with pytest.raises(ValueError, match="duplicate load"):
        list(moe.load_weights([item]))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_expert_bias_corr_rejected_at_load(tmp_path, vllm_env, valid_checkpoint, value):
    """Header admission cannot see payload values, so the loader must."""
    tensors, manifest = dict(valid_checkpoint[0]), json.loads(json.dumps(valid_checkpoint[1]))
    manifest["moe_modules"][EXPERTS_KEY]["projections"]["down_proj"]["bias_corr"] = True
    for e in range(E):
        tensors[f"{NS}.{e}.down_proj.bias_corr"] = torch.zeros(H)
    tensors[f"{NS}.5.down_proj.bias_corr"] = torch.full((H,), value)
    _rewrite(tmp_path, tensors, manifest)
    moe = _build_moe(_admit(tmp_path), "model.layers.0.mlp.experts")
    with pytest.raises(ValueError, match="bias_corr expert=5.*NaN or Inf"):
        list(moe.load_weights(_expert_items(tmp_path, "per_expert_qwen3moe")))


# ---------------------------------------------------------------------------
# BiasCorr scopes against an independent whole-model oracle
# ---------------------------------------------------------------------------
def _tiny_hf_moe(family):
    import transformers

    common = dict(vocab_size=64, hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=4, head_dim=64, max_position_embeddings=512,
                  tie_word_embeddings=False)
    if family == "qwen3_moe":
        config = transformers.Qwen3MoeConfig(
            **common, intermediate_size=192, moe_intermediate_size=192, num_experts=E,
            num_experts_per_tok=TOPK, decoder_sparse_step=1, mlp_only_layers=[])
        return config, transformers.Qwen3MoeForCausalLM
    if family == "olmoe":
        config = transformers.OlmoeConfig(**common, intermediate_size=192, num_experts=E, num_experts_per_tok=TOPK)
        return config, transformers.OlmoeForCausalLM
    # Gemma-4: sliding + full attention, scaled embeddings, flattened router
    # input, shared dense MLP next to the MoE block, Had64 on both down_projs.
    config = transformers.Gemma4TextConfig(
        **common, intermediate_size=192, global_head_dim=64, enable_moe_block=True, num_experts=E,
        top_k_experts=TOPK, moe_intermediate_size=192, hidden_size_per_layer_input=0,
        sliding_window=8, layer_types=["sliding_attention", "full_attention"], attention_k_eq_v=True)
    return config, transformers.Gemma4ForCausalLM


def _unrotate(rotated_weight: torch.Tensor) -> torch.Tensor:
    """Original-basis dequantized weight: H is symmetric orthonormal, signs are +-1."""
    k = rotated_weight.shape[-1]
    block = 128 if k % 128 == 0 else 64
    h = make_hadamard(block, "cpu", torch.float64)
    flat = (rotated_weight.reshape(-1, block) @ h).reshape(rotated_weight.shape)
    return flat * signs_for_k(k).double()


def _routed_means(model, token_ids):
    """Per-layer expert means taken from an ordinary transformers forward."""
    captured, hooks = {}, []
    for name, module in model.named_modules():
        if type(module).__name__.endswith("Experts") and hasattr(module, "gate_up_proj"):
            def hook(mod, args, name=name):
                hidden, ids = args[0].detach(), args[1].detach()
                mean_x = torch.zeros((E, hidden.shape[1]), dtype=torch.float64)
                mean_down = torch.zeros((E, mod.gate_up_proj.shape[1] // 2), dtype=torch.float64)
                counts = torch.zeros(E, dtype=torch.long)
                for e in range(E):
                    rows = torch.nonzero((ids == e).any(dim=1))[:, 0]
                    rows = rows[rows % token_ids.shape[1] != 0]
                    counts[e] = rows.numel()
                    if rows.numel():
                        gate, up = F.linear(hidden[rows], mod.gate_up_proj[e]).chunk(2, dim=-1)
                        mean_down[e] = (mod.act_fn(gate) * up).double().mean(dim=0).cpu()
                        mean_x[e] = hidden[rows].double().mean(dim=0).cpu()
                captured[name] = (mean_x, mean_down, counts, mod.gate_up_proj.detach().cpu(),
                                  mod.down_proj.detach().cpu())
            hooks.append(module.register_forward_pre_hook(hook))
    with torch.no_grad():
        model(token_ids)
    for hook in hooks:
        hook.remove()
    return captured


@pytest.mark.parametrize("family", ["qwen3_moe", "olmoe", "gemma4"])
def test_biascorr_scopes_on_real_transformers_moe(tmp_path, family):
    pytest.importorskip("transformers")
    from test_converter_streaming import _tiny_tokenizer

    config, model_class = _tiny_hf_moe(family)
    torch.manual_seed(4242)
    model = model_class(config)
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__.endswith("Experts") and hasattr(module, "gate_up_proj"):
                module.gate_up_proj.normal_(0.0, 0.05)
                module.down_proj.normal_(0.0, 0.05)
    model = model.to(torch.bfloat16).eval()
    src = tmp_path / "src"
    model.save_pretrained(src, safe_serialization=True)
    _tiny_tokenizer(src)
    corpus = src / "corpus.txt"
    corpus.write_text(" ".join(("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                                "nu xi omicron pi rho sigma tau upsilon phi chi psi omega ").split() * 8))

    reports, checkpoints = {}, {}
    for scope in ("none", "shared_only", "all_experts"):
        kwargs = {} if scope == "none" else {"biascorr_corpus": str(corpus), "biascorr_tokens": 96}
        reports[scope] = convert(src, tmp_path / scope, device="xpu:0", biascorr_scope=scope, **kwargs)
        checkpoints[scope] = _read_checkpoint(tmp_path / scope)

    layers = config.num_hidden_layers
    for scope, report in reports.items():
        assert report["biascorr_scope"] == scope
        assert report["stats"]["expert_namespaces"] == layers
        assert report["stats"]["quantized_expert_matrices"] == layers * E * 3
        assert not any(".gate.weight.q_pack" in k or ".router" in k and k.endswith(".q_pack")
                       for k in checkpoints[scope])
    expert_bias = lambda t: sorted(k for k in t if ".experts." in k and k.endswith(".bias_corr"))
    dense_bias = lambda t: sorted(k for k in t if ".experts." not in k and k.endswith(".bias_corr"))
    assert not expert_bias(checkpoints["none"]) and not dense_bias(checkpoints["none"])
    assert not expert_bias(checkpoints["shared_only"]) and dense_bias(checkpoints["shared_only"])
    assert len(expert_bias(checkpoints["all_experts"])) == layers * E * 3
    assert dense_bias(checkpoints["all_experts"]) == dense_bias(checkpoints["shared_only"])
    # Scope changes only which corrections exist, never a packed weight.
    for key, tensor in checkpoints["none"].items():
        assert torch.equal(tensor, checkpoints["all_experts"][key]), key
    for key in dense_bias(checkpoints["shared_only"]):
        assert torch.equal(checkpoints["shared_only"][key], checkpoints["all_experts"][key]), key

    # Independent oracle: b = (W - W_dequantized) @ mean_x over routed tokens
    # after position 0, with means from the full transformers forward.
    calibration = json.loads((tmp_path / "all_experts" / "quantization_config.json").read_text())["calibration"]
    from transformers import AutoTokenizer

    token_ids = AutoTokenizer.from_pretrained(src).encode(corpus.read_text(), add_special_tokens=False)
    token_ids = torch.tensor([token_ids[:calibration["tokens"]]], device=DEV)
    # Reload from disk: Module.to(bfloat16) on the in-memory model also rounds the
    # rotary inv_freq buffer, which moves a few near-tie routing decisions.
    reference = model_class.from_pretrained(src, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    means = _routed_means(reference, token_ids)
    assert len(means) == layers
    tensors = checkpoints["all_experts"]
    namespaces = sorted({k.rsplit(".", 3)[0] for k in expert_bias(tensors)})
    checked = 0
    for module_name, (mean_x, mean_down, counts, gate_up, down) in means.items():
        layer_tag = module_name[module_name.index("layers."):] + "."
        [namespace] = [n for n in namespaces if n.endswith(layer_tag[:-1].removesuffix(".experts") + ".experts")]
        inter = gate_up.shape[1] // 2
        for e in range(E):
            for proj in EXPERT_PROJECTIONS:
                base = f"{namespace}.{e}.{proj}."
                stored = tensors[base + "bias_corr"].double()
                if counts[e] == 0:
                    assert stored.abs().max().item() == 0.0
                    continue
                q, u, d = (tensors[base + EXPERT_TENSOR_SUFFIXES[r]] for r in ("q_pack", "u_pack", "d_row"))
                dequantized = _unrotate(unpack_hf_int4_reference(q, u).double() * d.double()[:, None])
                original = {"gate_proj": gate_up[e, :inter], "up_proj": gate_up[e, inter:],
                            "down_proj": down[e]}[proj].double()
                mean = mean_down[e] if proj == "down_proj" else mean_x[e]
                want = (original - dequantized) @ mean
                assert (stored - want).norm() <= 2e-3 * want.norm() + 1e-7, (namespace, e, proj)
                checked += 1
    assert checked >= layers * 3 * TOPK
    # The converted checkpoint with every correction still passes strict admission.
    if torch.xpu.is_available():
        pytest.importorskip("vllm")
        assert len(_admit(tmp_path / "all_experts").moe_modules) == layers
