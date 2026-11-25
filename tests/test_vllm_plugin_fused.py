"""vLLM fused-loader seam exactness tests for HF-INT4-G16.

These tests use nonzero, index-coded bias corrections so any member permutation,
TP-slice error, or double-add is visible in the output value itself.  The FP64
oracle evaluates the exact packed integer weights in rotated space.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("vllm")
import vllm.model_executor.layers.linear as linear_mod
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from hf_int4_g16_xpu.format import (
    HAD_SEED,
    make_had128,
    pack_native_words,
    signs_for_k,
    unpack_hf_int4_reference,
)
from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
DEV = "xpu"


def _pack_case(n: int, k: int, seed: int, bc_offset: int = 0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    u = torch.randint(0, 16, (n, k // 16), generator=g, dtype=torch.uint8)
    qp, up = pack_native_words(q, u)
    d = (torch.rand(n, generator=g, dtype=torch.float32) * 0.00007 + 0.00001)
    bc = (torch.arange(n, dtype=torch.float32) + float(bc_offset) + 1.0) * (2.0 ** -10)
    return {"q_pack": qp, "u_pack": up, "d_row": d, "bias_corr": bc}


def _rotated_oracle(x: torch.Tensor, payloads: list[dict], widths: list[int]) -> torch.Tensor:
    """FP64 ideal W4A16 result in canonical rotated space.

    ``widths`` is only a semantic check here: concatenated payload rows must
    exactly cover the logical fused output partition order.
    """
    ws, bcs = [], []
    for p in payloads:
        wi = unpack_hf_int4_reference(p["q_pack"], p["u_pack"]).to(torch.float64)
        ws.append(wi * p["d_row"].to(torch.float64)[:, None])
        bcs.append(p["bias_corr"].to(torch.float64))
    w = torch.cat(ws, dim=0)
    bc = torch.cat(bcs, dim=0)
    assert w.shape[0] == sum(widths)
    k = x.shape[-1]
    signs = signs_for_k(k, HAD_SEED, "cpu").to(torch.float64)
    h = make_had128("cpu", torch.float64)
    xf = x.detach().cpu().to(torch.float64)
    xr = ((xf * signs).reshape(-1, 128) @ h).reshape(xf.shape)
    return xr @ w.T + bc


def _local_column_slice(y: torch.Tensor, widths: list[int], tp: int, rank: int) -> torch.Tensor:
    outs = []
    base = 0
    for w in widths:
        lw = w // tp
        outs.append(y[:, base + rank * lw: base + (rank + 1) * lw])
        base += w
    return torch.cat(outs, dim=1)


@contextmanager
def _fake_tp(size: int, rank: int):
    with patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: size), \
         patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: rank):
        yield


def _config(prefix_base: str, member_specs: list[tuple[str, int, int]]):
    mods = {}
    for leaf, n, k in member_specs:
        mods[f"{prefix_base}.{leaf}"] = {"shape": [n, k], "bias_corr": True}
    cfg = HFInt4G16Config(quantized_modules=mods)
    cfg.validate_format()
    return cfg


def _load_member(layer, payload: dict, shard_id):
    for kind in ("q_pack", "u_pack", "d_row", "bias_corr"):
        p = getattr(layer, kind)
        p.weight_loader(p, payload[kind], shard_id)


def _finish_and_run(layer, x):
    layer = layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    # backend=auto exercises native fused decode where qualified and the
    # required exact Triton W4A16 fallback for row-parallel prefill.
    y = layer.quant_method.apply(layer, x.to(DEV))
    torch.xpu.synchronize()
    return y.detach().cpu()


def _column_case(kind: str, tp: int, rank: int):
    k = 256
    if kind == "qkv":
        widths = [256, 256, 256]
        specs = [("q_proj", 256, k), ("k_proj", 256, k), ("v_proj", 256, k)]
        cfg = _config("layers.0.self_attn", specs)
        with _fake_tp(tp, rank):
            layer = QKVParallelLinear(k, 64, 4, 4, bias=False, quant_config=cfg,
                                      prefix="model.layers.0.self_attn.qkv_proj")
        ids = ["q", "k", "v"]
    elif kind == "gate_up":
        widths = [256, 256]
        specs = [("gate_proj", 256, k), ("up_proj", 256, k)]
        cfg = _config("layers.0.mlp", specs)
        with _fake_tp(tp, rank):
            layer = MergedColumnParallelLinear(k, widths, bias=False, quant_config=cfg,
                                               prefix="model.layers.0.mlp.gate_up_proj")
        ids = [0, 1]
    elif kind == "qkvz":
        widths = [128, 128, 256, 256]
        specs = [("in_proj_qkv", 512, k), ("in_proj_z", 256, k)]
        cfg = _config("layers.0.linear_attn", specs)
        with _fake_tp(tp, rank):
            layer = MergedColumnParallelLinear(k, widths, bias=False, quant_config=cfg,
                                               prefix="model.layers.0.linear_attn.in_proj_qkvz")
        ids = [(0, 1, 2), 3]
    elif kind == "ba":
        widths = [64, 64]
        specs = [("in_proj_b", 64, k), ("in_proj_a", 64, k)]
        cfg = _config("layers.0.linear_attn", specs)
        with _fake_tp(tp, rank):
            layer = MergedColumnParallelLinear(k, widths, bias=False, quant_config=cfg,
                                               prefix="model.layers.0.linear_attn.in_proj_ba")
        ids = [0, 1]
    else:
        raise AssertionError(kind)
    payloads = []
    bc_offset = 0
    for i, (_, n, _) in enumerate(specs):
        payloads.append(_pack_case(n, k, 9100 + i * 17 + len(kind), bc_offset=bc_offset))
        bc_offset += n
    for payload, sid in zip(payloads, ids):
        _load_member(layer, payload, sid)
    return layer, payloads, widths, k


@pytest.mark.parametrize("kind", ["qkv", "gate_up", "qkvz", "ba"])
@pytest.mark.parametrize("tp", [1, 2])
def test_indexed_bias_fused_column_members_exact_vs_fp64(kind, tp):
    for rank in range(tp):
        layer, payloads, widths, k = _column_case(kind, tp, rank)
        x = torch.randn((2, k), generator=torch.Generator().manual_seed(6400 + tp * 10 + rank)).bfloat16()
        got = _finish_and_run(layer, x)
        full = _rotated_oracle(x, payloads, widths)
        want = _local_column_slice(full, widths, tp, rank)
        diff = (got.to(torch.float64) - want).abs()
        assert got.shape == want.shape
        assert diff.max().item() < 1e-2, f"{kind} tp{tp} rank{rank} maxdiff={diff.max().item()}"
        # Index-coded values are actually nonzero and span channels; an all-zero
        # correction or accidental member-wide broadcast cannot pass accidentally.
        assert torch.cat([p["bias_corr"] for p in payloads]).abs().max() > 0.1


@pytest.mark.parametrize("tp", [1, 2])
def test_indexed_bias_row_parallel_exact_vs_fp64(tp):
    n, k = 256, 512
    spec = [("down_proj", n, k)]
    cfg = _config("layers.0.mlp", spec)
    payload = _pack_case(n, k, 9701)
    x = torch.randn((8, k), generator=torch.Generator().manual_seed(9800)).bfloat16()
    partials = []
    # This fixture is the explicitly proven external-single-sum consumer.  The
    # plugin therefore requires the same explicit opt-in a production custom
    # consumer would need; the default remains fail-closed.
    with patch.dict(os.environ, {"HF_INT4_G16_ALLOW_UNREDUCED_ROW_TP": "1"}):
        for rank in range(tp):
            with _fake_tp(tp, rank):
                layer = RowParallelLinear(k, n, bias=False, input_is_parallel=True,
                                          reduce_results=False, quant_config=cfg,
                                          prefix="model.layers.0.mlp.down_proj")
            _load_member(layer, payload, 0)
            xr = x[:, rank * (k // tp):(rank + 1) * (k // tp)].contiguous()
            partials.append(_finish_and_run(layer, xr))
    got = sum(partials)
    want = _rotated_oracle(x, [payload], [n])
    diff = (got.to(torch.float64) - want).abs()
    assert diff.max().item() < 1e-2, f"row tp{tp} maxdiff={diff.max().item()}"


def test_row_parallel_tp2_is_forced_w4a16(monkeypatch):
    # Capture the mode argument at the plugin boundary.  TP>1 row sharding must
    # never select local-amax W4A8, even for prefill-sized M.
    import hf_int4_g16_xpu.vllm_plugin as vp
    calls = []
    real = vp.hf_int4_g16_linear
    def spy(*args, **kwargs):
        calls.append(kwargs.get("mode", "auto"))
        return real(*args, **kwargs)
    monkeypatch.setattr(vp, "hf_int4_g16_linear", spy)
    test_indexed_bias_row_parallel_exact_vs_fp64(2)
    assert calls and all(m == "w4a16" for m in calls)


def test_fail_closed_partial_fused_missing_member():
    cfg = _config("layers.0.mlp", [("gate_proj", 256, 256)])
    with pytest.raises(ValueError, match="partial fused"):
        cfg._members_for("model.layers.0.mlp.gate_up_proj")


def test_fail_closed_k_not_multiple_of_128():
    cfg = _config("layers.0.mlp", [("gate_proj", 256, 120), ("up_proj", 256, 120)])
    with pytest.raises(ValueError, match="multiple of 128"):
        with _fake_tp(1, 0):
            MergedColumnParallelLinear(120, [256, 256], bias=False, quant_config=cfg,
                                       prefix="model.layers.0.mlp.gate_up_proj")


def test_fail_closed_orphan_and_missing_biascorr(tmp_path):
    base = "layers.0.mlp.gate_proj"
    base_qc = {"group_size": 16, "hadamard_block": 128, "had_seed": 101,
               "rotation": "Had128NoPerm", "layout": "xe2_dpas_native_v1",
               "format_version": "hf_int4_g16/1", "quantized_modules": {}}
    _write_admission_fixture(tmp_path, base_qc, [base + ".bias_corr"])
    cfg = HFInt4G16Config()
    with pytest.raises(ValueError, match="orphan .*bias_corr"):
        cfg.maybe_update_config(str(tmp_path))

    qc = dict(base_qc)
    qc["quantized_modules"] = {base: {"shape": [256, 256], "bias_corr": True}}
    keys = [base + ".q_pack", base + ".u_pack", base + ".d_row"]
    _write_admission_fixture(tmp_path, qc, keys)
    cfg = HFInt4G16Config()
    with pytest.raises(ValueError, match="manifest bias_corr=True but checkpoint presence=False"):
        cfg.maybe_update_config(str(tmp_path))


def _blank_gate_up(k: int = 256, n: int = 256):
    specs = [("gate_proj", n, k), ("up_proj", n, k)]
    cfg = _config("layers.0.mlp", specs)
    with _fake_tp(1, 0):
        layer = MergedColumnParallelLinear(k, [n, n], bias=False, quant_config=cfg,
                                           prefix="model.layers.0.mlp.gate_up_proj")
    payloads = [_pack_case(n, k, 12000 + i * 31, bc_offset=i * n) for i in range(2)]
    return layer, payloads, [0, 1]


def test_shuffled_member_and_suffix_load_order_exact():
    layer, payloads, ids = _blank_gate_up()
    # Reverse both member order and suffix order.  Coverage must be semantic
    # (suffix, member-run), not arrival-order or q_pack-only bookkeeping.
    for payload, sid in zip(reversed(payloads), reversed(ids)):
        for kind in ("bias_corr", "d_row", "u_pack", "q_pack"):
            getattr(layer, kind).weight_loader(getattr(layer, kind), payload[kind], sid)
    x = torch.randn((2, 256), generator=torch.Generator().manual_seed(12100)).bfloat16()
    got = _finish_and_run(layer, x)
    want = _rotated_oracle(x, payloads, [256, 256])
    diff = (got.to(torch.float64) - want).abs()
    assert diff.max().item() < 1e-2


@pytest.mark.parametrize("missing", ["u_pack", "d_row", "bias_corr"])
def test_missing_required_suffix_fail_closed(missing):
    layer, payloads, ids = _blank_gate_up()
    for payload, sid in zip(payloads, ids):
        for kind in ("q_pack", "u_pack", "d_row", "bias_corr"):
            if kind == missing:
                continue
            getattr(layer, kind).weight_loader(getattr(layer, kind), payload[kind], sid)
    layer = layer.to(DEV)
    with pytest.raises(ValueError, match="incomplete fused suffix loading"):
        layer.quant_method.process_weights_after_loading(layer)


def test_duplicate_suffix_member_load_rejected():
    layer, payloads, ids = _blank_gate_up()
    payload, sid = payloads[0], ids[0]
    for kind in ("q_pack", "u_pack", "d_row", "bias_corr"):
        getattr(layer, kind).weight_loader(getattr(layer, kind), payload[kind], sid)
    with pytest.raises(ValueError, match="duplicate load"):
        layer.u_pack.weight_loader(layer.u_pack, payload["u_pack"], sid)


@pytest.mark.parametrize("kind,bad_dtype", [
    ("q_pack", torch.int64),
    ("u_pack", torch.int32),
    ("d_row", torch.float64),
    ("bias_corr", torch.bfloat16),
])
def test_wrong_suffix_dtype_rejected_before_copy(kind, bad_dtype):
    layer, payloads, ids = _blank_gate_up()
    bad = payloads[0][kind].to(bad_dtype)
    with pytest.raises(ValueError, match="dtype mismatch"):
        getattr(layer, kind).weight_loader(getattr(layer, kind), bad, ids[0])


def test_reload_after_process_rejected():
    layer, payloads, ids = _blank_gate_up()
    for payload, sid in zip(payloads, ids):
        _load_member(layer, payload, sid)
    layer = layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    with pytest.raises(ValueError, match="already-processed"):
        layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.parametrize("rank", [0, 1])
def test_replicated_kv_uses_replica_group_not_raw_tp_rank(rank):
    # Two q heads but one physical KV head under TP2.  vLLM logically replicates
    # that KV head on both ranks; rank 1 must load the same checkpoint rows as
    # rank 0 rather than advancing by tp_rank * local_width.
    k = 256
    specs = [("q_proj", 128, k), ("k_proj", 64, k), ("v_proj", 64, k)]
    cfg = _config("layers.0.self_attn", specs)
    with _fake_tp(2, rank):
        layer = QKVParallelLinear(k, 64, 2, 1, bias=False, quant_config=cfg,
                                  prefix="model.layers.0.self_attn.qkv_proj")
    assert layer.num_kv_head_replicas == 2
    payloads = []
    bc_offset = 0
    for i, (_, n, _) in enumerate(specs):
        payloads.append(_pack_case(n, k, 12200 + i * 19, bc_offset=bc_offset))
        bc_offset += n
    for payload, sid in zip(payloads, ["q", "k", "v"]):
        _load_member(layer, payload, sid)
    x = torch.randn((2, k), generator=torch.Generator().manual_seed(12300 + rank)).bfloat16()
    got = _finish_and_run(layer, x)
    qfull = _rotated_oracle(x, [payloads[0]], [128])
    kfull = _rotated_oracle(x, [payloads[1]], [64])
    vfull = _rotated_oracle(x, [payloads[2]], [64])
    want = torch.cat([qfull[:, rank * 64:(rank + 1) * 64], kfull, vfull], dim=1)
    diff = (got.to(torch.float64) - want).abs()
    assert got.shape == (2, 192)
    assert diff.max().item() < 1e-2, f"replicated-kv rank{rank} maxdiff={diff.max().item()}"


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("oversized", ["k_proj", "v_proj"])
def test_replicated_kv_rejects_logical_width_and_accepts_physical_width(rank, oversized):
    # head_dim=64, one physical KV head, TP2 -> num_kv_head_replicas=2.  A
    # 128-row K or V member equals the *logically replicated* width; accepting it
    # would silently drop rows 64..127 on both ranks.
    k = 256

    def build(widths: dict[str, int]):
        cfg = _config("layers.0.self_attn", [(leaf, n, k) for leaf, n in widths.items()])
        with _fake_tp(2, rank):
            return QKVParallelLinear(k, 64, 2, 1, bias=False, quant_config=cfg,
                                     prefix="model.layers.0.self_attn.qkv_proj")

    exact = {"q_proj": 128, "k_proj": 64, "v_proj": 64}
    layer = build(exact)
    assert layer.num_kv_head_replicas == 2
    assert layer.hf_physical_output_sizes == [128, 64, 64]
    with pytest.raises(ValueError, match="replicated K/V physical width mismatch"):
        build({**exact, oversized: 128})


def test_physical_member_n_mismatch_rejected_before_loading():
    cfg = _config("layers.0.mlp", [("gate_proj", 257, 256), ("up_proj", 256, 256)])
    with pytest.raises(ValueError, match="physical output width mismatch"):
        with _fake_tp(1, 0):
            MergedColumnParallelLinear(256, [256, 256], bias=False, quant_config=cfg,
                                       prefix="model.layers.0.mlp.gate_up_proj")


def test_reduce_results_false_rejected_by_default(monkeypatch):
    monkeypatch.delenv("HF_INT4_G16_ALLOW_UNREDUCED_ROW_TP", raising=False)
    cfg = _config("layers.0.mlp", [("down_proj", 256, 512)])
    with pytest.raises(ValueError, match="requires reduce_results=True"):
        with _fake_tp(2, 0):
            RowParallelLinear(512, 256, bias=False, input_is_parallel=True,
                              reduce_results=False, quant_config=cfg,
                              prefix="model.layers.0.mlp.down_proj")


def test_row_parallel_zero_input_model_bias_and_correction():
    n, k = 256, 512
    cfg = _config("layers.0.mlp", [("down_proj", n, k)])
    payload = _pack_case(n, k, 12400)
    with _fake_tp(2, 0):
        layer = RowParallelLinear(k, n, bias=True, input_is_parallel=True,
                                  reduce_results=True, quant_config=cfg,
                                  prefix="model.layers.0.mlp.down_proj")
    _load_member(layer, payload, 0)
    layer = layer.to(DEV)
    layer.bias.data.copy_(torch.randn(n, generator=torch.Generator().manual_seed(12401)).to(DEV).bfloat16())
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.zeros((4, k // 2), device=DEV, dtype=torch.bfloat16)
    got = layer.quant_method.apply(layer, x, layer.bias)
    torch.xpu.synchronize()
    want = layer.bias.float().cpu() + payload["bias_corr"].float()
    diff = (got.cpu().float() - want).abs()
    assert diff.max().item() < 2e-2
    assert layer.bias.abs().max() > 0
    assert payload["bias_corr"].abs().max() > 0


def test_row_parallel_skip_bias_add_defers_model_bias_but_not_correction(monkeypatch):
    n, k = 256, 512
    monkeypatch.setenv("HF_INT4_G16_ALLOW_UNREDUCED_ROW_TP", "1")
    cfg = _config("layers.0.mlp", [("down_proj", n, k)])
    payload = _pack_case(n, k, 12500)
    with _fake_tp(2, 0):
        layer = RowParallelLinear(k, n, bias=True, input_is_parallel=True,
                                  reduce_results=False, skip_bias_add=True, quant_config=cfg,
                                  prefix="model.layers.0.mlp.down_proj")
    _load_member(layer, payload, 0)
    layer = layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.zeros((2, k // 2), device=DEV, dtype=torch.bfloat16)
    out, deferred = layer(x)
    torch.xpu.synchronize()
    assert deferred is layer.bias
    # Zero input isolates the epilogue: the architectural bias is deferred, while
    # the quantization correction is still applied exactly once on rank 0.
    assert torch.allclose(out.cpu().float(), payload["bias_corr"].float().unsqueeze(0).expand_as(out.cpu().float()), atol=2e-2, rtol=1e-3)


def test_biascorr_mode_off_passes_true_none(monkeypatch):
    import hf_int4_g16_xpu.vllm_plugin as vp
    monkeypatch.setenv("HF_INT4_G16_BIASCORR_MODE", "off")
    layer, payloads, ids = _blank_gate_up()
    for payload, sid in zip(payloads, ids):
        _load_member(layer, payload, sid)
    layer = layer.to(DEV)
    layer.quant_method.process_weights_after_loading(layer)
    seen = []
    real = vp.hf_int4_g16_linear
    def spy(*args, **kwargs):
        seen.append(kwargs.get("bias_corr", "missing"))
        return real(*args, **kwargs)
    monkeypatch.setattr(vp, "hf_int4_g16_linear", spy)
    x = torch.randn((2, 256), device=DEV, dtype=torch.bfloat16)
    layer.quant_method.apply(layer, x)
    torch.xpu.synchronize()
    assert seen and seen[0] is None


def _write_admission_fixture(tmp_path, qc, index_keys, n: int = 256, k: int = 256,
                             write_shard: bool = True, shard_name: str = "model-00001.safetensors",
                             shard_keys=None):
    """Write a real minimal safetensors shard plus index/config for strict admission."""
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: shard_name for key in index_keys}
    }))
    (tmp_path / "quantization_config.json").write_text(json.dumps(qc))
    if not write_shard:
        return
    tensors = {}
    for key in (index_keys if shard_keys is None else shard_keys):
        if key.endswith(".q_pack"):
            tensors[key] = torch.zeros((k // 8, n), dtype=torch.int32)
        elif key.endswith(".u_pack"):
            tensors[key] = torch.zeros((k // 64, n), dtype=torch.int16)
        elif key.endswith(".d_row"):
            tensors[key] = torch.ones((n,), dtype=torch.float32)
        elif key.endswith(".bias_corr"):
            tensors[key] = torch.zeros((n,), dtype=torch.float32)
    from safetensors.torch import save_file
    save_file(tensors, str(tmp_path / shard_name))


def test_sign_digest_accepted_and_mismatch_rejected(tmp_path):
    from hf_int4_g16_xpu.format import sign_sha256_for_k
    base = "layers.0.mlp.gate_proj"
    keys = [base + ".q_pack", base + ".u_pack", base + ".d_row"]
    qc = {"group_size": 16, "hadamard_block": 128, "had_seed": 101,
          "rotation": "Had128NoPerm", "layout": "xe2_dpas_native_v1",
          "format_version": "hf_int4_g16/1",
          "sign_sha256_by_k": {"256": sign_sha256_for_k(256)},
          "quantized_modules": {base: {"shape": [256, 256], "bias_corr": False}}}
    _write_admission_fixture(tmp_path, qc, keys)
    cfg = HFInt4G16Config()
    cfg.maybe_update_config(str(tmp_path))
    assert base in cfg.quantized_modules

    bad = dict(qc)
    bad["sign_sha256_by_k"] = {"256": "0" * 64}
    _write_admission_fixture(tmp_path, bad, keys)
    with pytest.raises(ValueError, match="sign digest mismatch"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))


def test_strict_admission_negative_matrix(tmp_path):
    from hf_int4_g16_xpu.format import sign_sha256_for_k
    base = "layers.0.mlp.gate_proj"
    qc = {"group_size": 16, "hadamard_block": 128, "had_seed": 101,
          "rotation": "Had128NoPerm", "layout": "xe2_dpas_native_v1",
          "format_version": "hf_int4_g16/1",
          "quantized_modules": {base: {"shape": [256, 256], "bias_corr": False}}}
    keys = [base + ".q_pack", base + ".u_pack", base + ".d_row"]

    # The index references a shard that does not exist.
    _write_admission_fixture(tmp_path, qc, keys, write_shard=False)
    with pytest.raises(FileNotFoundError, match="checkpoint shard"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))

    # q_pack comes from one raw namespace while u_pack/d_row come from another.
    raw_q = "model." + base + ".q_pack"
    raw_u = "language_model.model." + base + ".u_pack"
    raw_d = "language_model.model." + base + ".d_row"
    _write_admission_fixture(tmp_path, qc, [raw_q, raw_u, raw_d])
    with pytest.raises(ValueError, match="cross-namespace packed suffix mixing"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))

    # Manifest correction absence must not be promoted by an index correction key.
    _write_admission_fixture(tmp_path, qc, keys + [base + ".bias_corr"])
    with pytest.raises(ValueError, match="manifest bias_corr=False but checkpoint presence=True"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))

    # A complete packed module absent from the quantization manifest is undeclared.
    qc_no_manifest = dict(qc)
    qc_no_manifest["quantized_modules"] = {}
    _write_admission_fixture(tmp_path, qc_no_manifest, keys)
    with pytest.raises(ValueError, match="missing from quantization manifest"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))

    # Digest coverage is keyed by the actual q_pack header K, not manifest prose.
    qc_uncovered = dict(qc)
    qc_uncovered["sign_sha256_by_k"] = {"512": sign_sha256_for_k(512)}
    _write_admission_fixture(tmp_path, qc_uncovered, keys)
    with pytest.raises(ValueError, match="missing digests for K"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))


@pytest.mark.parametrize("namespace", ["", "model."])
def test_unindexed_shard_bias_corr_is_rejected(tmp_path, namespace):
    # The shard really holds q/u/d AND .bias_corr, while both the index and the
    # manifest claim there is no correction.  Index->header membership alone
    # cannot see this; admission must compare the inventories in both directions.
    base = "layers.0.mlp.gate_proj"
    qc = {"group_size": 16, "hadamard_block": 128, "had_seed": 101,
          "rotation": "Had128NoPerm", "layout": "xe2_dpas_native_v1",
          "format_version": "hf_int4_g16/1",
          "quantized_modules": {base: {"shape": [256, 256], "bias_corr": False}}}
    keys = [namespace + base + sfx for sfx in (".q_pack", ".u_pack", ".d_row")]
    _write_admission_fixture(tmp_path, qc, keys)
    HFInt4G16Config().maybe_update_config(str(tmp_path))  # control: consistent checkpoint admits

    _write_admission_fixture(tmp_path, qc, keys, shard_keys=keys + [namespace + base + ".bias_corr"])
    with pytest.raises(ValueError, match=r"unindexed packed tensors.*bias_corr"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))


def test_missing_metadata_and_format_version_fail_closed(tmp_path):
    qc = {"group_size": 16, "hadamard_block": 128, "had_seed": 101,
          "rotation": "Had128NoPerm", "layout": "xe2_dpas_native_v1",
          "format_version": "hf_int4_g16/1", "quantized_modules": {}}
    (tmp_path / "quantization_config.json").write_text(json.dumps(qc))
    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))

    bad = dict(qc)
    bad["format_version"] = "hf_int4_g16/0"
    with pytest.raises(ValueError, match="unsupported layout/version"):
        HFInt4G16Config(format_version="hf_int4_g16/0").validate_format()


def test_packed_expert_keys_fail_closed(tmp_path):
    base = "layers.0.mlp.experts.0.gate_proj"
    keys = [base + ".q_pack", base + ".u_pack", base + ".d_row"]
    qc = {"group_size": 16, "hadamard_block": 128, "had_seed": 101,
          "rotation": "Had128NoPerm", "layout": "xe2_dpas_native_v1",
          "format_version": "hf_int4_g16/1", "quantized_modules": {}}
    _write_admission_fixture(tmp_path, qc, keys)
    with pytest.raises(ValueError, match="expert/MoE"):
        HFInt4G16Config().maybe_update_config(str(tmp_path))


def test_linear_mode_default_strict_and_explicit_auto(monkeypatch):
    from hf_int4_g16_xpu.vllm_plugin import linear_mode
    monkeypatch.delenv("HF_INT4_G16_MODE", raising=False)
    assert linear_mode() == "w4a16"
    monkeypatch.setenv("HF_INT4_G16_MODE", "auto")
    assert linear_mode() == "auto"
    monkeypatch.setenv("HF_INT4_G16_MODE", "w4a8")
    assert linear_mode() == "w4a8"
