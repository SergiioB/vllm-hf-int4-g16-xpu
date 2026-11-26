#!/usr/bin/env python3
"""Latency of one packed routed-expert layer on realistic MoE geometries.

Random packed experts, random router logits.  Every number is the median of
synchronized single calls after warm-up.  ``stock_bf16_runner_ms`` is vLLM's
own UnquantizedFusedMoEMethod through the same MoERunner forward (router
included), so compare it with ``runner_ms``, not with ``experts_ms``.

    python3 bench_moe.py --out exp17_moe_plugin_bench_gpu0.json [--hwmon /sys/class/hwmon/hwmonN]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch

from hf_int4_g16_xpu.format import had_block_for_k, signs_for_k
from hf_int4_g16_xpu.moe import MOE_ACTIVATIONS, PackedExperts, hf_int4_g16_moe, moe_pair_gemm
from hf_int4_g16_xpu.moe_admission import ExpertProjection, MoEModule

DEV = "xpu"
SHAPES = {
    "gemma4_26b_a4b": {"E": 128, "topk": 8, "H": 2816, "I": 704, "activation": "gelu_tanh"},
    "qwen36_35b_a3b": {"E": 256, "topk": 8, "H": 2048, "I": 512, "activation": "silu"},
}
BATCHES = (1, 8, 64, 512)
REPS = {1: 200, 8: 100, 64: 50, 512: 20}
TRITON_REPS = {1: 20, 8: 10, 64: 5, 512: 3}


class Sensors:
    def __init__(self, hwmon: str | None):
        self.root = Path(hwmon) if hwmon else None

    def read(self, name: str) -> int | None:
        if self.root is None:
            return None
        try:
            return int((self.root / name).read_text())
        except OSError:
            return None

    def temp_c(self) -> float | None:
        value = self.read("temp2_input")
        return None if value is None else value / 1000

    def cool(self, target: float = 55.0, limit_s: float = 120.0) -> float | None:
        start = time.monotonic()
        while (temp := self.temp_c()) is not None and temp > target and time.monotonic() - start < limit_s:
            time.sleep(2)
        return temp


def sync():
    torch.xpu.synchronize()


def timed(fn, reps: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p10_ms": round(ordered[int(0.1 * (reps - 1))], 4),
        "p90_ms": round(ordered[int(0.9 * (reps - 1))], 4),
        "n": reps,
    }


def random_packed(shape, k: int, n: int, bias: bool, generator) -> PackedExperts:
    """Random codes with a scale that keeps activations near unit variance."""
    e = shape["E"]
    q = torch.randint(-2**31, 2**31 - 1, (e, k // 8, n), generator=generator, dtype=torch.int32)
    u = torch.randint(-2**15, 2**15 - 1, (e, k // 64, n), generator=generator, dtype=torch.int16)
    d = (torch.rand((e, n), generator=generator) * 0.5 + 0.75) * (0.022 / k ** 0.5)
    bc = torch.randn((e, n), generator=generator) * 1e-3 if bias else None
    return PackedExperts(q.to(DEV), u.to(DEV), d.to(DEV), signs_for_k(k, device=DEV),
                         None if bc is None else bc.to(DEV))


def weight_bytes(stack: PackedExperts) -> int:
    return sum(t.numel() * t.element_size() for t in (stack.q_pack, stack.u_pack, stack.d_row))


def vllm_runner(shape, quant_config):
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory

    with torch.device(DEV):
        return FusedMoEFactory(
            num_experts=shape["E"], top_k=shape["topk"], hidden_size=shape["H"], intermediate_size=shape["I"],
            params_dtype=torch.bfloat16, prefix="model.layers.0.mlp.experts",
            activation=shape["activation"], quant_config=quant_config,
        )


def packed_runner(shape, w13: PackedExperts, w2: PackedExperts, bias: bool):
    """A real vLLM MoERunner using HFInt4G16MoEMethod, filled through its loader."""
    from hf_int4_g16_xpu.format import FORMAT_VERSION_V2
    from hf_int4_g16_xpu.vllm_plugin import HFInt4G16Config

    module = MoEModule(shape["E"], shape["H"], shape["I"], {
        proj: ExpertProjection(had_block_for_k(shape["I"] if proj == "down_proj" else shape["H"]), bias)
        for proj in ("gate_proj", "up_proj", "down_proj")
    })
    config = HFInt4G16Config(format_version=FORMAT_VERSION_V2, moe_modules={"layers.0.mlp.experts": module})
    runner = vllm_runner(shape, config)
    layer = runner.routed_experts
    inter = shape["I"]
    roles = ("qweight", "scales", "scale_super") + (("bias_corr",) if bias else ())
    for group, stack, shards in (("w13", w13, ("w1", "w3")), ("w2", w2, ("w2",))):
        tensors = dict(zip(("qweight", "scales", "scale_super", "bias_corr"),
                           (stack.q_pack, stack.u_pack, stack.d_row, stack.bias_corr)))
        for role in roles:
            parameter = getattr(layer, f"{group}_{role}")
            for expert in range(shape["E"]):
                for index, shard in enumerate(shards):
                    piece = tensors[role][expert]
                    if group == "w13":
                        piece = piece.narrow(-1, index * inter, inter)
                    parameter.weight_loader(parameter, piece.contiguous(), "bench", shard_id=shard, expert_id=expert)
    layer.quant_method.process_weights_after_loading(layer)
    return runner


def runner_call(runner, x, logits, config):
    from vllm.forward_context import set_forward_context

    def call():
        with set_forward_context(None, config), torch.inference_mode():
            return runner(x.clone(), logits)

    return call


def stock_bf16_runner(shape):
    runner = vllm_runner(shape, None)
    layer = runner.routed_experts
    for name, parameter in layer.named_parameters():
        k = shape["H"] if name.startswith("w13") else shape["I"]
        parameter.data.normal_(0.0, k ** -0.5)
    with torch.device(DEV):
        layer.quant_method.process_weights_after_loading(layer)
    return runner


def stage_breakdown(x, weights, ids, w13, w2, activation, reps) -> dict:
    flat = ids.reshape(-1).to(torch.int32)
    slots = ids.shape[1]
    act = MOE_ACTIVATIONS[activation]
    gate_up = moe_pair_gemm(x, flat, w13, slots, torch.bfloat16)
    gate, up = gate_up.chunk(2, dim=-1)
    inner = (act(gate.float()) * up.float()).bfloat16()
    y = moe_pair_gemm(inner, flat, w2, 1, torch.float32)
    sync()
    return {
        "w13_gemm_ms": timed(lambda: moe_pair_gemm(x, flat, w13, slots, torch.bfloat16), reps)["median_ms"],
        "activation_ms": timed(lambda: (act(gate.float()) * up.float()).bfloat16(), reps)["median_ms"],
        "w2_gemm_fp32_ms": timed(lambda: moe_pair_gemm(inner, flat, w2, 1, torch.float32), reps)["median_ms"],
        "weighted_sum_ms": timed(
            lambda: (y.reshape(x.shape[0], slots, -1) * weights.float()[:, :, None]).sum(1).bfloat16(), reps
        )["median_ms"],
    }


def graph_replay(call, reps) -> dict:
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        call()
    return timed(graph.replay, reps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hwmon", default=None)
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args()

    torch.xpu.memory.set_per_process_memory_fraction(0.55)
    torch.use_deterministic_algorithms(True, warn_only=True)
    sensors = Sensors(args.hwmon)

    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from vllm.v1.worker.workspace import init_workspace_manager

    rendezvous = tempfile.mkdtemp()
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("xpu"))):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                     distributed_init_method=f"file://{rendezvous}/init")
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device(DEV))

    root = Path(__file__).resolve().parent
    receipt = {
        "experiment": "Exp 17 follow-up: hf-int4-g16-xpu 0.2.0 MoE layer latency",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.xpu.get_device_name(0),
        "torch": torch.__version__,
        "image": "vllm/vllm-openai-xpu:v0.31.0",
        "vram_fraction_cap": 0.55,
        "power_cap_w": (sensors.read("power1_cap") or 0) / 1e6 or None,
        "kernel_metadata": json.loads((root / "hf_int4_g16_xpu/native/build/build_metadata.json").read_text()),
        "method": "synchronized single calls, median over n after 3 warm-up calls; random packed weights, "
                  "random router logits; eager unless the key says graph_replay",
        "notes": args.note,
        "shapes": {},
    }

    for name, shape in SHAPES.items():
        e, topk, h, inter, activation = (shape[k] for k in ("E", "topk", "H", "I", "activation"))
        generator = torch.Generator().manual_seed(20261125)
        mac_per_token = topk * (h * 2 * inter + inter * h)
        entry = {**shape, "had_block_gate_up": had_block_for_k(h), "had_block_down": had_block_for_k(inter),
                 "gmac_per_token": mac_per_token / 1e9, "batches": {}}
        w13 = random_packed(shape, h, 2 * inter, False, generator)
        w2 = random_packed(shape, inter, h, False, generator)
        biased = [PackedExperts(s.q_pack, s.u_pack, s.d_row, s.signs,
                                torch.randn((e, s.n), generator=generator).to(DEV) * 1e-3) for s in (w13, w2)]
        per_expert_bytes = (weight_bytes(w13) + weight_bytes(w2)) / e
        entry["packed_weight_mib_per_layer"] = round((weight_bytes(w13) + weight_bytes(w2)) / 2**20, 1)
        entry["bf16_weight_mib_per_layer"] = round(e * 3 * h * inter * 2 / 2**20, 1)

        config =VllmConfig(device_config=DeviceConfig("xpu"))
        with set_current_vllm_config(config):
            ours = packed_runner(shape, w13, w2, False)
        stock_config = VllmConfig(device_config=DeviceConfig("xpu"))
        with set_current_vllm_config(stock_config):
            try:
                stock, stock_error = stock_bf16_runner(shape), None
            except Exception as error:  # recorded, never hidden
                stock, stock_error = None, f"{type(error).__name__}: {error}"
        entry["stock_bf16_error"] = stock_error

        for m in BATCHES:
            torch.manual_seed(m)
            x = torch.randn((m, h), generator=torch.Generator().manual_seed(m + 1)).bfloat16().to(DEV)
            logits = torch.randn((m, e), generator=torch.Generator().manual_seed(m + 2)).bfloat16().to(DEV)
            probabilities = logits.float().softmax(-1)
            weights, ids = probabilities.topk(topk, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            temperature = sensors.cool()
            reps = REPS[m]
            call = lambda: hf_int4_g16_moe(x, weights, ids, w13, w2, activation, backend="native")
            energy_before = sensors.read("energy1_input")
            started = time.perf_counter()
            row = {"start_temp_c": temperature, "experts_ms": timed(call, reps)}
            elapsed, energy_after = time.perf_counter() - started, sensors.read("energy1_input")
            if energy_before is not None and energy_after is not None:
                row["mean_card_power_w_during_loop"] = round((energy_after - energy_before) / 1e6 / elapsed, 1)
            row["experts_with_expert_bias_ms"] = timed(
                lambda: hf_int4_g16_moe(x, weights, ids, *biased, activation, backend="native"), reps)
            row["experts_graph_replay_ms"] = graph_replay(call, reps)
            row["stages"] = stage_breakdown(x, weights, ids, w13, w2, activation, reps)
            median_s = row["experts_ms"]["median_ms"] / 1e3
            row["tokens_per_s_this_layer"] = round(m / median_s, 1)
            row["effective_gmac_per_s"] = round(mac_per_token * m / median_s / 1e9, 1)
            # Each (token, slot) pair re-reads its expert's packed weights.
            row["packed_weight_gib_read_per_call"] = round(m * topk * per_expert_bytes / 2**30, 4)
            row["weight_read_gb_per_s_if_uncached"] = round(m * topk * per_expert_bytes / median_s / 1e9, 1)
            row["distinct_experts_routed"] = int(ids.unique().numel())
            with set_current_vllm_config(config):
                row["runner_ms"] = timed(runner_call(ours, x, logits, config), reps)
            if stock is not None:
                with set_current_vllm_config(stock_config):
                    try:
                        row["stock_bf16_runner_ms"] = timed(runner_call(stock, x, logits, stock_config), reps)
                    except Exception as error:
                        row["stock_bf16_runner_error"] = f"{type(error).__name__}: {error}"
            reference = hf_int4_g16_moe(x[:min(m, 8)], weights[:min(m, 8)], ids[:min(m, 8)], w13, w2,
                                        activation, backend="triton")
            native = call()[:min(m, 8)]
            sync()
            row["native_vs_triton_rel_l2_first8"] = float(
                (native.float() - reference.float()).norm() / reference.float().norm())
            row["triton_experts_ms"] = timed(
                lambda: hf_int4_g16_moe(x, weights, ids, w13, w2, activation, backend="triton"),
                TRITON_REPS[m], warmup=1)
            row["end_temp_c"] = sensors.temp_c()
            entry["batches"][str(m)] = row
            print(name, f"M={m}", "experts", row["experts_ms"]["median_ms"], "ms; replay",
                  row["experts_graph_replay_ms"]["median_ms"], "ms; runner", row["runner_ms"]["median_ms"],
                  "ms; stock bf16", row.get("stock_bf16_runner_ms", {}).get("median_ms"),
                  "ms; triton", row["triton_experts_ms"]["median_ms"], "ms", flush=True)
        entry["peak_device_gib"] = round(torch.xpu.max_memory_allocated() / 2**30, 3)
        receipt["shapes"][name] = entry
        ours = stock = w13 = w2 = biased = None  # release this shape's weights
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
