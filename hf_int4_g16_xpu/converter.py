"""Streaming BF16/FP16 safetensors -> HF-INT4-G16 converter (dense and MoE).

Dense Had128-only checkpoints are written as ``hf_int4_g16/1``.  Any Had64
projection or packed routed expert makes the output ``hf_int4_g16/2``.

``--biascorr-scope``: ``none`` writes no correction, ``shared_only`` corrects
every dense linear (attention and shared MLPs) and no expert, ``all_experts``
also corrects each routed expert over the calibration tokens routed to it
(zero for an expert that received none).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .calibration import (
    load_corpus_calibration,
    load_mixed_train_calibration,
    require_chat_template,
    stream_calibration_means,
)
from .experts_source import require_packable, scan_experts
from .format import (
    EXPERT_PROJECTIONS,
    EXPERT_TENSOR_SUFFIXES,
    FORMAT_VERSION,
    FORMAT_VERSION_V2,
    GROUP_SIZE,
    HAD_BLOCK_RULE_V2,
    HAD_SEED,
    HADAMARD_BLOCK,
    ROTATION_V1,
    ROTATION_V2,
    had_block_for_k,
    quantize_pack_hf_int4,
    rotate_had_noperm,
    sign_sha256_for_k,
    signs_for_k,
    unpack_hf_int4_reference,
)

# Routers stay in the source dtype: "router" (Gemma4, Granite router.layer),
# ".gate." (Qwen/OLMoE/Granite routers) and "shared_expert_gate".
DEFAULT_SKIP = ("lm_head,embed_tokens,mtp,visual,vision_tower,audio_tower,multi_modal_projector,"
                "router,.gate.,shared_expert_gate")
BIASCORR_SCOPES = ("none", "shared_only", "all_experts")


def norm_key(k: str) -> str:
    import re
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


def safetensors_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
    }.get(dtype, str(dtype).replace("torch.", "").upper())


class ShardWriter:
    def __init__(self, out_dir: Path, max_bytes: int = 2 * 1024**3):
        self.out_dir = Path(out_dir)
        self.max_bytes = int(max_bytes)
        self.tensors: dict[str, torch.Tensor] = {}
        self.weight_map: dict[str, str] = {}
        self.nbytes = 0
        self.shard_idx = 0
        self.shard_names: list[str] = []
        self.total_bytes = 0
        self.expected_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}

    def _name(self) -> str:
        return f"model-{self.shard_idx + 1:05d}.safetensors"

    def add(self, key: str, tensor: torch.Tensor):
        t = tensor.detach().cpu().contiguous()
        nb = t.numel() * t.element_size()
        if self.tensors and self.nbytes + nb > self.max_bytes:
            self.flush()
        if nb > self.max_bytes:
            raise ValueError(f"tensor {key} is larger than one shard: {nb / 2**30:.2f} GiB")
        name = self._name()
        self.tensors[key] = t
        self.expected_meta[key] = (tuple(t.shape), t.dtype)
        self.weight_map[key] = name
        self.nbytes += nb
        self.total_bytes += nb

    def flush(self):
        if not self.tensors:
            return
        name = self._name()
        save_file(self.tensors, str(self.out_dir / name), metadata={"format": "pt"})
        self.shard_names.append(name)
        self.shard_idx += 1
        self.tensors.clear()
        self.nbytes = 0

    def finish(self):
        self.flush()
        return self.weight_map, self.shard_names, self.total_bytes


def load_index(model_dir: Path) -> tuple[dict[str, str], list[Path]]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
        return wmap, sorted({model_dir / v for v in wmap.values()})
    shards = sorted(model_dir.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights found in {model_dir}")
    wmap = {}
    for sh in shards:
        with safe_open(str(sh), framework="pt", device="cpu") as f:
            for k in f.keys():
                wmap[k] = sh.name
    return wmap, shards


def should_skip(key: str, patterns: list[str]) -> bool:
    return any(p and p in key for p in patterns)


def decoder_linear_eligibility(key: str, shape, dtype, patterns: list[str]) -> tuple[bool, str]:
    """Single dry-run/conversion admission predicate for decoder LINEAR weights.

    Only floating rank-2 ``.weight`` tensors below a decoder ``.layers.N``
    namespace are eligible.  This prevents embeddings, norms, convolutions,
    nondecoder matrices and already-integer tensors from being interpreted as
    quantizable linears.
    """
    if not key.endswith(".weight") or len(shape) != 2 or should_skip(key, patterns):
        return False, "skip-pattern-or-not-linear"
    if ".layers." not in key:
        return False, "outside-decoder-layers"
    dtype_txt = str(dtype).lower()
    if not any(x in dtype_txt for x in ("bf16", "float16", "float32", "f16", "f32")):
        return False, f"unsupported-source-dtype-{dtype}"
    n, k = int(shape[0]), int(shape[1])
    if n <= 0 or k <= 0:
        return False, "rank-or-shape"
    if k % 64:
        return False, f"K={k} not divisible by 128 (Had128) or 64 (Had64)"
    return True, "eligible"


def _nagel_bias(weight: torch.Tensor, packed: dict, mean_x: torch.Tensor, mean_rx: torch.Tensor,
                device: str) -> torch.Tensor:
    """FP32 original mean response minus quantized rotated mean rotated response,
    i.e. each in the basis in which that weight consumes its input."""
    w_fp32 = weight.to(device=device, dtype=torch.float32)
    wi = unpack_hf_int4_reference(packed["q_pack"].to(device), packed["u_pack"].to(device)).float()
    what = wi * packed["d_row"].to(device)[:, None]
    bc = (w_fp32 @ mean_x.to(device)) - (what @ mean_rx.to(device))
    if not torch.isfinite(bc).all().item():
        raise ValueError("non-finite bias_corr")
    return bc.cpu().float()


def resolve_biascorr_scope(scope: str | None, biascorr: bool, corpus: str | None) -> str:
    requested = bool(biascorr or corpus)
    if scope is None:
        return "shared_only" if requested else "none"
    if scope not in BIASCORR_SCOPES:
        raise ValueError(f"biascorr_scope must be one of {BIASCORR_SCOPES}, got {scope!r}")
    if scope == "none" and requested:
        raise ValueError("--biascorr-scope none contradicts --biascorr / --biascorr-corpus")
    return scope


def _metadata_reader(src: Path, wmap: dict[str, str]):
    """``(read, meta)`` over the source shards, keeping at most two handles open."""
    handles: dict = {}

    def handle(key):
        fn = wmap[key]
        if fn not in handles:
            # Keys are sorted and normally shard-local.
            if len(handles) >= 2:
                handles.clear()
            handles[fn] = safe_open(str(src / fn), framework="pt", device="cpu")
        return handles[fn]

    def read(key, index=None):
        if index is None:
            return handle(key).get_tensor(key)
        return handle(key).get_slice(key)[index]

    def meta(key):
        sl = handle(key).get_slice(key)
        return list(sl.get_shape()), str(sl.get_dtype())

    return read, meta, handles


def _source_config(src: Path) -> dict:
    path = src / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def dry_run_layer_mapping(input_dir: str | Path, skip_patterns: str = DEFAULT_SKIP) -> dict:
    """Metadata-only decoder-linear mapping audit for large checkpoints.

    This is intentionally cheap enough to run against the full 27B index.  It
    verifies that every candidate is a decoder ``model*.layers.N`` rank-2 linear
    and records the exclusion classes required by the deployment contract.
    """
    src = Path(input_dir)
    wmap, _ = load_index(src)
    patterns = [p.strip() for p in skip_patterns.split(",") if p.strip()]
    candidates = []
    # "k_not_multiple_128" counts widths no Hadamard block fits (not 128 and not 64).
    excluded = {"skip_pattern_or_not_linear": 0, "k_not_multiple_128": 0, "rank_or_shape": 0}
    examples = {"skipped": [], "quantized": []}
    _, meta_for, shard_handles = _metadata_reader(src, wmap)

    try:
        bf16_expert_keys: set[str] = set()
        experts = scan_experts(
            sorted(wmap), meta_for, _source_config(src),
            skipped=lambda name: should_skip(name, patterns), opted_out=bf16_expert_keys,
        )
        require_packable(experts)
        expert_keys = {k for source in experts for k in source.keys}
        for key in sorted(set(wmap) - expert_keys):
            shape, dtype = meta_for(key)
            if key in bf16_expert_keys:
                eligible, reason = False, "expert-bank-opted-out"
            else:
                eligible, reason = decoder_linear_eligibility(key, shape, dtype, patterns)
            if not eligible:
                if reason.startswith("K="):
                    excluded["k_not_multiple_128"] += 1
                elif reason in ("outside-decoder-layers", "rank-or-shape") or reason.startswith("unsupported-source-dtype"):
                    excluded["rank_or_shape"] += 1
                else:
                    excluded["skip_pattern_or_not_linear"] += 1
                if len(examples["skipped"]) < 12:
                    examples["skipped"].append(key)
                continue
            n, k = shape
            candidates.append({"key": key, "shape": [n, k], "dtype": str(dtype)})
            if len(examples["quantized"]) < 12:
                examples["quantized"].append(key)
    finally:
        shard_handles.clear()

    layers = sorted({int(k["key"].split(".layers.", 1)[1].split(".", 1)[0]) for k in candidates})
    return {
        "input": str(src),
        "decoder_linear_candidates": len(candidates),
        "layers_covered": len(layers),
        "first_layer": layers[0] if layers else None,
        "last_layer": layers[-1] if layers else None,
        "excluded": excluded,
        "examples": examples,
        "expert_namespaces": [
            {"namespace": x.namespace, "num_experts": x.num_experts, "hidden_size": x.hidden_size,
             "intermediate_size": x.intermediate_size,
             "layout": "per_expert" if x.per_expert_keys is not None else "fused_3d",
             "had_block": {proj: had_block_for_k(x.shape(proj)[1]) for proj in EXPERT_PROJECTIONS}}
            for x in experts
        ],
        "keys": [c["key"] for c in candidates],
    }


def convert(
    input_dir: str | Path,
    output_dir: str | Path,
    device: str = "xpu:0",
    skip_patterns: str = DEFAULT_SKIP,
    max_shard_bytes: int = 2 * 1024**3,
    biascorr_corpus: str | None = None,
    biascorr_tokens: int = 2048,
    biascorr: bool = False,
    cal_mode: str = "auto",
    seed: int = HAD_SEED,
    quantize_fn=quantize_pack_hf_int4,
    biascorr_scope: str | None = None,
    cal_chat_template: bool = False,
) -> dict:
    scope = resolve_biascorr_scope(biascorr_scope, biascorr, biascorr_corpus)
    if cal_chat_template and scope == "none":
        raise ValueError("--cal-chat-template does nothing without BiasCorr calibration (--biascorr-scope is none)")
    src = Path(input_dir)
    final_dst = Path(output_dir)
    if src.resolve() == final_dst.resolve():
        raise ValueError("input and output directories must differ")
    if final_dst.exists() and any(final_dst.iterdir()):
        raise ValueError(
            f"output directory {final_dst} is non-empty; refusing to mix conversion generations"
        )
    stage = final_dst.parent / (final_dst.name + ".hf-int4-stage")
    if stage.exists():
        raise ValueError(f"stale conversion staging directory exists: {stage}")
    tok = None
    if scope != "none":
        # Loaded before anything is written, so a missing chat template leaves no staging directory.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(src))
        if cal_chat_template:
            require_chat_template(tok)
    if final_dst.exists():
        final_dst.rmdir()  # empty only, checked above
    stage.mkdir(parents=True)
    dst = stage
    t0 = time.time()
    patterns = [p.strip() for p in skip_patterns.split(",") if p.strip()]
    wmap, _ = load_index(src)
    all_keys = sorted(wmap)

    # First pass over metadata only: decide quantization set.
    read, meta_for, shard_handles = _metadata_reader(src, wmap)
    config = _source_config(src)
    bf16_expert_keys: set[str] = set()
    expert_sources = scan_experts(
        all_keys, meta_for, config,
        skipped=lambda name: should_skip(name, patterns), opted_out=bf16_expert_keys,
    )
    require_packable(expert_sources)
    expert_keys = {k for source in expert_sources for k in source.keys}

    quantizable = []
    skipped = {}
    for key in all_keys:
        if key in expert_keys:
            continue
        if key in bf16_expert_keys:
            # An opted-out bank stays whole: none of it is a dense linear.
            skipped[key] = "expert-bank-opted-out"
            continue
        shape, dtype = meta_for(key)
        eligible, reason = decoder_linear_eligibility(key, shape, dtype, patterns)
        if not eligible:
            skipped[key] = reason
            continue
        n, k = shape
        quantizable.append((key, tuple(shape)))
    shard_handles.clear()

    means = {}
    expert_stats: dict[str, dict[str, torch.Tensor]] = {}
    calibration_info = None
    streaming_meta = None
    if scope != "none":
        print(f"[convert] streaming pos0-excluded calibration for {len(quantizable)} linears "
              f"(biascorr_scope={scope})", flush=True)
        if cal_mode == "mixed_train" or (cal_mode == "auto" and not biascorr_corpus):
            cal = load_mixed_train_calibration(
                tok, tokens_per_split=max(16, biascorr_tokens // 2), chat_template=cal_chat_template
            )
        elif biascorr_corpus:
            cal = load_corpus_calibration(
                tok, Path(biascorr_corpus), max_tokens=biascorr_tokens, chat_template=cal_chat_template
            )
        else:
            raise ValueError("--biascorr requires --cal-mode mixed_train or --biascorr-corpus")
        # The forward needs every expert, including namespaces kept in BF16.
        all_sources = scan_experts(all_keys, meta_for, config)
        shard_handles.clear()
        means, streaming_meta = stream_calibration_means(
            src, wmap, cal, device, seed=seed, expert_sources=all_sources,
            expert_stats=expert_stats if scope == "all_experts" else None,
        )
        calibration_info = dict(cal.info)
        calibration_info["streaming"] = streaming_meta
        missing_cal = [k[: -len(".weight")] for k, _ in quantizable if k[: -len(".weight")] not in means]
        if missing_cal:
            raise ValueError(f"streaming calibration did not capture quantized decoder linears: {missing_cal[:8]}")
        if scope == "all_experts":
            uncaptured = [x.namespace for x in expert_sources if x.namespace not in expert_stats]
            if uncaptured:
                raise ValueError(f"streaming calibration did not capture routed experts: {uncaptured[:8]}")

    quant_shapes = dict(quantizable)
    writer = ShardWriter(dst, max_shard_bytes)
    manifest = {}
    moe_manifest = {}
    expert_routing = {}
    expert_ks: set[int] = set()
    stats = {"quantized_linears": 0, "copied_tensors": 0, "skipped_linears": 0, "biascorr_linears": 0,
             "had64_linears": 0, "expert_namespaces": 0, "quantized_expert_matrices": 0,
             "had64_expert_matrices": 0, "biascorr_expert_matrices": 0}

    def release_device_memory():
        gc.collect()
        if device.startswith("xpu"):
            torch.xpu.empty_cache()

    try:
        for key in all_keys:
            if key in expert_keys:
                continue
            if key not in quant_shapes:
                writer.add(key, read(key))
                stats["copied_tensors"] += 1
                continue
            n, k = quant_shapes[key]
            block = had_block_for_k(k)
            print(f"[convert] quantize {key} [{n},{k}] Had{block}", flush=True)
            w = read(key)
            packed = quantize_fn(w, seed=seed, device=device)
            prefix = key[: -len(".weight")]
            writer.add(prefix + ".q_pack", packed["q_pack"])
            writer.add(prefix + ".u_pack", packed["u_pack"])
            writer.add(prefix + ".d_row", packed["d_row"])
            entry = {
                "shape": [n, k],
                "group_size": GROUP_SIZE,
                "hadamard_block": block,
                "layout": "xe2_dpas_native_v1",
                "q_pack_shape": [k // 8, n],
                "u_pack_shape": [k // 64, n],
                "bias_corr": False,
            }
            if means:
                pair = means[prefix]
                writer.add(prefix + ".bias_corr", _nagel_bias(w, packed, pair["mean_x"], pair["mean_rx"], device))
                entry["bias_corr"] = True
                stats["biascorr_linears"] += 1
            if norm_key(prefix) in manifest:
                raise ValueError(f"two checkpoint linears normalize to the same module name: {norm_key(prefix)}")
            manifest[norm_key(prefix)] = entry
            stats["quantized_linears"] += 1
            stats["had64_linears"] += block == 64
            del w, packed
            release_device_memory()

        for source in expert_sources:
            namespace = norm_key(source.namespace)
            if namespace in moe_manifest:
                raise ValueError(f"two expert namespaces normalize to the same module name: {namespace}")
            routing = expert_stats.get(source.namespace)
            print(f"[convert] quantize experts {source.namespace} E={source.num_experts} "
                  f"H={source.hidden_size} I={source.intermediate_size}", flush=True)
            for expert in range(source.num_experts):
                for proj in EXPERT_PROJECTIONS:
                    n, k = source.shape(proj)
                    w = source.read(read, expert, proj).contiguous()
                    packed = quantize_fn(w, seed=seed, device=device)
                    prefix = f"{source.namespace}.{expert}.{proj}"
                    for role, tensor in packed.items():
                        writer.add(f"{prefix}.{EXPERT_TENSOR_SUFFIXES[role]}", tensor)
                    if routing is not None:
                        bc = torch.zeros(n, dtype=torch.float32)
                        if int(routing["counts"][expert]) > 0:
                            mean_x = routing["mean_down" if proj == "down_proj" else "mean_x"][expert].to(device)
                            mean_rx = rotate_had_noperm(mean_x[None], signs_for_k(k, seed, device))[0]
                            bc = _nagel_bias(w, packed, mean_x, mean_rx, device)
                        writer.add(f"{prefix}.{EXPERT_TENSOR_SUFFIXES['bias_corr']}", bc)
                        stats["biascorr_expert_matrices"] += 1
                    stats["quantized_expert_matrices"] += 1
                    stats["had64_expert_matrices"] += had_block_for_k(k) == 64
                    del w, packed
            moe_manifest[namespace] = {
                "num_experts": source.num_experts,
                "hidden_size": source.hidden_size,
                "intermediate_size": source.intermediate_size,
                "projections": {
                    proj: {"had_block": had_block_for_k(source.shape(proj)[1]), "bias_corr": routing is not None}
                    for proj in EXPERT_PROJECTIONS
                },
            }
            if routing is not None:
                counts = routing["counts"]
                expert_routing[namespace] = {
                    "routed_tokens_min": int(counts.min()), "routed_tokens_max": int(counts.max()),
                    "experts_without_tokens": int((counts == 0).sum()),
                }
            expert_ks.update((source.hidden_size, source.intermediate_size))
            stats["expert_namespaces"] += 1
            release_device_memory()
    finally:
        shard_handles.clear()

    weight_map_out, shard_names, total_bytes = writer.finish()

    unique_ks = sorted({int(shape[1]) for shape in quant_shapes.values()} | expert_ks)
    version_2 = bool(moe_manifest) or stats["had64_linears"] > 0
    if version_2:
        for entry in manifest.values():
            entry["had_block"] = entry.pop("hadamard_block")
    sign_sha256_by_k = {str(k): sign_sha256_for_k(k, seed) for k in unique_ks}

    block_fields = (
        {"had_block_rule": HAD_BLOCK_RULE_V2}
        if version_2 else {"hadamard_block": HADAMARD_BLOCK}
    )
    qconfig = {
        "quant_method": "hf_int4_g16",
        "format_version": FORMAT_VERSION_V2 if version_2 else FORMAT_VERSION,
        "group_size": GROUP_SIZE,
        **block_fields,
        "rotation": ROTATION_V2 if version_2 else ROTATION_V1,
        "had_seed": int(seed),
        "sign_generation": "torch.Generator(cpu).manual_seed(had_seed * 100003 + K); randint(0,2)*2-1",
        "sign_sha256_by_k": sign_sha256_by_k,
        "layout": "xe2_dpas_native_v1",
        "layout_spec": {
            "q_pack": "int32 bit container [K//8,N]; byte b low=q(base+b)+8 high=q(base+4+b)+8, base=8*r",
            "u_pack": "int16 bit container [K//64,N]; nibble j=(u-1) for group 4*r+j",
            "d_row": "float32 [N]",
            "bias_corr": "optional float32 [N], fp32 epilogue, Nagel pos0-excluded",
        },
        "quantized_modules": manifest,
    }
    if version_2:
        qconfig["biascorr_scope"] = scope
        if moe_manifest:
            qconfig["moe_modules"] = moe_manifest
            qconfig["layout_spec"]["expert_tensors"] = (
                "<ns>.<e>.<gate_proj|up_proj|down_proj>.<qweight=q_pack|scales=u_pack|"
                "scale_super=d_row|bias_corr>"
            )
    if calibration_info:
        qconfig["calibration"] = calibration_info
        qconfig["cal_token_sha256"] = calibration_info.get("cal_token_sha256")

    # Copy small tokenizer/config assets into the private staging directory,
    # then patch config.json.  Nothing is visible at the requested output path
    # until every shard has been reloaded and verified.
    for p in src.iterdir():
        if not p.is_file() or p.suffix == ".safetensors" or p.name == "model.safetensors.index.json":
            continue
        if p.stat().st_size > 64 * 1024 * 1024:
            continue
        shutil.copy2(p, dst / p.name)
    cfg_path = dst / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = {}
    cfg["quantization_config"] = qconfig
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "quantization_config.json").write_text(json.dumps(qconfig, indent=2) + "\n")

    # Fail-closed shard reload verification: every indexed key must reopen with
    # exactly the shape/dtype intended at write time.  Packed scales must be
    # finite positive FP32 and corrections finite FP32 before publication.
    verified = 0
    for shard in shard_names:
        with safe_open(str(dst / shard), framework="pt", device="cpu") as f:
            present = set(f.keys())
            expected = {k for k, v in weight_map_out.items() if v == shard}
            if present != expected:
                raise ValueError(f"shard {shard} key mismatch after reload")
            for key in sorted(expected):
                sl = f.get_slice(key)
                shape = tuple(int(x) for x in sl.get_shape())
                dtype = sl.get_dtype()
                want_shape, want_dtype = writer.expected_meta[key]
                if shape != tuple(want_shape) or dtype != safetensors_dtype_name(want_dtype):
                    raise ValueError(
                        f"reloaded tensor {key} metadata mismatch: {shape}/{dtype} != {tuple(want_shape)}/{want_dtype}"
                    )
                if key.endswith((".d_row", "." + EXPERT_TENSOR_SUFFIXES["d_row"])):
                    vals = f.get_tensor(key)
                    if vals.dtype != torch.float32 or not torch.isfinite(vals).all() or not (vals > 0).all():
                        raise ValueError(f"d_row tensor {key} is not finite positive fp32")
                elif key.endswith(".bias_corr"):
                    vals = f.get_tensor(key)
                    if vals.dtype != torch.float32 or not torch.isfinite(vals).all():
                        raise ValueError(f"bias_corr tensor {key} is not finite fp32")
                verified += 1

    # Publish the index only after verification.  The report is part of the
    # staged artifact; os.replace then atomically exposes one complete generation.
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map_out,
    }, indent=2) + "\n")

    report = {
        "input": str(src), "output": str(final_dst), "device": device,
        "elapsed_s": round(time.time() - t0, 3),
        "stats": stats,
        "skipped": skipped,
        "quantized_modules": len(manifest),
        "shards": shard_names,
        "reload_verified_tensors": verified,
        "total_bytes": total_bytes,
        "calibration": calibration_info,
        "cal_chat_template": bool(cal_chat_template),
        "format_version": qconfig["format_version"],
        "biascorr_scope": scope,
        "expert_routing_calibration": expert_routing,
        "atomic_publish": "verified-staging-os.replace",
    }
    (dst / "hf_int4_g16_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    os.replace(dst, final_dst)
    print(f"[convert] done: {stats}, shards={len(shard_names)}, verified={verified}", flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--skip-patterns", default=DEFAULT_SKIP)
    ap.add_argument("--max-shard-gib", type=float, default=2.0)
    ap.add_argument("--biascorr", action="store_true", help="Enable streaming pos0-excluded BiasCorr calibration")
    ap.add_argument("--biascorr-scope", choices=list(BIASCORR_SCOPES), default=None,
                    help="none | shared_only (dense linears only; the default with --biascorr) | all_experts")
    ap.add_argument("--biascorr-corpus", default=None, help="Optional custom corpus file/parquet (legacy corpus mode)")
    ap.add_argument("--biascorr-tokens", type=int, default=2048)
    ap.add_argument("--cal-mode", choices=["auto", "mixed_train", "corpus_file"], default="auto",
                    help="mixed_train: 1024 WikiText-2 train + 1024 disjoint Python-stdlib tokens")
    ap.add_argument("--cal-chat-template", action="store_true",
                    help="Wrap each calibration text block as a user turn with the tokenizer's chat_template "
                         "and generation prompt; fails if the tokenizer has none")
    ap.add_argument("--dry-run-mapping", action="store_true", help="Only audit decoder linear mapping; write nothing")
    ap.add_argument("--had-seed", type=int, default=HAD_SEED)
    args = ap.parse_args(argv)
    if args.dry_run_mapping:
        report = dry_run_layer_mapping(args.input_dir, args.skip_patterns)
        print(json.dumps({k: v for k, v in report.items() if k != "keys"}, indent=2))
        return report
    convert(
        args.input_dir, args.output_dir, device=args.device,
        skip_patterns=args.skip_patterns,
        max_shard_bytes=int(args.max_shard_gib * 1024**3),
        biascorr_corpus=args.biascorr_corpus,
        biascorr_tokens=args.biascorr_tokens,
        biascorr=args.biascorr,
        cal_mode=args.cal_mode,
        seed=args.had_seed,
        biascorr_scope=args.biascorr_scope,
        cal_chat_template=args.cal_chat_template,
    )


if __name__ == "__main__":
    main()
