"""Streaming BF16/FP16 safetensors -> HF-INT4-G16 converter."""
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
    stream_calibration_means,
)
from .format import (
    FORMAT_VERSION,
    GROUP_SIZE,
    HAD_SEED,
    HADAMARD_BLOCK,
    quantize_pack_hf_int4,
    rotate_had128_noperm,
    sign_sha256_for_k,
    signs_for_k,
    unpack_hf_int4_reference,
)

DEFAULT_SKIP = "lm_head,embed_tokens,mtp,visual,vision_tower,multi_modal_projector,router,experts"


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
    if k % HADAMARD_BLOCK:
        return False, f"K={k} not divisible by 128"
    return True, "eligible"


def read_corpus(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        col = "text" if "text" in tbl.column_names else tbl.column_names[0]
        return [str(x) for x in tbl[col].to_pylist()]
    return [path.read_text(errors="ignore")]


@torch.no_grad()
def calibrate_rotated_means(
    model_dir: Path,
    module_names: list[str],
    corpus_path: Path,
    device: str,
    max_tokens: int = 2048,
    max_rows: int = 65536,
) -> dict[str, torch.Tensor]:
    """Measure pos0-excluded E[rotated x] with forward pre-hooks.

    This path is intentionally generic HuggingFace code.  It is appropriate for
    small verification models and documents the exact math for Wave-2 large
    conversions; large-model conversion can supply precomputed means in a future
    streaming implementation without changing the checkpoint format.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    texts = read_corpus(corpus_path)
    ids: list[int] = []
    for text in texts:
        ids.extend(tok.encode(text, add_special_tokens=True))
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    if len(ids) < 16:
        raise ValueError("bias-correction corpus produced fewer than 16 tokens")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    module_map = dict(model.named_modules())
    missing = [m for m in module_names if m not in module_map]
    if missing:
        raise ValueError(f"calibration target modules not found: {missing[:8]}")
    sums = {m: None for m in module_names}
    counts = {m: 0 for m in module_names}
    hooks = []

    def make_hook(name):
        def hook(mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            k = x.shape[-1]
            if k % HADAMARD_BLOCK:
                return
            xx = x.reshape(-1, x.shape[-2] if x.ndim >= 2 else 1, k) if x.ndim > 2 else x
            # Exclude position zero of every sequence.  For flattened batches the
            # second dimension is sequence position in standard decoder modules.
            if xx.ndim == 3:
                xx = xx[:, 1:, :]
            else:
                xx = xx[1:, :]
            xx = xx.reshape(-1, k)
            if xx.shape[0] == 0:
                return
            signs = signs_for_k(k, HAD_SEED, x.device)
            xr = rotate_had128_noperm(xx, signs).sum(dim=0, dtype=torch.float32)
            if sums[name] is None:
                sums[name] = torch.zeros_like(xr)
            sums[name] += xr
            counts[name] += xx.shape[0]
        return hook

    for name in module_names:
        hooks.append(module_map[name].register_forward_pre_hook(make_hook(name)))
    try:
        rows = 0
        pos = 0
        seq_len = min(2048, max(32, len(ids)))
        while pos + 2 < len(ids) and rows < max_rows:
            chunk = ids[pos:pos + seq_len]
            if len(chunk) < 2:
                break
            batch = torch.tensor([chunk], dtype=torch.long, device=device)
            model(batch, use_cache=False)
            rows += batch.numel() - 1
            pos += seq_len
    finally:
        for h in hooks:
            h.remove()
        del model
        gc.collect()
        if device.startswith("xpu"):
            torch.xpu.empty_cache()

    out = {}
    for name in module_names:
        if sums[name] is None or counts[name] == 0:
            raise ValueError(f"no activation samples captured for {name}")
        out[name] = (sums[name] / counts[name]).cpu().float()
    return out


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
    excluded = {"skip_pattern_or_not_linear": 0, "k_not_multiple_128": 0, "rank_or_shape": 0}
    examples = {"skipped": [], "quantized": []}
    shard_handles = {}

    def meta_for(key):
        fn = wmap[key]
        if fn not in shard_handles:
            if len(shard_handles) >= 2:
                shard_handles.clear()
            shard_handles[fn] = safe_open(str(src / fn), framework="pt", device="cpu")
        sl = shard_handles[fn].get_slice(key)
        return sl.get_shape(), sl.get_dtype()

    try:
        for key in sorted(wmap):
            shape, dtype = meta_for(key)
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
) -> dict:
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
    if final_dst.exists():
        final_dst.rmdir()  # empty only, checked above
    stage.mkdir(parents=True)
    dst = stage
    t0 = time.time()
    patterns = [p.strip() for p in skip_patterns.split(",") if p.strip()]
    wmap, _ = load_index(src)
    all_keys = sorted(wmap)

    # First pass over metadata only: decide quantization set.
    tensor_meta = {}
    shard_handles = {}
    def meta_for(key):
        fn = wmap[key]
        if fn not in shard_handles:
            # Keep a small cache; keys are sorted and normally shard-local.
            if len(shard_handles) >= 2:
                shard_handles.clear()
            shard_handles[fn] = safe_open(str(src / fn), framework="pt", device="cpu")
        f = shard_handles[fn]
        sl = f.get_slice(key)
        return sl.get_shape(), sl.get_dtype()

    quantizable = []
    skipped = {}
    for key in all_keys:
        shape, dtype = meta_for(key)
        eligible, reason = decoder_linear_eligibility(key, shape, dtype, patterns)
        if not eligible:
            skipped[key] = reason
            continue
        n, k = shape
        quantizable.append((key, tuple(shape)))
    for f in shard_handles.values():
        del f
    shard_handles.clear()

    means = {}
    calibration_info = None
    streaming_meta = None
    if biascorr or biascorr_corpus:
        from transformers import AutoTokenizer
        print(f"[convert] streaming pos0-excluded calibration for {len(quantizable)} linears", flush=True)
        tok = AutoTokenizer.from_pretrained(str(src))
        if cal_mode == "mixed_train" or (cal_mode == "auto" and not biascorr_corpus):
            cal = load_mixed_train_calibration(tok, tokens_per_split=max(16, biascorr_tokens // 2))
        elif biascorr_corpus:
            cal = load_corpus_calibration(tok, Path(biascorr_corpus), max_tokens=biascorr_tokens)
        else:
            raise ValueError("--biascorr requires --cal-mode mixed_train or --biascorr-corpus")
        means, streaming_meta = stream_calibration_means(src, wmap, cal, device, seed=seed)
        calibration_info = dict(cal.info)
        calibration_info["streaming"] = streaming_meta
        missing_cal = [k[: -len(".weight")] for k, _ in quantizable if k[: -len(".weight")] not in means]
        if missing_cal:
            raise ValueError(f"streaming calibration did not capture quantized decoder linears: {missing_cal[:8]}")

    quant_shapes = dict(quantizable)
    writer = ShardWriter(dst, max_shard_bytes)
    manifest = {}
    shard_handles = {}
    stats = {"quantized_linears": 0, "copied_tensors": 0, "skipped_linears": 0, "biascorr_linears": 0}

    def get_tensor(key):
        fn = wmap[key]
        if fn not in shard_handles:
            if len(shard_handles) >= 2:
                shard_handles.clear()
            shard_handles[fn] = safe_open(str(src / fn), framework="pt", device="cpu")
        return shard_handles[fn].get_tensor(key)

    try:
        for key in all_keys:
            if key not in quant_shapes:
                writer.add(key, get_tensor(key))
                stats["copied_tensors"] += 1
                continue
            n, k = quant_shapes[key]
            print(f"[convert] quantize {key} [{n},{k}]", flush=True)
            w = get_tensor(key)
            packed = quantize_fn(w, seed=seed, device=device)
            prefix = key[: -len(".weight")]
            writer.add(prefix + ".q_pack", packed["q_pack"])
            writer.add(prefix + ".u_pack", packed["u_pack"])
            writer.add(prefix + ".d_row", packed["d_row"])
            entry = {
                "shape": [n, k],
                "group_size": GROUP_SIZE,
                "hadamard_block": HADAMARD_BLOCK,
                "layout": "xe2_dpas_native_v1",
                "q_pack_shape": [k // 8, n],
                "u_pack_shape": [k // 64, n],
                "bias_corr": False,
            }
            if means:
                pair = means[prefix]
                mean_x = pair["mean_x"].to(device)
                mean_rx = pair["mean_rx"].to(device)
                # Explicit Nagel correction in FP32, in the basis in which each
                # linear actually consumes x: reference original-weight mean
                # response minus quantized rotated-weight mean rotated response.
                w_fp32 = w.to(device=device, dtype=torch.float32)
                wi = unpack_hf_int4_reference(packed["q_pack"].to(device), packed["u_pack"].to(device)).float()
                what = wi * packed["d_row"].to(device)[:, None]
                bc = (w_fp32 @ mean_x) - (what @ mean_rx)
                if not torch.isfinite(bc).all().item():
                    raise ValueError(f"non-finite bias_corr for {prefix}")
                writer.add(prefix + ".bias_corr", bc.cpu().float())
                entry["bias_corr"] = True
                stats["biascorr_linears"] += 1
            manifest[norm_key(prefix)] = entry
            stats["quantized_linears"] += 1
            del w, packed
            gc.collect()
            if device.startswith("xpu"):
                torch.xpu.empty_cache()
    finally:
        shard_handles.clear()

    weight_map_out, shard_names, total_bytes = writer.finish()

    unique_ks = sorted({int(shape[1]) for shape in quant_shapes.values()})
    sign_sha256_by_k = {str(k): sign_sha256_for_k(k, seed) for k in unique_ks}

    qconfig = {
        "quant_method": "hf_int4_g16",
        "format_version": FORMAT_VERSION,
        "group_size": GROUP_SIZE,
        "hadamard_block": HADAMARD_BLOCK,
        "rotation": "Had128NoPerm",
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
                if key.endswith(".d_row"):
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
    ap.add_argument("--biascorr-corpus", default=None, help="Optional custom corpus file/parquet (legacy corpus mode)")
    ap.add_argument("--biascorr-tokens", type=int, default=2048)
    ap.add_argument("--cal-mode", choices=["auto", "mixed_train", "corpus_file"], default="auto",
                    help="mixed_train: 1024 WikiText-2 train + 1024 disjoint Python-stdlib tokens")
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
    )


if __name__ == "__main__":
    main()
