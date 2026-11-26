"""Bounded-memory streaming activation calibration for HF-INT4-G16 BiasCorr.

The full model is never instantiated in memory.  A meta-device model is used only
for architecture/masking metadata; each decoder layer is materialized, loaded
from safetensors, executed, and destroyed before moving to the next layer.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .format import HAD_SEED, rotate_had_noperm, signs_for_k

KNOWN_LAYER_TYPES = {"full_attention", "linear_attention", "sliding_attention"}


@dataclass
class CalibrationData:
    rows: list[list[int]]
    token_sha256: str
    info: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)


def _sha_rows(rows: list[list[int]]) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int32).tobytes()).hexdigest()


def _ngrams(ids: list[int], n: int = 6) -> set[tuple[int, ...]]:
    return {tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1)}


def _overlap_count(a: list[int], b: list[int], n: int = 6) -> int:
    return len(_ngrams(a, n) & _ngrams(b, n))


def _find_disjoint_window(ids: list[int], forbidden: list[int], width: int, label: str) -> tuple[list[int], int, int]:
    """Find the first deterministic width-token window with zero 6-gram overlap."""
    if len(ids) < width:
        raise ValueError(f"{label} has only {len(ids)} tokens, need {width}")
    # WikiText train/test are disjoint by construction, but natural language can
    # still share short exact phrases.  This deterministic scan makes the chosen
    # provenance explicit instead of silently accepting a contaminated window.
    max_start = len(ids) - width
    for start in range(0, max_start + 1, width):
        window = ids[start:start + width]
        overlap = _overlap_count(window, forbidden, 6)
        if overlap == 0:
            return window, start, 0
    # Fall back to a token-by-token scan before failing closed.  The scan is
    # still deterministic and bounded by the corpus length.
    forbidden_set = _ngrams(forbidden, 6)
    for start in range(max_start + 1):
        window = ids[start:start + width]
        if not (_ngrams(window, 6) & forbidden_set):
            return window, start, 0
    raise ValueError(f"no zero-overlap {width}-token {label} calibration window found")


def _encode(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


# How many trailing content tokens a row may give up so all chat rows have one length.
_CHAT_MAX_TRIM = 64


def require_chat_template(tok) -> None:
    if not getattr(tok, "chat_template", None):
        raise ValueError("--cal-chat-template needs a tokenizer with a chat_template, and this one has none")


def _chat_rows(tok, windows: list[list[int]]) -> tuple[list[list[int]], list[str]]:
    """Render each text window as one user turn with a generation prompt.

    Returns ``(rows, chunks)``.  The template writes its own special tokens, so
    nothing is added on encode and a template BOS lands on position 0, which
    calibration already excludes.  Calibration rows must be rectangular and the
    re-encoded lengths can differ, so content is trimmed from the end (never the
    template) to the longest length every window can reach.
    """
    require_chat_template(tok)
    trims = _CHAT_MAX_TRIM + 1 if len(windows) > 1 else 1
    options = []
    for window in windows:
        by_length: dict[int, tuple[list[int], str]] = {}
        for trim in range(min(trims, len(window))):
            chunk = tok.decode(window[:len(window) - trim], clean_up_tokenization_spaces=False)
            text = tok.apply_chat_template(
                [{"role": "user", "content": chunk}], tokenize=False, add_generation_prompt=True
            )
            ids = _encode(tok, text)
            by_length.setdefault(len(ids), (ids, chunk))
        options.append(by_length)
    shared = set.intersection(*(set(by_length) for by_length in options))
    if not shared:
        raise ValueError(
            f"--cal-chat-template cannot give all calibration rows one length within {_CHAT_MAX_TRIM} trimmed tokens"
        )
    length = max(shared)
    return [o[length][0] for o in options], [o[length][1] for o in options]


def _chat_info(tok, rows: list[list[int]], chunks: list[str]) -> dict[str, Any]:
    return {
        "cal_chat_template_sha256": hashlib.sha256(
            json.dumps(tok.chat_template, sort_keys=True).encode()
        ).hexdigest(),
        "cal_row_tokens": len(rows[0]),
        "cal_content_tokens": [len(_encode(tok, chunk)) for chunk in chunks],
    }


def load_mixed_train_calibration(tok, tokens_per_split: int = 1024, chat_template: bool = False) -> CalibrationData:
    """Load canonical WikiText-2 *train* + disjoint Python-stdlib calibration.

    The prose window is explicitly checked against the WikiText-2 test head and
    the code window against ``difflib.py``.  Both exact token 6-gram overlap
    counts must be zero; otherwise conversion fails closed.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("--cal-mode mixed_train requires the datasets package") from e

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_ids = _encode(tok, "\n".join(ds["train"]["text"]))
    test_ids = _encode(tok, "\n".join(ds["test"]["text"]))
    # The prose oracle is explicitly the WT2 test *head* requested by the audit.
    test_head = test_ids[:max(tokens_per_split, 4096)]
    prose, prose_start, wt_overlap = _find_disjoint_window(
        train_ids, test_head, tokens_per_split, "WikiText-2 train"
    )

    # Deterministic stdlib code corpus.  difflib.py itself is deliberately not a
    # source; it is the disjointness oracle requested for provenance.
    import sysconfig
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])
    code_files = [
        stdlib_dir / "os.py",
        stdlib_dir / "pathlib.py",
        stdlib_dir / "enum.py",
        stdlib_dir / "json" / "__init__.py",
        stdlib_dir / "argparse.py",
    ]
    code_text_parts = []
    used_files = []
    for p in code_files:
        if p.name == "difflib.py":
            continue
        if p.is_file():
            code_text_parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            used_files.append(str(p))
    code_text = "\n\n".join(code_text_parts)
    diff_path = stdlib_dir / "difflib.py"
    diff_text = diff_path.read_text(encoding="utf-8", errors="ignore")
    code_ids = _encode(tok, code_text)
    diff_ids = _encode(tok, diff_text)
    code, code_start, code_overlap = _find_disjoint_window(
        code_ids, diff_ids, tokens_per_split, "Python stdlib code"
    )

    # Count exact final-window overlaps once more and fail closed if invariant broke.
    wt_overlap = _overlap_count(prose, test_head, 6)
    code_overlap = _overlap_count(code, diff_ids, 6)
    if wt_overlap != 0 or code_overlap != 0:
        raise ValueError(f"calibration overlap invariant failed: wt2={wt_overlap}, difflib={code_overlap}")

    rows = [prose, code]
    chat_info: dict[str, Any] = {}
    if chat_template:
        rows, chunks = _chat_rows(tok, rows)
        # Template tokens are not corpus text; the invariant covers the content.
        wt_overlap = _overlap_count(_encode(tok, chunks[0]), test_head, 6)
        code_overlap = _overlap_count(_encode(tok, chunks[1]), diff_ids, 6)
        if wt_overlap != 0 or code_overlap != 0:
            raise ValueError(f"chat calibration overlap invariant failed: wt2={wt_overlap}, difflib={code_overlap}")
        chat_info = _chat_info(tok, rows, chunks)
    sha = _sha_rows(rows)
    info = {
        "mode": "mixed_train",
        "tokens_per_split": tokens_per_split,
        "rows": 2,
        "prose_source": "Salesforce/wikitext wikitext-2-raw-v1 train",
        "prose_token_start": prose_start,
        "code_sources": used_files,
        "code_token_start": code_start,
        "wt2_test_head_6gram_overlap": wt_overlap,
        "difflib_py_6gram_overlap": code_overlap,
        "cal_token_sha256": sha,
        "cal_chat_template": chat_template,
        **chat_info,
    }
    return CalibrationData(rows, sha, info)


def sysconfig_path() -> str:  # never called; keeps the deterministic path expression obvious
    import sysconfig
    return sysconfig.get_paths()["stdlib"]


def load_corpus_calibration(tok, corpus_path: Path, max_tokens: int, chat_template: bool = False) -> CalibrationData:
    if corpus_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        tbl = pq.read_table(corpus_path)
        col = "text" if "text" in tbl.column_names else tbl.column_names[0]
        text = "\n".join(str(x) for x in tbl[col].to_pylist() if x and str(x).strip())
    else:
        text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    ids = _encode(tok, text)[:max_tokens]
    if len(ids) < 16:
        raise ValueError("bias-correction corpus produced fewer than 16 tokens")
    rows = [ids]
    chat_info: dict[str, Any] = {}
    if chat_template:
        rows, chunks = _chat_rows(tok, rows)
        chat_info = _chat_info(tok, rows, chunks)
    sha = _sha_rows(rows)
    return CalibrationData(rows, sha, {
        "mode": "corpus_file",
        "corpus_path": str(corpus_path),
        "tokens": len(rows[0]),
        "cal_token_sha256": sha,
        "cal_chat_template": chat_template,
        **chat_info,
    })


def _device_sync(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _empty_cache(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def _peak_rss_gib() -> float:
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class _ShardReader:
    def __init__(self, root: Path, wmap: dict[str, str]):
        self.root = Path(root)
        self.wmap = wmap
        self.cache: dict[str, Any] = {}

    def get(self, key: str) -> torch.Tensor:
        fn = self.wmap[key]
        if fn not in self.cache:
            if len(self.cache) >= 2:
                self.cache.clear()
            self.cache[fn] = safe_open(str(self.root / fn), framework="pt", device="cpu")
        return self.cache[fn].get_tensor(key)

    def close(self):
        self.cache.clear()


def _validate_raw_layer_types(payload: dict[str, Any]) -> None:
    """Reject unsupported masks before transformers config validation/coercion."""
    layer_types = payload.get("layer_types")
    if layer_types is None:
        return
    unknown = sorted(set(layer_types) - KNOWN_LAYER_TYPES)
    if unknown:
        raise ValueError(f"streaming calibration unsupported layer_types: {unknown}")


def _pin_sdpa_attention(cfg) -> None:
    """Pin the exact attention backend before any masks or layers are built."""
    cfg._attn_implementation = "sdpa"
    if getattr(cfg, "_attn_implementation", None) != "sdpa":
        raise ValueError(
            f"streaming calibration requires SDPA attention, got {getattr(cfg, '_attn_implementation', None)!r}"
        )


def _is_tower_key(key: str) -> bool:
    """Vision/audio encoders also have ``layers.N`` namespaces; never calibrate on them."""
    return "vision" in key or "audio" in key


def _load_architecture(model_dir: Path, wmap: dict[str, str]):
    """Return (text_config, LayerCls, base-module, source embedding key)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    _validate_raw_layer_types(raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw)
    arch = (raw.get("architectures") or [""])[0]
    if arch == "Qwen3_5ForConditionalGeneration" or isinstance(raw.get("text_config"), dict):
        import transformers

        text_type = raw["text_config"].get("model_type") or "qwen3_5_text"
        wrapped = {
            "qwen3_5_text": ("Qwen3_5TextConfig", "Qwen3_5ForCausalLM"),
            "qwen3_5_moe_text": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM"),
            "gemma4_text": ("Gemma4TextConfig", "Gemma4ForCausalLM"),
        }
        if text_type not in wrapped:
            raise ValueError(f"streaming calibration unsupported text model_type: {text_type}")
        config_cls, model_cls = (getattr(transformers, name) for name in wrapped[text_type])
        cfg = config_cls(**raw["text_config"])
        _pin_sdpa_attention(cfg)
        with torch.device("meta"):
            dummy = model_cls(cfg)
    else:
        cfg = AutoConfig.from_pretrained(str(model_dir))
        _pin_sdpa_attention(cfg)
        with torch.device("meta"):
            dummy = AutoModelForCausalLM.from_config(cfg)
        cfg = getattr(dummy, "config", cfg)

    if not hasattr(dummy, "model") or not hasattr(dummy.model, "layers"):
        del dummy
        raise ValueError("streaming calibration supports causal models with model.layers only")
    base = dummy.model
    layer_cls = type(base.layers[0])
    layer0 = base.layers[0]
    first_param = next(iter(layer0.state_dict().keys()))
    suffix = f"layers.0.{first_param}"
    matches = sorted(k for k in wmap if k.endswith(suffix) and not _is_tower_key(k))
    if not matches:
        del dummy
        raise ValueError(f"cannot map first decoder-layer parameter {suffix} into checkpoint index")
    src_key = matches[0]
    layer0_src_prefix = src_key[: -len(first_param)]

    # Resolve embedding by namespace.  Qwen3.5 conditional checkpoints use
    # model.language_model.embed_tokens; standard causal models use model.embed_tokens.
    emb_candidates = [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ]
    emb_key = next((k for k in emb_candidates if k in wmap), None)
    if emb_key is None:
        emb_key = next((k for k in sorted(wmap) if k.endswith("embed_tokens.weight")), None)
    if emb_key is None:
        del dummy
        raise ValueError("cannot locate embed_tokens.weight for streaming calibration")
    return cfg, layer_cls, base, emb_key, layer0_src_prefix, dummy


def _forward_kwargs(layer: nn.Module, pos_emb, mask, pos_ids, shared_kv_states: dict) -> dict[str, Any]:
    sig = inspect.signature(layer.forward)
    kw: dict[str, Any] = {}
    if "position_embeddings" in sig.parameters:
        kw["position_embeddings"] = pos_emb
    if "attention_mask" in sig.parameters:
        kw["attention_mask"] = mask
    if "position_ids" in sig.parameters:
        kw["position_ids"] = pos_ids
    if "use_cache" in sig.parameters:
        kw["use_cache"] = False
    if "shared_kv_states" in sig.parameters:
        kw["shared_kv_states"] = shared_kv_states
    return kw


def _expert_stats_hook(stats: dict[str, torch.Tensor], batch: int, seq: int):
    """Routed pos0-excluded means for one ``*Experts`` module.

    ``mean_x[e]`` averages the hidden states routed to expert ``e``;
    ``mean_down[e]`` averages ``act(gate) * up`` from the original weights over
    the same tokens.  An expert with no routed token keeps zero means and a zero
    count, and the converter writes a zero correction for it.
    """
    def hook(module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        hidden, top_k_index = bound["hidden_states"], bound["top_k_index"]
        if hidden.ndim != 2 or hidden.shape[0] != batch * seq:
            raise RuntimeError(
                f"routed experts received {tuple(hidden.shape)}, expected [{batch * seq},H]; "
                "cannot exclude position zero"
            )
        gate_up_w, num_experts = module.gate_up_proj, module.gate_up_proj.shape[0]
        routed = torch.zeros((hidden.shape[0], num_experts), device=hidden.device)
        routed.scatter_(1, top_k_index.reshape(hidden.shape[0], -1), 1.0)
        routed[torch.arange(hidden.shape[0], device=hidden.device) % seq == 0] = 0.0
        counts = routed.sum(dim=0)
        mean_x = (routed.t() @ hidden.float()) / counts.clamp(min=1)[:, None]
        mean_down = torch.zeros((num_experts, gate_up_w.shape[1] // 2), device=hidden.device)
        for expert in torch.nonzero(counts)[:, 0].tolist():
            selected = torch.nonzero(routed[:, expert])[:, 0]
            gate, up = F.linear(hidden[selected], gate_up_w[expert]).chunk(2, dim=-1)
            mean_down[expert] = (module.act_fn(gate) * up).float().sum(dim=0) / counts[expert]
        stats.update(counts=counts.long().cpu(), mean_x=mean_x.cpu(), mean_down=mean_down.cpu())

    return hook


@torch.no_grad()
def stream_calibration_means(
    model_dir: str | Path,
    wmap: dict[str, str],
    calibration: CalibrationData,
    device: str | torch.device,
    seed: int = HAD_SEED,
    return_hidden: bool = False,
    expert_sources: list | None = None,
    expert_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """Stream the decoder and return pos0-excluded per-linear activation means.

    Returns ``means[full_checkpoint_linear_prefix] = {'mean_x', 'mean_rx'}`` as
    CPU FP32 tensors.  The only full-model state carried across layers is the
    small calibration hidden activation and these mean vectors.

    ``expert_sources`` (from ``experts_source.scan_experts``) lets MoE decoder
    layers be materialized from any supported expert layout.  When
    ``expert_stats`` is a dict, it is filled with routed means keyed by each
    source namespace (see ``_expert_stats_hook``).
    """
    model_dir = Path(model_dir)
    dev = torch.device(device)
    if dev.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("streaming calibration requested XPU but torch.xpu is unavailable")
    if dev.type == "xpu":
        # Force lazy device initialization before querying allocator statistics;
        # torch.xpu.is_available() alone does not always initialize the device.
        torch.empty(1, device=dev)
        torch.xpu.reset_peak_memory_stats(dev.index if dev.index is not None else 0)

    cfg, layer_cls, base, emb_key, layer0_prefix, dummy = _load_architecture(model_dir, wmap)
    rotary_cls = type(base.rotary_emb) if hasattr(base, "rotary_emb") else None
    qmod = inspect.getmodule(type(base))
    layer_types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * int(cfg.num_hidden_layers))
    unknown_layer_types = sorted(set(layer_types) - KNOWN_LAYER_TYPES)
    if unknown_layer_types:
        raise ValueError(f"streaming calibration unsupported layer_types: {unknown_layer_types}")
    if len(layer_types) < int(cfg.num_hidden_layers):
        raise ValueError("streaming calibration layer_types shorter than num_hidden_layers")
    if getattr(cfg, "hidden_size_per_layer_input", None):
        raise ValueError("streaming calibration does not support per-layer input embeddings")
    # Gemma-family embeddings are scaled by sqrt(H), rounded in the weight dtype.
    embed_scale = getattr(getattr(base, "embed_tokens", None), "scalar_embed_scale", None)
    del base, dummy
    gc.collect()

    reader = _ShardReader(model_dir, wmap)
    rows = torch.tensor(calibration.rows, dtype=torch.long, device=dev)
    B, S = rows.shape
    t0 = time.perf_counter()

    emb = reader.get(emb_key).to(device=dev, dtype=torch.bfloat16)
    hidden = F.embedding(rows, emb)
    if embed_scale is not None:
        hidden = hidden * torch.tensor(embed_scale, device=dev).to(hidden.dtype)
    del emb
    _empty_cache(dev)

    pos_ids = torch.arange(S, device=dev, dtype=torch.long).unsqueeze(0).expand(B, -1)
    rotary_pos_ids = pos_ids
    if str(getattr(cfg, "model_type", "")).startswith("qwen3_5"):
        # Qwen3.5 reuses the multimodal rope contract: positions are
        # (text, temporal, height, width).  TextModel strips slot zero for the
        # causal-mask helpers and passes slots 1:4 to rotary_emb.
        pos4 = torch.arange(S, device=dev, dtype=torch.long).view(1, 1, -1).expand(4, B, -1)
        pos_ids = pos4[0].contiguous()
        rotary_pos_ids = pos4[1:].contiguous()
    pos_emb = None
    if rotary_cls is not None:
        rotary = rotary_cls(config=cfg, device=dev)
        if "layer_type" in inspect.signature(rotary.forward).parameters:
            # Separate local/global RoPE per attention type (Gemma family).
            pos_emb = {lt: rotary(hidden, rotary_pos_ids, lt) for lt in sorted(set(layer_types))}
        else:
            pos_emb = rotary(hidden, rotary_pos_ids)
        del rotary

    mask_kwargs = {"config": cfg, "inputs_embeds": hidden, "attention_mask": None,
                   "past_key_values": None, "position_ids": pos_ids}
    masks: dict[str, Any] = {}
    if hasattr(qmod, "create_causal_mask"):
        masks["full_attention"] = qmod.create_causal_mask(**mask_kwargs)
    else:
        masks["full_attention"] = None
    if hasattr(qmod, "create_recurrent_attention_mask"):
        masks["linear_attention"] = qmod.create_recurrent_attention_mask(**mask_kwargs)
    else:
        masks["linear_attention"] = masks["full_attention"]
    if "sliding_attention" in layer_types:
        if not hasattr(qmod, "create_sliding_window_causal_mask"):
            raise ValueError("streaming calibration cannot build a sliding_attention mask for this model")
        masks["sliding_attention"] = qmod.create_sliding_window_causal_mask(**mask_kwargs)

    means: dict[str, dict[str, torch.Tensor]] = {}
    linears_seen = 0
    shared_kv_states: dict = {}
    experts_by_name: dict[str, tuple[Any, str]] = {}
    for source in expert_sources or []:
        for fused_name in ("gate_up_proj", "down_proj"):
            experts_by_name[f"{source.namespace}.{fused_name}"] = (source, fused_name)
    sign_cache: dict[int, torch.Tensor] = {}

    for lidx in range(int(cfg.num_hidden_layers)):
        with torch.device("meta"):
            layer = layer_cls(cfg, lidx) if "layer_idx" in inspect.signature(layer_cls.__init__).parameters or len(inspect.signature(layer_cls.__init__).parameters) > 1 else layer_cls(cfg)
        layer = layer.to_empty(device="cpu")
        state_names = list(layer.state_dict().keys())
        if not state_names:
            raise ValueError(f"decoder layer {lidx} has no loadable state")
        first = state_names[0]
        suffix = f"layers.{lidx}.{first}"
        matches = sorted(k for k in wmap if k.endswith(suffix) and not _is_tower_key(k))
        if not matches:
            raise ValueError(f"cannot map decoder layer {lidx} parameter {suffix}")
        src_prefix = matches[0][: -len(first)]

        def read(key, index=None):
            tensor = reader.get(key)
            return tensor if index is None else tensor[index]

        def load_state(pname):
            key = src_prefix + pname
            if key not in wmap and key in experts_by_name:
                source, fused_name = experts_by_name[key]
                return source.read_fused(read, fused_name)
            return reader.get(key)

        sd = {pname: load_state(pname) for pname in state_names}
        layer.load_state_dict(sd, strict=True)
        del sd
        layer = layer.to(device=dev, dtype=torch.bfloat16).eval()

        linears = [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]
        stats: dict[str, dict[str, torch.Tensor | int]] = {}
        hooks = []

        def make_hook(nm: str, k: int):
            def hook(mod, inp):
                if not inp or not torch.is_tensor(inp[0]):
                    return
                x = inp[0].detach()
                if x.ndim == 2 and x.shape[0] == B * S:
                    x = x.reshape(B, S, x.shape[1])  # MoE routers flatten [B,S,H]
                if x.ndim == 3:
                    x = x[:, 1:, :] if x.shape[1] > 1 else x[:, :0, :]
                elif x.ndim == 2:
                    # Standard decoder linears are 3D.  A flattened fallback cannot
                    # identify position zero, so fail closed rather than silently
                    # contaminating the pos0-excluded mean.
                    raise RuntimeError(f"activation for {nm} is rank-2; cannot enforce pos0 exclusion")
                else:
                    return
                if x.shape[-1] != k or x.shape[0] == 0 or x.shape[1] == 0:
                    return
                xf = x.float()
                if k not in sign_cache:
                    sign_cache[k] = signs_for_k(k, seed, dev)
                xr = rotate_had_noperm(xf, sign_cache[k])
                ent = stats.setdefault(nm, {
                    "sum_x": torch.zeros(k, device=dev, dtype=torch.float32),
                    "sum_rx": torch.zeros(k, device=dev, dtype=torch.float32),
                    "count": 0,
                })
                ent["sum_x"] += xf.reshape(-1, k).sum(dim=0, dtype=torch.float32)
                ent["sum_rx"] += xr.reshape(-1, k).sum(dim=0, dtype=torch.float32)
                ent["count"] += int(x.reshape(-1, k).shape[0])
            return hook

        for n, m in linears:
            # A width off the 64 grid has no rotation and is never quantized,
            # for example inside an expert bank that stays in BF16.
            if m.in_features % 64 == 0:
                hooks.append(m.register_forward_pre_hook(make_hook(n, m.in_features)))
        layer_expert_stats: dict[str, dict[str, torch.Tensor]] = {}
        if expert_stats is not None:
            for n, m in layer.named_modules():
                if type(m).__name__.endswith("Experts") and hasattr(m, "gate_up_proj"):
                    layer_expert_stats[n] = {}
                    hooks.append(m.register_forward_pre_hook(
                        _expert_stats_hook(layer_expert_stats[n], B, S), with_kwargs=True
                    ))

        ltype = layer_types[lidx] if lidx < len(layer_types) else "full_attention"
        mask = masks.get(ltype, masks["full_attention"])
        layer_pos_emb = pos_emb[ltype] if isinstance(pos_emb, dict) else pos_emb
        out = layer(hidden, **_forward_kwargs(layer, layer_pos_emb, mask, pos_ids, shared_kv_states))
        hidden = out[0] if isinstance(out, tuple) else out
        for h in hooks:
            h.remove()

        for n, m in linears:
            ent = stats.get(n)
            if ent is None or int(ent["count"]) == 0:
                raise ValueError(f"no pos0-excluded activation samples captured for layer {lidx}.{n}")
            c = float(ent["count"])
            means[src_prefix + n] = {
                "mean_x": (ent["sum_x"] / c).detach().cpu().float(),
                "mean_rx": (ent["sum_rx"] / c).detach().cpu().float(),
            }
            linears_seen += 1

        for n, captured in layer_expert_stats.items():
            if not captured:
                raise ValueError(f"routed experts {lidx}.{n} were never called during calibration")
            expert_stats[src_prefix + n] = captured

        del layer, linears, stats, hooks, out
        if lidx % 4 == 3 or lidx == int(cfg.num_hidden_layers) - 1:
            gc.collect()
            _empty_cache(dev)
            print(f"[biascorr-stream] layer {lidx + 1}/{cfg.num_hidden_layers} means={len(means)}", flush=True)

    reader.close()
    _device_sync(dev)
    peak_vram = torch.xpu.max_memory_allocated(dev.index if dev.index is not None else 0) / 1024**3 if dev.type == "xpu" else 0.0
    meta = {
        "layers": int(cfg.num_hidden_layers),
        "linears_seen": linears_seen,
        "batch_shape": [B, S],
        "pos0_excluded": True,
        "wall_s": round(time.perf_counter() - t0, 3),
        "peak_rss_gib": round(_peak_rss_gib(), 3),
        "peak_device_gib": round(peak_vram, 3),
        "embedding_key": emb_key,
        "layer0_source_prefix": layer0_prefix,
        "attn_implementation": getattr(cfg, "_attn_implementation", None),
    }
    if return_hidden:
        return means, meta, hidden.detach().cpu().clone()
    return means, meta
