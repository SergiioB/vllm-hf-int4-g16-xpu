"""vLLM out-of-tree quantization plugin for Had128NoPerm + HF-INT4-G16.

The plugin is intentionally fail-closed: a module is quantized iff the
checkpoint manifest contains its canonical packed tensors; malformed metadata,
partial fused members, or orphan bias-correction keys are hard errors.
"""
from __future__ import annotations

import json
import os
import re
import struct
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .format import FORMAT_VERSION, GROUP_SIZE, HAD_SEED, HADAMARD_BLOCK, sign_sha256_for_k, signs_for_k
from .ops import hf_int4_g16_linear

logger = init_logger("vllm.hf_int4_g16_xpu")

FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}
SKIP_PARTS = {"visual", "vision_tower", "multi_modal_projector"}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    v = os.environ.get(name, default).strip().lower()
    if v not in choices:
        raise ValueError(f"hf_int4_g16: {name}={v!r}, expected one of {sorted(choices)}")
    return v


def biascorr_mode() -> str:
    return _env_choice("HF_INT4_G16_BIASCORR_MODE", "default", {"default", "off", "neg_zero"})


def linear_mode() -> str:
    return _env_choice("HF_INT4_G16_MODE", "w4a16", {"auto", "w4a16", "w4a8"})


def _norm_key(k: str) -> str:
    """Canonical key shared by checkpoint tensors and vLLM module prefixes."""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


PACKED_SUFFIXES = (".q_pack", ".u_pack", ".d_row", ".bias_corr")
CORE_PACKED_SUFFIXES = (".q_pack", ".u_pack", ".d_row")


def _is_expert_key(key: str) -> bool:
    parts = set(key.split("."))
    return "experts" in parts or "expert" in parts


def _packed_base(name: str) -> tuple[str, str] | tuple[None, None]:
    for suffix in PACKED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None, None


def _read_safetensors_header(path: str) -> dict[str, Any]:
    """Read only a safetensors JSON header; tensor payload bytes are never touched."""
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                raise ValueError("truncated header length")
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len == 0 or header_len > 256 * 1024 * 1024:
                raise ValueError(f"implausible header length {header_len}")
            raw = f.read(header_len)
            if len(raw) != header_len:
                raise ValueError("truncated JSON header")
            obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"hf_int4_g16: unreadable safetensors header {path}: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"hf_int4_g16: safetensors header {path} is not an object")
    return obj


def _header_shape_dtype(entry: Any) -> tuple[list[int], str]:
    if not isinstance(entry, dict):
        return [], ""
    shape = entry.get("shape")
    dtype = entry.get("dtype")
    if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
        shape = []
    return shape, dtype if isinstance(dtype, str) else ""


@register_quantization_config("hf_int4_g16")
class HFInt4G16Config(QuantizationConfig):
    def __init__(
        self,
        group_size: int = GROUP_SIZE,
        hadamard_block: int = HADAMARD_BLOCK,
        had_seed: int = HAD_SEED,
        rotation: str = "Had128NoPerm",
        layout: str = "xe2_dpas_native_v1",
        format_version: str = FORMAT_VERSION,
        quantized_modules: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__()
        self.group_size = int(group_size)
        self.hadamard_block = int(hadamard_block)
        self.had_seed = int(had_seed)
        self.rotation = rotation
        self.layout = layout
        self.format_version = format_version
        self.quantized_modules = quantized_modules or {}

    def __repr__(self):
        return (f"HFInt4G16Config(group_size={self.group_size}, hadamard_block={self.hadamard_block}, "
                f"had_seed={self.had_seed}, modules={len(self.quantized_modules)})")

    def get_name(self):
        return "hf_int4_g16"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HFInt4G16Config":
        q = config.get("hf_int4_g16", config)
        return cls(
            group_size=q.get("group_size", GROUP_SIZE),
            hadamard_block=q.get("hadamard_block", HADAMARD_BLOCK),
            had_seed=q.get("had_seed", HAD_SEED),
            rotation=q.get("rotation", "Had128NoPerm"),
            layout=q.get("layout", "xe2_dpas_native_v1"),
            format_version=q.get("format_version", FORMAT_VERSION),
            quantized_modules=q.get("quantized_modules") or q.get("modules") or {},
        )

    def validate_format(self):
        if self.group_size != GROUP_SIZE:
            raise ValueError(f"hf_int4_g16: unsupported group_size {self.group_size}")
        if self.hadamard_block != HADAMARD_BLOCK or self.rotation != "Had128NoPerm":
            raise ValueError(f"hf_int4_g16: unsupported rotation {self.rotation}/{self.hadamard_block}")
        if self.layout != "xe2_dpas_native_v1" or self.format_version != FORMAT_VERSION:
            raise ValueError(f"hf_int4_g16: unsupported layout/version {self.layout}/{self.format_version}")

    def _model_file(self, model_name, fname, revision):
        path = os.path.join(model_name, fname)
        if os.path.isfile(path):
            return path
        if os.path.isdir(model_name):
            raise FileNotFoundError(f"hf_int4_g16: required checkpoint file {fname} is missing in {model_name}")
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(model_name, fname, revision=revision)
        except Exception as e:
            raise FileNotFoundError(
                f"hf_int4_g16: required checkpoint file {fname} could not be read for {model_name}: {e}"
            ) from e

    @staticmethod
    def _read_json_file(path: str, label: str) -> dict[str, Any]:
        try:
            with open(path) as f:
                obj = json.load(f)
        except Exception as e:
            raise ValueError(f"hf_int4_g16: unreadable {label} at {path}: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"hf_int4_g16: {label} at {path} is not a JSON object")
        return obj

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        """Discover and strictly admit quantized modules from checkpoint metadata.

        Admission is fail-closed and evidence-based: the index must reference
        existing shards, one logical packed module must come from one raw
        checkpoint namespace, the manifest and index must agree exactly, and
        every packed key's dtype/shape is verified in the referenced
        safetensors header (an O(KB) read, with no tensor payload load).
        """
        del hf_config
        self.validate_format()
        idx_path = self._model_file(model_name, "model.safetensors.index.json", revision)
        qc_path = self._model_file(model_name, "quantization_config.json", revision)

        qc = self._read_json_file(qc_path, "quantization_config.json")
        q = qc.get("hf_int4_g16", qc)
        if not isinstance(q, dict):
            raise ValueError("hf_int4_g16: quantization config payload is not an object")
        stored: dict[str, dict[str, Any]] = {}
        collisions = []
        manifest_modules = q.get("quantized_modules") or q.get("modules") or {}
        if not isinstance(manifest_modules, dict):
            raise ValueError("hf_int4_g16: quantized_modules manifest is not an object")
        for key, v in manifest_modules.items():
            if not isinstance(v, dict):
                raise ValueError(f"hf_int4_g16: manifest metadata for {key} is not an object")
            nk = _norm_key(key)
            if nk in stored:
                collisions.append(nk)
            if _is_expert_key(key) or _is_expert_key(nk):
                raise ValueError(f"hf_int4_g16: packed expert/MoE module is unsupported: {key}")
            stored[nk] = dict(v)
        if collisions:
            raise ValueError(f"hf_int4_g16: normalized manifest key collisions: {sorted(collisions)[:8]}")

        # Fail closed on sidecar format disagreement before any weights load.
        for field, got, want in [
            ("group_size", q.get("group_size", self.group_size), self.group_size),
            ("hadamard_block", q.get("hadamard_block", self.hadamard_block), self.hadamard_block),
            ("had_seed", q.get("had_seed", self.had_seed), self.had_seed),
            ("rotation", q.get("rotation", self.rotation), self.rotation),
            ("layout", q.get("layout", self.layout), self.layout),
            ("format_version", q.get("format_version", self.format_version), self.format_version),
        ]:
            if got != want:
                raise ValueError(f"hf_int4_g16: quantization_config {field} mismatch: {got} != {want}")

        sign_digests = q.get("sign_sha256_by_k")
        if sign_digests is not None:
            if not isinstance(sign_digests, dict) or not sign_digests:
                raise ValueError("hf_int4_g16: sign_sha256_by_k must be a non-empty object when present")
            for k_txt, digest in sign_digests.items():
                try:
                    k_int = int(k_txt)
                except Exception as e:
                    raise ValueError(f"hf_int4_g16: sign digest key {k_txt!r} is not an integer K") from e
                want_digest = sign_sha256_for_k(k_int, self.had_seed)
                if digest != want_digest:
                    raise ValueError(
                        f"hf_int4_g16: sign digest mismatch for K={k_int}: {digest} != {want_digest}"
                    )

        idx = self._read_json_file(idx_path, "model.safetensors.index.json")
        wmap = idx.get("weight_map")
        if not isinstance(wmap, dict) or not wmap:
            raise ValueError("hf_int4_g16: checkpoint index has no weight_map")
        if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in wmap.items()):
            raise ValueError("hf_int4_g16: checkpoint index weight_map entries must be non-empty strings")

        # Every shard referenced by the index must exist and have a parseable
        # header.  This is deliberate even for non-packed tensors: a missing
        # shard is checkpoint corruption, not something quantization admission
        # may ignore because the missing key is outside this format.
        shard_headers: dict[str, dict[str, Any]] = {}
        missing_shards = []
        for shard in sorted(set(wmap.values())):
            try:
                shard_path = self._model_file(model_name, shard, revision)
                shard_headers[shard] = _read_safetensors_header(shard_path)
            except (FileNotFoundError, ValueError) as e:
                missing_shards.append((shard, str(e)))
        if missing_shards:
            shard, err = missing_shards[0]
            raise FileNotFoundError(f"hf_int4_g16: checkpoint shard {shard} missing/unreadable: {err}")

        # Two-way packed inventory check: every packed key present in any shard
        # header must appear in weight_map pointing to that exact shard so an
        # unindexed .bias_corr or packed tensor cannot silently hide in a shard.
        unindexed_packed = []
        for shard, hdr in sorted(shard_headers.items()):
            for hname in hdr:
                if hname == "__metadata__":
                    continue
                hbase, _ = _packed_base(hname)
                if hbase is not None and wmap.get(hname) != shard:
                    unindexed_packed.append((hname, shard))
        if unindexed_packed:
            raise ValueError(
                f"hf_int4_g16: shard header contains unindexed packed tensors: {unindexed_packed[:8]}"
            )

        packed: dict[str, dict[str, Any]] = {}
        expert_packed = []
        norm_collisions = []
        for name, shard in wmap.items():
            base, suffix = _packed_base(name)
            if base is None:
                continue
            if _is_expert_key(base):
                expert_packed.append(name)
                continue
            nb = _norm_key(base)
            ent = packed.setdefault(nb, {"raw_bases": set(), "suffixes": {}})
            ent["raw_bases"].add(base)
            if suffix in ent["suffixes"]:
                norm_collisions.append((nb, suffix, ent["suffixes"][suffix][0], name))
            ent["suffixes"][suffix] = (name, shard)
        if expert_packed:
            raise ValueError(f"hf_int4_g16: packed expert/MoE tensors are unsupported: {expert_packed[:8]}")
        if norm_collisions:
            raise ValueError(f"hf_int4_g16: normalized checkpoint key collisions: {norm_collisions[:4]}")

        incomplete = []
        orphans = []
        undeclared = []
        cross_namespace = []
        complete_bases = []
        for base, ent in sorted(packed.items()):
            if len(ent["raw_bases"]) != 1:
                cross_namespace.append((base, sorted(ent["raw_bases"])))
                continue
            suffixes = ent["suffixes"]
            has_q = ".q_pack" in suffixes
            missing = [sfx[1:] for sfx in CORE_PACKED_SUFFIXES if sfx not in suffixes]
            if has_q and missing:
                incomplete.append((base, missing))
                continue
            if not has_q:
                orphans.append((base, sorted(sfx[1:] for sfx in suffixes)))
                continue
            if base not in stored:
                undeclared.append(base)
                continue
            complete_bases.append(base)
        if cross_namespace:
            raise ValueError(f"hf_int4_g16: cross-namespace packed suffix mixing: {cross_namespace[:4]}")
        if incomplete:
            raise ValueError(f"hf_int4_g16: incomplete packed tensor inventory: {incomplete[:8]}")
        if orphans:
            raise ValueError(f"hf_int4_g16: orphan u_pack/d_row/bias_corr keys: {orphans[:8]}")
        if undeclared:
            raise ValueError(f"hf_int4_g16: packed modules missing from quantization manifest: {undeclared[:8]}")

        header_dtypes = {
            ".q_pack": "I32",
            ".u_pack": "I16",
            ".d_row": "F32",
            ".bias_corr": "F32",
        }
        found: dict[str, dict[str, Any]] = {}
        header_errors = []
        actual_ks = set()
        for base in complete_bases:
            meta = dict(stored[base])
            shape = meta.get("shape")
            if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                    or not all(isinstance(v, int) and v > 0 for v in shape)):
                header_errors.append((base, "manifest shape must be two positive integers"))
                continue
            declared_n, declared_k = int(shape[0]), int(shape[1])
            if declared_k % HADAMARD_BLOCK:
                header_errors.append((base, f"declared K={declared_k} is not a multiple of 128"))
                continue
            suffixes = packed[base]["suffixes"]
            entries = {}
            for suffix in CORE_PACKED_SUFFIXES + ((".bias_corr",) if ".bias_corr" in suffixes else ()):  # header-only admission
                raw_name, shard = suffixes[suffix]
                entry = shard_headers[shard].get(raw_name)
                if entry is None:
                    header_errors.append((base, f"index key {raw_name} absent from shard {shard} header"))
                    continue
                hshape, hdtype = _header_shape_dtype(entry)
                if hdtype != header_dtypes[suffix]:
                    header_errors.append((base, f"{suffix} dtype {hdtype} != {header_dtypes[suffix]}"))
                    continue
                entries[suffix] = hshape
            if len(entries) < 3:
                continue
            qs = entries[".q_pack"]
            us = entries[".u_pack"]
            ds = entries[".d_row"]
            if len(qs) != 2 or len(us) != 2 or len(ds) != 1:
                header_errors.append((base, f"bad packed ranks q={qs} u={us} d={ds}"))
                continue
            actual_k = qs[0] * 8
            actual_n = qs[1]
            if us[0] * 64 != actual_k or us[1] != actual_n or ds[0] != actual_n:
                header_errors.append((base, f"packed q/u/d shape disagreement q={qs} u={us} d={ds}"))
                continue
            if actual_k != declared_k or actual_n != declared_n:
                header_errors.append((base, f"manifest shape {[declared_n, declared_k]} != header {[actual_n, actual_k]}"))
                continue
            index_has_bias = ".bias_corr" in suffixes
            manifest_has_bias = bool(meta.get("bias_corr", False))
            if index_has_bias != manifest_has_bias:
                header_errors.append((base, f"manifest bias_corr={manifest_has_bias} but checkpoint presence={index_has_bias}"))
                continue
            if index_has_bias:
                bs = entries.get(".bias_corr")
                if bs != [actual_n]:
                    header_errors.append((base, f"bias_corr header shape {bs} != [{actual_n}]"))
                    continue
            meta["shape"] = [actual_n, actual_k]
            meta["bias_corr"] = index_has_bias
            found[base] = meta
            actual_ks.add(actual_k)
        if header_errors:
            raise ValueError(f"hf_int4_g16: safetensors header admission failed: {header_errors[:8]}")

        missing = sorted(set(stored) - set(found))
        if missing:
            raise ValueError(f"hf_int4_g16: manifest modules missing q_pack tensors: {missing[:8]}")

        if sign_digests is not None:
            covered = {int(k) for k in sign_digests}
            missing_digest_ks = sorted(actual_ks - covered)
            if missing_digest_ks:
                raise ValueError(f"hf_int4_g16: sign_sha256_by_k missing digests for K={missing_digest_ks[:8]}")

        if not found:
            raise ValueError("hf_int4_g16: checkpoint contains no complete quantized linear inventory")
        self.quantized_modules = found
        logger.info("hf_int4_g16: %d quantized linear modules (%d with bias_corr); biascorr_mode=%s mode=%s backend=%s",
                    len(found), sum(1 for v in found.values() if v.get("bias_corr")),
                    biascorr_mode(), linear_mode(), os.environ.get("HF_INT4_G16_BACKEND", "auto"))

    def _members_for(self, prefix: str) -> list[str] | None:
        parts = prefix.split(".")
        if any(p in SKIP_PARTS for p in parts):
            return None
        key = _norm_key(prefix)
        if not self.quantized_modules:
            return None
        base, _, leaf = key.rpartition(".")
        members = FUSED.get(leaf, [leaf])
        names = [f"{base}.{p}" if base else p for p in members]
        present = [n in self.quantized_modules for n in names]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"hf_int4_g16: partial fused quantization at {prefix}: {list(zip(names, present))}")
        return names

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            members = self._members_for(prefix)
            if members is None:
                return UnquantizedLinearMethod()
            return HFInt4G16LinearMethod(self, members)
        # FusedMoE in vLLM 0.31 is a RoutedExperts/MoERunner composition rather
        # than a LinearBase.  HFInt4G16MoEMethod is intentionally not enabled
        # until expert-weight checkpoint remapping is implemented; router gates
        # and dense linears still use this method or exact bf16 fallback.
        return None


class HFInt4G16LinearMethod(LinearMethodBase):
    supports_pre_processed_weights = True

    def __init__(self, config: HFInt4G16Config, members: list[str]):
        self.config = config
        self.members = list(members)
        metas = [config.quantized_modules[m] for m in members]
        flags = [bool(m.get("bias_corr")) for m in metas]
        if len(set(flags)) != 1:
            raise ValueError(f"hf_int4_g16: mixed bias_corr presence in fused members {members}")
        self.has_bias_corr = flags[0]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        del extra_weight_attrs
        k = int(input_size_per_partition)
        n = int(sum(output_partition_sizes))
        if k % HADAMARD_BLOCK:
            raise ValueError(f"hf_int4_g16: input partition K={k} is not a multiple of 128")
        if input_size % HADAMARD_BLOCK:
            raise ValueError(f"hf_int4_g16: global input K={input_size} is not a multiple of 128")
        tp_size = int(getattr(layer, "tp_size", 1))
        if isinstance(layer, RowParallelLinear) and tp_size > 1 and input_size_per_partition % HADAMARD_BLOCK:
            raise ValueError("hf_int4_g16: row-parallel input shards must align to Hadamard blocks")

        layer.hf_output_sizes = list(output_partition_sizes)
        layer.hf_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.hf_input_size = int(input_size)
        layer.hf_input_size_per_partition = int(input_size_per_partition)
        layer.hf_loaded_members = set()  # legacy diagnostic; authoritative coverage is hf_loaded_shards
        layer.hf_loaded_shards = set()
        layer.hf_processed = False
        layer.hf_has_bias_corr = self.has_bias_corr
        layer.hf_backend = os.environ.get("HF_INT4_G16_BACKEND", "auto")
        layer.hf_biascorr_mode = biascorr_mode()
        layer.hf_linear_mode = linear_mode()

        # Preserve global output sizes before vLLM TP division where available.
        global_sizes = getattr(layer, "output_sizes", None)
        if isinstance(layer, RowParallelLinear):
            # Row-parallel outputs are not output-sharded: every rank produces
            # the full output width from its input shard.  Never infer a TP
            # multiple as this layer's physical checkpoint geometry.
            global_sizes = list(output_partition_sizes)
        elif global_sizes is None or len(global_sizes) != len(output_partition_sizes):
            global_sizes = [s * tp_size for s in output_partition_sizes]
        layer.hf_global_output_sizes = list(global_sizes)

        # Same load-member-run logic as the proven EXL3 plugin.  This covers the
        # in_proj_qkvz case where in_proj_qkv spans several vLLM output partitions.
        if len(self.members) == len(output_partition_sizes):
            n_span = 1
        elif len(self.members) < len(output_partition_sizes):
            n_span = len(output_partition_sizes) - len(self.members) + 1
        else:
            raise ValueError(f"hf_int4_g16: more checkpoint members than partitions for {self.members}")
        layer.hf_member_runs = [list(range(n_span))] + [[i] for i in range(n_span, len(output_partition_sizes))]
        if len(layer.hf_member_runs) != len(self.members):
            raise ValueError(f"hf_int4_g16: member/run mapping mismatch for {self.members}")
        layer.hf_member_meta = [dict(self.config.quantized_modules[m]) for m in self.members]

        # Validate physical checkpoint row counts against the layer's model
        # geometry before using them.  The manifest/header may prove that a
        # packed tensor exists, but they do not prove that its rows belong to
        # this layer: a 257-row member must not be silently truncated into a
        # 256-row output partition.  Replicated K/V is the single exception:
        # vLLM may expose the logically replicated width while the checkpoint
        # stores only the physical head rows.
        physical: list[int | None] = [None] * len(output_partition_sizes)
        kv_replicas = max(1, int(getattr(layer, "num_kv_head_replicas", 1)))
        for member_name, run, meta in zip(self.members, layer.hf_member_runs, layer.hf_member_meta):
            logical_total = int(sum(layer.hf_global_output_sizes[p] for p in run))
            shape = meta.get("shape")
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(f"hf_int4_g16: missing admitted physical shape for {member_name}")
            member_width = int(shape[0])
            is_replicated_kv = (
                tp_size > 1 and kv_replicas > 1
                and all(p in (1, 2) for p in run)
            )
            if is_replicated_kv:
                if logical_total % kv_replicas != 0 or member_width * kv_replicas != logical_total:
                    raise ValueError(
                        f"hf_int4_g16: replicated K/V physical width mismatch for {member_name}: "
                        f"checkpoint N={member_width}, expected physical N={logical_total // kv_replicas}, "
                        f"logical={logical_total}, kv_replicas={kv_replicas}, tp_size={tp_size}"
                    )
            elif member_width != logical_total:
                raise ValueError(
                    f"hf_int4_g16: physical output width mismatch for {member_name}: "
                    f"checkpoint N={member_width}, layer geometry N={logical_total}, "
                    f"partitions={run}, tp_size={tp_size}"
                )
            if len(run) == 1:
                physical[run[0]] = member_width
                continue
            assigned = 0
            widths = []
            for j, pidx in enumerate(run):
                if j == len(run) - 1:
                    width = member_width - assigned
                else:
                    width = int(round(member_width * layer.hf_global_output_sizes[pidx] / logical_total))
                    assigned += width
                widths.append(width)
            if any(w <= 0 for w in widths) or sum(widths) != member_width:
                raise ValueError(f"hf_int4_g16: cannot derive physical member widths for {member_name}")
            for pidx, width in zip(run, widths):
                physical[pidx] = width
        if any(v is None or int(v) <= 0 for v in physical):
            raise ValueError("hf_int4_g16: incomplete physical output width inventory")
        layer.hf_physical_output_sizes = [int(v) for v in physical]

        if (isinstance(layer, RowParallelLinear) and tp_size > 1 and self.has_bias_corr
                and not bool(getattr(layer, "reduce_results", True))
                and os.environ.get("HF_INT4_G16_ALLOW_UNREDUCED_ROW_TP") != "1"):
            raise ValueError(
                "hf_int4_g16: RowParallelLinear bias_corr requires reduce_results=True; "
                "set HF_INT4_G16_ALLOW_UNREDUCED_ROW_TP=1 only for a proven external single sum"
            )

        def reg(name, tensor, loader):
            p = Parameter(tensor, requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        reg("q_pack", torch.empty((k // 8, n), dtype=torch.int32), self._make_loader(layer, "q_pack"))
        reg("u_pack", torch.empty((k // 64, n), dtype=torch.int16), self._make_loader(layer, "u_pack"))
        reg("d_row", torch.empty((n,), dtype=torch.float32), self._make_loader(layer, "d_row"))
        # Always allocate zero bias_corr so unexpected checkpoint keys never meet
        # an unregistered parameter; process_weights_after_loading fail-closes on
        # presence/usage mismatch.  Keeping it zero also gives a deterministic
        # no-corr path without a second branch in weight loading.
        reg("bias_corr", torch.zeros((n,), dtype=torch.float32), self._make_loader(layer, "bias_corr"))

    @staticmethod
    def _shard_list(layer, sid=None) -> list[int]:
        if sid is None:
            return list(range(len(layer.hf_output_sizes)))
        if isinstance(sid, str):
            return [QKV_IDS[sid]]
        if isinstance(sid, int):
            return [sid]
        return list(sid)

    def _target_range(self, layer, sid=None):
        s = self._shard_list(layer, sid)
        if not s or s != list(range(s[0], s[-1] + 1)):
            raise ValueError(f"hf_int4_g16: non-contiguous shard id set {s}")
        return s, layer.hf_offsets[s[0]], layer.hf_offsets[s[-1] + 1]

    def _slice_input_if_row(self, layer, w: torch.Tensor, kind: str) -> torch.Tensor:
        if not isinstance(layer, RowParallelLinear) or int(getattr(layer, "tp_size", 1)) == 1:
            return w
        if kind not in ("q_pack", "u_pack"):
            return w
        factor = 8 if kind == "q_pack" else 64
        rows = layer.hf_input_size_per_partition // factor
        start = int(layer.tp_rank) * rows
        if w.shape[0] < start + rows:
            raise ValueError(f"hf_int4_g16: row-parallel {kind} input shard overflow {tuple(w.shape)}")
        return w.narrow(0, start, rows)

    def _extract_output_slice(self, layer, w: torch.Tensor, run: list[int], kind: str,
                              member_width: int) -> torch.Tensor:
        """Extract local TP output columns for one checkpoint member covering run.

        QKV K/V members may be replicated when ``num_kv_heads < tp_size``.  The
        checkpoint tensor still contains only the physical heads, so the replica
        group must use ``kv_rank = tp_rank // num_kv_head_replicas`` rather than
        the raw TP rank.  ``member_width`` is the physical checkpoint row count,
        not vLLM's logical replicated width.
        """
        out_dim = 1 if kind in ("q_pack", "u_pack") else 0
        tp_size = int(getattr(layer, "tp_size", 1))
        is_row = isinstance(layer, RowParallelLinear)
        pieces = []
        member_global_base = 0
        for p in run:
            local_w = int(layer.hf_output_sizes[p])
            physical_w = int(layer.hf_physical_output_sizes[p])
            if is_row or tp_size == 1:
                start, width = member_global_base, local_w
            else:
                shard_rank = int(layer.tp_rank)
                replicas = int(getattr(layer, "num_kv_head_replicas", 1))
                if replicas > 1 and p in (1, 2) and hasattr(layer, "num_kv_head_replicas"):
                    shard_rank = shard_rank // replicas
                start, width = member_global_base + shard_rank * local_w, local_w
            if width > physical_w:
                raise ValueError(
                    f"hf_int4_g16: {kind} local width {width} exceeds physical member width {physical_w}"
                )
            if start + width > w.shape[out_dim]:
                raise ValueError(
                    f"hf_int4_g16: {kind} output shard overflow: slice [{start},{start+width}) "
                    f"of {tuple(w.shape)} for partitions {run}"
                )
            pieces.append(w.narrow(out_dim, start, width))
            member_global_base += physical_w
        if member_global_base != member_width:
            raise ValueError(
                f"hf_int4_g16: member width accounting mismatch: partitions cover {member_global_base}, "
                f"checkpoint member has {member_width}"
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=out_dim)

    @staticmethod
    def _expected_global_shape(layer, run_idx: int, kind: str) -> tuple[int, ...]:
        meta = layer.hf_member_meta[run_idx]
        shape = meta.get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            n, k = int(shape[0]), int(shape[1])
        else:
            n = int(sum(layer.hf_physical_output_sizes[p] for p in layer.hf_member_runs[run_idx]))
            k = int(layer.hf_input_size)
        if k != int(layer.hf_input_size):
            raise ValueError(f"hf_int4_g16: manifest/input K mismatch for member {run_idx}: {k} != {layer.hf_input_size}")
        if kind == "q_pack":
            return (k // 8, n)
        if kind == "u_pack":
            return (k // 64, n)
        return (n,)

    @staticmethod
    def _expected_dtype(kind: str) -> torch.dtype:
        return {"q_pack": torch.int32, "u_pack": torch.int16,
                "d_row": torch.float32, "bias_corr": torch.float32}[kind]

    def _make_loader(self, layer, kind: str):
        def f(param, loaded_weight, shard_id=None):
            # shard_id is an attribute attached by vLLM to the *source* tensor.
            # Read it before any detach()/device conversion; .to() does not
            # preserve Python attributes.
            effective_shard_id = shard_id if shard_id is not None else getattr(loaded_weight, "shard_id", None)
            if kind == "bias_corr" and not layer.hf_has_bias_corr:
                raise ValueError("hf_int4_g16: checkpoint supplied bias_corr but manifest says it is absent")

            s, a, b = self._target_range(layer, effective_shard_id)
            run = None
            run_idx = None
            for idx, candidate in enumerate(layer.hf_member_runs):
                if candidate == s:
                    run = candidate
                    run_idx = idx
                    break
            if run is None:
                if len(s) == 1 and len(layer.hf_member_runs) == len(layer.hf_output_sizes):
                    run = s
                    run_idx = s[0]
                else:
                    raise ValueError(f"hf_int4_g16: load covers partitions {s}, not a member run {layer.hf_member_runs}")

            expected_shape = self._expected_global_shape(layer, run_idx, kind)
            expected_dtype = self._expected_dtype(kind)
            if tuple(loaded_weight.shape) != expected_shape:
                raise ValueError(
                    f"hf_int4_g16: {kind} global shape mismatch for member {run_idx}: "
                    f"{tuple(loaded_weight.shape)} != {expected_shape}"
                )
            if loaded_weight.dtype != expected_dtype:
                raise ValueError(
                    f"hf_int4_g16: {kind} dtype mismatch for member {run_idx}: "
                    f"{loaded_weight.dtype} != {expected_dtype}; refusing silent copy cast"
                )
            coverage_key = (kind, int(run_idx))
            if coverage_key in layer.hf_loaded_shards:
                raise ValueError(f"hf_int4_g16: duplicate load of {kind} member run {run_idx}")

            w = loaded_weight.detach().cpu() if loaded_weight.is_xpu else loaded_weight
            w = self._slice_input_if_row(layer, w, kind)
            member_width = expected_shape[-1] if kind in ("q_pack", "u_pack") else expected_shape[0]
            w = self._extract_output_slice(layer, w, run, kind, int(member_width))
            target_width = b - a
            out_dim = 1 if kind in ("q_pack", "u_pack") else 0
            if w.shape[out_dim] != target_width:
                raise ValueError(
                    f"hf_int4_g16: {kind} width mismatch {w.shape[out_dim]} != target {target_width}; "
                    f"loaded {tuple(loaded_weight.shape)} partitions {s}"
                )
            dst = param.data.narrow(out_dim, a, target_width)
            if dst.shape != w.shape:
                raise ValueError(f"hf_int4_g16: {kind} shape mismatch {tuple(w.shape)} -> {tuple(dst.shape)}")
            if dst.dtype != w.dtype:
                raise ValueError(f"hf_int4_g16: {kind} destination dtype mismatch {dst.dtype} != {w.dtype}")
            dst.copy_(w)
            layer.hf_loaded_shards.add(coverage_key)
            if kind == "q_pack":
                layer.hf_loaded_members.add(tuple(run))
        return f

    def process_weights_after_loading(self, layer):
        if getattr(layer, "hf_processed", False):
            raise ValueError("hf_int4_g16: refusing to process an already-processed layer")
        expected = set()
        for run_idx in range(len(layer.hf_member_runs)):
            for kind in ("q_pack", "u_pack", "d_row"):
                expected.add((kind, run_idx))
            if layer.hf_has_bias_corr:
                expected.add(("bias_corr", run_idx))
        got = set(layer.hf_loaded_shards)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(
                f"hf_int4_g16: incomplete fused suffix loading: missing {missing[:12]}, extra {extra[:12]}"
            )

        k_local = int(layer.hf_input_size_per_partition)
        n_local = int(sum(layer.hf_output_sizes))
        expected_param_shapes = {
            "q_pack": (k_local // 8, n_local),
            "u_pack": (k_local // 64, n_local),
            "d_row": (n_local,),
            "bias_corr": (n_local,),
        }
        for name, dtype in [("q_pack", torch.int32), ("u_pack", torch.int16),
                            ("d_row", torch.float32), ("bias_corr", torch.float32)]:
            p = getattr(layer, name)
            if p.dtype != dtype:
                raise ValueError(f"hf_int4_g16: {name} dtype mutated to {p.dtype}")
            if tuple(p.shape) != expected_param_shapes[name]:
                raise ValueError(f"hf_int4_g16: {name} parameter shape {tuple(p.shape)} != {expected_param_shapes[name]}")
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        tp_size = int(getattr(layer, "tp_size", 1))
        rank = int(getattr(layer, "tp_rank", 0))
        if isinstance(layer, RowParallelLinear) and tp_size > 1:
            offset = rank * layer.hf_input_size_per_partition
        else:
            offset = 0
        full = signs_for_k(layer.hf_input_size, self.config.had_seed, "cpu")
        signs = full[offset:offset + k_local].contiguous()
        if signs.numel() != k_local:
            raise ValueError("hf_int4_g16: sign slice shorter than local input partition")
        layer.register_buffer("hf_signs", signs.to(layer.q_pack.device), persistent=False)
        if layer.hf_has_bias_corr and layer.hf_biascorr_mode == "neg_zero":
            layer.bias_corr.data.fill_(-0.0)
        layer.hf_processed = True

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        bc = None
        if layer.hf_has_bias_corr and layer.hf_biascorr_mode != "off":
            # RowParallelLinear output is reduced across ranks.  rank 0 alone owns
            # the additive correction, exactly like vLLM's native bias handling.
            # mode=off is a true null pointer, not a zero-filled tensor.
            if not isinstance(layer, RowParallelLinear) or int(getattr(layer, "tp_rank", 0)) == 0:
                bc = layer.bias_corr
        # A row-sharded TP>1 layer may not use per-rank local-amax W4A8: the
        # scale is not globally reduced, so exact/speculative numerics would
        # silently become rank-partition dependent.  W4A16 remains exact under
        # block-wise row partitioning because the Hadamard blocks do not cross
        # shard boundaries.
        mode = "w4a16" if isinstance(layer, RowParallelLinear) and int(getattr(layer, "tp_size", 1)) > 1 else layer.hf_linear_mode
        return hf_int4_g16_linear(
            x, layer.q_pack, layer.u_pack, layer.d_row, layer.hf_signs,
            bias=bias, bias_corr=bc, backend=layer.hf_backend, mode=mode,
        )


def register():
    """vllm.general_plugins entry point (import side effect registers the config)."""
    return None
