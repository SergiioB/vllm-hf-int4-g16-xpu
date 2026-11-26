"""Routed-expert tensors in a BF16/FP16 source checkpoint.

Three on-disk layouts become one :class:`ExpertSource` per namespace:

* per expert: ``<ns>.<e>.{gate_proj,up_proj,down_proj}.weight`` ``[N,K]``
* fused 3D: ``<ns>.gate_up_proj`` ``[E,2I,H]`` and ``<ns>.down_proj``
  ``[E,H,I]``, with or without a ``.weight`` suffix
* GraniteMoE legacy: ``<m>.block_sparse_moe.{input_linear,output_linear}.weight``
  with the fused 3D shapes; the namespace becomes ``<m>.block_sparse_moe.experts``

``<ns>`` ends in ``.experts``.  Any other tensor that looks like it belongs to a
routed expert is an error: copying or half-quantizing it would produce a
checkpoint vLLM cannot load correctly.

Finding a bank (:func:`scan_experts`) is separate from deciding that it can be
packed (:func:`require_packable`).  A caller's opt-out applies to a whole bank:
if it matches any of the bank's keys or its canonical namespace, every tensor of
the bank is left as it is in the source.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .format import EXPERT_PROJECTIONS, had_block_for_k

_PER_EXPERT = re.compile(r"^(?P<ns>.+\.experts)\.(?P<expert>0|[1-9]\d*)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")
_FUSED = re.compile(r"^(?P<ns>.+\.experts)\.(?P<proj>gate_up_proj|down_proj)(\.weight)?$")
_GRANITE = re.compile(r"^(?P<base>.+\.block_sparse_moe)\.(?P<io>input_linear|output_linear)\.weight$")
_LOOKS_LIKE_EXPERT = re.compile(r"\.experts\.|\.block_sparse_moe\.(input_linear|output_linear)\.")
_GRANITE_PROJ = {"input_linear": "gate_up_proj", "output_linear": "down_proj"}
_GRANITE_PREFIX = re.compile(r"\.block_sparse_moe\.(input_linear|output_linear)\.")
_CONFIG_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_FLOAT_DTYPES = ("BF16", "F16", "F32")

# (key) -> (shape, safetensors dtype name)
MetaFn = Callable[[str], tuple[list[int], str]]
# (key) -> full tensor; (key, index) -> tensor[index] without reading the rest
ReadFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class ExpertSource:
    namespace: str  # raw checkpoint namespace ending in ".experts"
    num_experts: int
    hidden_size: int
    intermediate_size: int
    fused_keys: dict[str, str] | None  # {"gate_up_proj": key, "down_proj": key}
    per_expert_keys: dict[tuple[int, str], str] | None

    @property
    def keys(self) -> list[str]:
        return sorted((self.fused_keys or self.per_expert_keys).values())

    def shape(self, proj: str) -> tuple[int, int]:
        if proj == "down_proj":
            return self.hidden_size, self.intermediate_size
        return self.intermediate_size, self.hidden_size

    def read(self, read: ReadFn, expert: int, proj: str) -> torch.Tensor:
        """One expert's ``[N,K]`` weight."""
        if self.per_expert_keys is not None:
            return read(self.per_expert_keys[(expert, proj)])
        if proj == "down_proj":
            return read(self.fused_keys["down_proj"], expert)
        both = read(self.fused_keys["gate_up_proj"], expert)
        i = self.intermediate_size
        return both[:i] if proj == "gate_proj" else both[i:]

    def read_fused(self, read: ReadFn, fused_name: str) -> torch.Tensor:
        """The stacked ``gate_up_proj`` ``[E,2I,H]`` or ``down_proj`` ``[E,H,I]``."""
        if self.fused_keys is not None:
            return read(self.fused_keys[fused_name])
        projs = ("gate_proj", "up_proj") if fused_name == "gate_up_proj" else ("down_proj",)
        return torch.stack([
            torch.cat([self.read(read, e, proj) for proj in projs], dim=0)
            for e in range(self.num_experts)
        ])


def configured_expert_count(config: dict) -> int | None:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    for key in _CONFIG_EXPERT_COUNT_KEYS:
        value = text.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _classify(key: str) -> tuple[str, str, int | None] | None:
    """``(namespace, projection or fused name, expert or None)`` for an expert weight."""
    if match := _PER_EXPERT.match(key):
        return match["ns"], match["proj"], int(match["expert"])
    if match := _FUSED.match(key):
        return match["ns"], match["proj"], None
    if match := _GRANITE.match(key):
        return match["base"] + ".experts", _GRANITE_PROJ[match["io"]], None
    return None


def is_granite_legacy_name(key: str) -> bool:
    """Whether ``key`` is a raw GraniteMoE ``<m>.block_sparse_moe.{input,output}_linear.weight``."""
    return _GRANITE.match(key) is not None


def canonical_expert_name(key: str) -> str:
    """``key`` with GraniteMoE's legacy names moved under their ``.experts`` namespace."""
    return _GRANITE_PREFIX.sub(r".block_sparse_moe.experts.\1.", key)


def _fused_source(namespace: str, keys: dict[str, str], meta: MetaFn) -> ExpertSource:
    if set(keys) != {"gate_up_proj", "down_proj"}:
        raise ValueError(f"expert namespace {namespace}: fused layout needs gate_up_proj and down_proj, got {sorted(keys)}")
    gate_up, down = (meta(keys[name]) for name in ("gate_up_proj", "down_proj"))
    for name, (shape, dtype) in (("gate_up_proj", gate_up), ("down_proj", down)):
        if len(shape) != 3 or dtype not in _FLOAT_DTYPES:
            raise ValueError(f"expert namespace {namespace}: {name} must be a float 3D tensor, got {shape} {dtype}")
    experts, two_i, hidden = gate_up[0]
    intermediate = two_i // 2
    if two_i % 2 or down[0] != [experts, hidden, intermediate]:
        raise ValueError(
            f"expert namespace {namespace}: gate_up_proj {gate_up[0]} and down_proj {down[0]} are not "
            "[E,2I,H] and [E,H,I]"
        )
    return ExpertSource(namespace, experts, hidden, intermediate, dict(keys), None)


def _per_expert_source(namespace: str, keys: dict[tuple[int, str], str], meta: MetaFn) -> ExpertSource:
    experts = max(expert for expert, _ in keys) + 1
    missing = sorted(
        (e, proj) for e in range(experts) for proj in EXPERT_PROJECTIONS if (e, proj) not in keys
    )
    if missing:
        raise ValueError(f"expert namespace {namespace}: missing expert tensors (expert, projection): {missing[:8]}")
    intermediate, hidden = meta(keys[(0, "gate_proj")])[0]
    source = ExpertSource(namespace, experts, hidden, intermediate, None, dict(keys))
    for (expert, proj), key in sorted(keys.items()):
        shape, dtype = meta(key)
        if shape != list(source.shape(proj)) or dtype not in _FLOAT_DTYPES:
            raise ValueError(
                f"expert namespace {namespace}: {key} is {shape} {dtype}, expected {list(source.shape(proj))} float"
            )
    return source


def scan_experts(
    keys: list[str], meta: MetaFn, config: dict, skipped: Callable[[str], bool] | None = None,
    opted_out: set[str] | None = None,
) -> list[ExpertSource]:
    """Find and structurally validate every routed-expert bank that is not opted out.

    ``skipped`` is the caller's opt-out predicate.  Opted-out banks are neither
    validated nor returned; all of their keys are added to ``opted_out`` so the
    caller can copy them unchanged instead of treating them as dense linears.
    Hadamard grid limits are not checked here (see :func:`require_packable`).
    """
    skipped = skipped or (lambda name: False)
    banks: dict[str, list[tuple[str, str, int | None]]] = {}
    unknown = []
    for key in keys:
        found = _classify(key)
        if found is None:
            if _LOOKS_LIKE_EXPERT.search(key) and not (skipped(key) or skipped(canonical_expert_name(key))):
                unknown.append(key)
            continue
        namespace, proj, expert = found
        banks.setdefault(namespace, []).append((key, proj, expert))
    if unknown:
        raise ValueError(f"unrecognized routed-expert tensors (biases or an unsupported layout): {sorted(unknown)[:8]}")

    fused: dict[str, dict[str, str]] = {}
    per_expert: dict[str, dict[tuple[int, str], str]] = {}
    for namespace, entries in banks.items():
        if skipped(namespace + ".") or any(skipped(key) for key, _, _ in entries):
            if opted_out is not None:
                opted_out.update(key for key, _, _ in entries)
            continue
        for key, proj, expert in entries:
            if expert is None:
                if proj in fused.setdefault(namespace, {}):
                    raise ValueError(f"expert namespace {namespace}: duplicate {proj}")
                fused[namespace][proj] = key
            else:
                per_expert.setdefault(namespace, {})[(expert, proj)] = key
    both = sorted(set(fused) & set(per_expert))
    if both:
        raise ValueError(f"expert namespaces mix fused and per-expert layouts: {both[:4]}")

    sources = [_fused_source(ns, ns_keys, meta) for ns, ns_keys in sorted(fused.items())]
    sources += [_per_expert_source(ns, ns_keys, meta) for ns, ns_keys in sorted(per_expert.items())]
    wanted = configured_expert_count(config)
    for source in sources:
        if wanted is not None and source.num_experts != wanted:
            raise ValueError(
                f"expert namespace {source.namespace}: checkpoint has {source.num_experts} experts, "
                f"config.json declares {wanted}"
            )
    return sorted(sources, key=lambda s: s.namespace)


def require_packable(sources: list[ExpertSource]) -> None:
    """Every projection of a bank that will be packed must fit a Hadamard block."""
    for source in sources:
        for proj in EXPERT_PROJECTIONS:
            try:
                had_block_for_k(source.shape(proj)[1])
            except ValueError as e:
                raise ValueError(
                    f"expert namespace {source.namespace} {proj}: {e}. "
                    "Keep the experts in BF16 with --skip-patterns ...,experts"
                ) from e
