"""Fail-closed admission of packed routed-expert (MoE) checkpoints.

Everything here is metadata only: the manifest, the safetensors index and the
shard headers.  Payload values (finite, positive scales) are checked when the
tensors are loaded.

Every routed-expert layer the model builds must be covered one of two ways: a
complete packed bank (:func:`admit_expert_tensors`) or a complete
full-precision bank in the same checkpoint (:class:`FullPrecisionBanks`).
:class:`PackedCoverage` then proves that each admitted packed module was bound
to a layer and processed.

Manifest, one entry per expert namespace (keys end in ``.experts``)::

    "moe_modules": {
      "layers.0.mlp.experts": {
        "num_experts": E, "hidden_size": H, "intermediate_size": I,
        "projections": {"gate_proj": {"had_block": 128, "bias_corr": false},
                        "up_proj":   {...}, "down_proj": {...}}
      }
    }

Checkpoint tensors are per expert:
``<raw namespace>.<e>.<projection>.<qweight|scales|scale_super|bias_corr>``.
All experts of a namespace share shape, Hadamard block and bias presence.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .experts_source import canonical_expert_name, is_granite_legacy_name
from .format import EXPERT_PROJECTIONS, EXPERT_TENSOR_SUFFIXES, had_block_for_k, is_json_int

_SUFFIX_ALTERNATION = "|".join(EXPERT_TENSOR_SUFFIXES.values())
EXPERT_KEY_RE = re.compile(
    r"^(?P<ns>.+\.experts)\.(?P<expert>0|[1-9]\d*)\."
    rf"(?P<proj>{'|'.join(EXPERT_PROJECTIONS)})\.(?P<suffix>{_SUFFIX_ALTERNATION})$"
)
# Suffixes that only the expert layout uses.  Seen anywhere else, they are corrupt.
_EXPERT_ONLY_SUFFIXES = (
    "." + EXPERT_TENSOR_SUFFIXES["q_pack"],
    "." + EXPERT_TENSOR_SUFFIXES["d_row"],
)
_CORE_ROLES = ("q_pack", "u_pack", "d_row")
_HEADER_DTYPES = {"q_pack": "I32", "u_pack": "I16", "d_row": "F32", "bias_corr": "F32"}
_FULL_PRECISION_DTYPES = ("BF16", "F16", "F32")
_BANK_MEMBER_RE = re.compile(r"^(?P<ns>.+?\.experts)\.(?P<member>.+)$")
# Structured member key: (is_raw_granite_legacy, member_name).  Keeping raw
# GraniteMoE provenance separate from the member string prevents a literal
# checkpoint member name from colliding with an internal prefix marker.
MemberKey = tuple[bool, str]
# Per-expert projection names: (gate, up, down).
_PER_EXPERT_NAMES = (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2"))
VISION_PARTS = frozenset({"visual", "vision_tower", "multi_modal_projector"})

BankId = tuple[bool, str]  # (inside a vision tower, normalized expert namespace)


def bank_id(namespace: str, norm_key: Callable[[str], str]) -> BankId:
    """Identity of an expert namespace.  The tower flag keeps ``norm_key`` from
    merging ``visual...layers.0`` with the text decoder's ``layers.0``."""
    return bool(VISION_PARTS & set(namespace.split("."))), norm_key(namespace)


def bank_member_of(name: str, norm_key: Callable[[str], str]) -> tuple[BankId, str] | None:
    """``(bank, name inside the bank)`` for any tensor under an expert namespace,
    including GraniteMoE's legacy ``block_sparse_moe.{input,output}_linear``."""
    match = _BANK_MEMBER_RE.match(canonical_expert_name(name))
    if match is None:
        return None
    return bank_id(match["ns"], norm_key), match["member"]


def is_expert_packed_name(name: str) -> bool:
    return EXPERT_KEY_RE.match(name) is not None or name.endswith(_EXPERT_ONLY_SUFFIXES)


@dataclass(frozen=True)
class ExpertProjection:
    had_block: int
    bias_corr: bool


@dataclass(frozen=True)
class MoEModule:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    projections: dict[str, ExpertProjection]

    def shape(self, proj: str) -> tuple[int, int]:
        """Logical ``(N, K)`` of one expert's projection."""
        if proj == "down_proj":
            return self.hidden_size, self.intermediate_size
        return self.intermediate_size, self.hidden_size

    def tensor_shape(self, proj: str, role: str) -> tuple[int, ...]:
        n, k = self.shape(proj)
        if role == "q_pack":
            return (k // 8, n)
        if role == "u_pack":
            return (k // 64, n)
        return (n,)

    def roles(self, proj: str) -> tuple[str, ...]:
        return _CORE_ROLES + (("bias_corr",) if self.projections[proj].bias_corr else ())


def _positive_int(value: Any) -> bool:
    return is_json_int(value) and value > 0


def _parse_module(key: str, raw: Any) -> MoEModule:
    fail = f"hf_int4_g16: MoE manifest entry {key}"
    fields = {"num_experts", "hidden_size", "intermediate_size", "projections"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"{fail} must have exactly the fields {sorted(fields)}")
    for field in ("num_experts", "hidden_size", "intermediate_size"):
        if not _positive_int(raw[field]):
            raise ValueError(f"{fail}: {field} must be a positive integer, got {raw[field]!r}")
    projections = raw["projections"]
    if not isinstance(projections, dict) or set(projections) != set(EXPERT_PROJECTIONS):
        raise ValueError(f"{fail}: projections must be exactly {list(EXPERT_PROJECTIONS)}")
    parsed = {}
    for proj in EXPERT_PROJECTIONS:
        item = projections[proj]
        if (not isinstance(item, dict) or set(item) != {"had_block", "bias_corr"}
                or not isinstance(item["bias_corr"], bool)):
            raise ValueError(f"{fail}: {proj} needs an integer had_block and a boolean bias_corr")
        k = raw["intermediate_size"] if proj == "down_proj" else raw["hidden_size"]
        try:
            expected_block = had_block_for_k(k)
        except ValueError as e:
            raise ValueError(f"{fail}: {proj}: {e}") from e
        if not is_json_int(item["had_block"]) or item["had_block"] != expected_block:
            raise ValueError(
                f"{fail}: {proj} had_block {item['had_block']!r} does not match K={k} (expected {expected_block})"
            )
        parsed[proj] = ExpertProjection(expected_block, item["bias_corr"])
    if parsed["gate_proj"].bias_corr != parsed["up_proj"].bias_corr:
        raise ValueError(f"{fail}: gate_proj and up_proj share one packed tensor, so bias_corr presence must match")
    return MoEModule(raw["num_experts"], raw["hidden_size"], raw["intermediate_size"], parsed)


def parse_moe_manifest(raw: Any, norm_key: Callable[[str], str]) -> dict[str, MoEModule]:
    if not isinstance(raw, dict):
        raise ValueError("hf_int4_g16: moe_modules manifest is not an object")
    modules: dict[str, MoEModule] = {}
    for key, entry in raw.items():
        normalized = norm_key(key)
        if not normalized.endswith(".experts"):
            raise ValueError(f"hf_int4_g16: MoE manifest key {key} is not an expert namespace ending in .experts")
        if normalized in modules:
            raise ValueError(f"hf_int4_g16: normalized MoE manifest key collision: {normalized}")
        modules[normalized] = _parse_module(key, entry)
    return modules


def admit_expert_tensors(
    modules: dict[str, MoEModule],
    weight_map: dict[str, str],
    shard_headers: dict[str, dict[str, Any]],
    norm_key: Callable[[str], str],
) -> set[int]:
    """Require the index and headers to hold exactly the declared expert tensors.

    Returns every global K used by admitted experts, for sign-digest coverage.
    """
    inventory: dict[str, dict[tuple[int, str, str], str]] = {}
    raw_namespaces: dict[str, set[str]] = {}
    stray = []
    role_of = {suffix: role for role, suffix in EXPERT_TENSOR_SUFFIXES.items()}
    for name in weight_map:
        match = EXPERT_KEY_RE.match(name)
        if match is None:
            if name.endswith(_EXPERT_ONLY_SUFFIXES):
                stray.append(name)
            continue
        namespace = norm_key(match["ns"])
        raw_namespaces.setdefault(namespace, set()).add(match["ns"])
        slot = (int(match["expert"]), match["proj"], role_of[match["suffix"]])
        inventory.setdefault(namespace, {})[slot] = name
    if stray:
        raise ValueError(f"hf_int4_g16: packed expert/MoE tensors outside the per-expert layout: {sorted(stray)[:8]}")
    undeclared = sorted(set(inventory) - set(modules))
    if undeclared:
        raise ValueError(f"hf_int4_g16: packed expert/MoE namespaces missing from moe_modules: {undeclared[:8]}")
    mixed = {ns: sorted(raw) for ns, raw in raw_namespaces.items() if len(raw) != 1}
    if mixed:
        raise ValueError(f"hf_int4_g16: cross-namespace expert tensor mixing: {mixed}")

    # An expert namespace is either fully packed or not packed at all.  vLLM
    # reads every tensor in every shard, so a member that only a shard header
    # knows about would still reach the loader: check headers and index alike.
    packed_banks = {bank_id(next(iter(raw)), norm_key) for raw in raw_namespaces.values()}
    locations = {(name, shard) for shard, header in shard_headers.items() for name in header if name != "__metadata__"}
    locations |= set(weight_map.items())
    leftovers = set()
    for name, shard in locations:
        member = bank_member_of(name, norm_key)
        if member is None or member[0] not in packed_banks:
            continue
        if EXPERT_KEY_RE.match(name) is None or weight_map.get(name) != shard:
            leftovers.add(name)
    if leftovers:
        raise ValueError(f"hf_int4_g16: unpacked tensors inside a packed expert namespace: {sorted(leftovers)[:8]}")

    errors: list[str] = []
    widths: set[int] = set()
    for namespace, module in sorted(modules.items()):
        present = inventory.get(namespace, {})
        expected = {
            (expert, proj, role)
            for expert in range(module.num_experts)
            for proj in EXPERT_PROJECTIONS
            for role in module.roles(proj)
        }
        for slot in sorted(expected - set(present)):
            errors.append(f"{namespace}: missing expert tensor expert={slot[0]} {slot[1]}.{EXPERT_TENSOR_SUFFIXES[slot[2]]}")
        for slot in sorted(set(present) - expected):
            errors.append(
                f"{namespace}: unexpected expert tensor {present[slot]} "
                f"(num_experts={module.num_experts}, bias_corr declared="
                f"{module.projections[slot[1]].bias_corr})"
            )
        for slot in sorted(expected & set(present)):
            _, proj, role = slot
            name = present[slot]
            entry = shard_headers[weight_map[name]].get(name)
            if not isinstance(entry, dict):
                errors.append(f"{namespace}: index key {name} absent from shard {weight_map[name]} header")
                continue
            want_shape = list(module.tensor_shape(proj, role))
            if entry.get("dtype") != _HEADER_DTYPES[role]:
                errors.append(f"{name}: dtype {entry.get('dtype')} != {_HEADER_DTYPES[role]}")
            if entry.get("shape") != want_shape:
                errors.append(f"{name}: header shape {entry.get('shape')} != {want_shape} (wrong K or N)")
        widths.update((module.hidden_size, module.intermediate_size))
    if errors:
        raise ValueError(f"hf_int4_g16: expert/MoE admission failed ({len(errors)} problems): {errors[:8]}")
    return widths


class FullPrecisionBanks:
    """The unpacked routed-expert tensors a checkpoint holds, by header only.

    A layer without a ``moe_modules`` entry may only fall back to vLLM's
    unquantized method if its weights are really there.  vLLM does not report
    every missing expert weight, so an absent bank would otherwise run
    uninitialized memory.
    """

    def __init__(self, weight_map: dict[str, str], shard_headers: dict[str, dict[str, Any]],
                 norm_key: Callable[[str], str]):
        self._banks: dict[BankId, dict[MemberKey, tuple[Any, Any]]] = {}
        self._norm_key = norm_key
        for name, shard in weight_map.items():
            member = bank_member_of(name, norm_key)
            entry = shard_headers[shard].get(name)
            if member is None or EXPERT_KEY_RE.match(name) or not isinstance(entry, dict):
                continue
            inner: MemberKey = (is_granite_legacy_name(name), member[1])
            self._banks.setdefault(member[0], {})[inner] = (entry.get("shape"), entry.get("dtype"))

    def require(self, prefix: str, num_experts: int, hidden_size: int, intermediate_size: int, gated: bool) -> None:
        """Raise unless the bank of the layer at ``prefix`` is complete."""
        vision, key = bank_id(prefix, self._norm_key)
        candidates = [(vision, key), (vision, f"{key}.experts")]
        members = next((self._banks[c] for c in candidates if c in self._banks), None)
        if members is None:
            raise ValueError(
                f"hf_int4_g16: routed experts at {prefix} have no packed moe_modules entry and no "
                "full-precision weights in the checkpoint"
            )
        problems = _bank_problems(members, num_experts, hidden_size, intermediate_size, gated)
        if problems:
            raise ValueError(
                f"hf_int4_g16: routed experts at {prefix} have no packed moe_modules entry and their "
                f"full-precision bank is incomplete ({len(problems)} problems): {problems[:6]}"
            )


def _bank_problems(members: dict[MemberKey, tuple[Any, Any]], experts: int, hidden: int, inter: int,
                   gated: bool) -> list[str]:
    def check(found: list[str], key: MemberKey, shape: list[int]) -> None:
        got_shape, dtype = members[key] if key in members else (None, None)
        label = key[1]
        if key not in members:
            found.append(f"missing {label}")
        elif got_shape != shape or dtype not in _FULL_PRECISION_DTYPES:
            found.append(f"{label} is {got_shape} {dtype}, expected {shape} in {_FULL_PRECISION_DTYPES}")

    def fused(gate_up: MemberKey, down: MemberKey) -> list[str]:
        found: list[str] = []
        check(found, gate_up, [experts, (2 if gated else 1) * inter, hidden])
        check(found, down, [experts, hidden, inter])
        return found

    stem = "gate_up_proj" if gated else "up_proj"
    for gate_up, down in ((stem, "down_proj"), (f"{stem}.weight", "down_proj.weight")):
        if (False, gate_up) in members:
            return fused((False, gate_up), (False, down))
    if (True, "input_linear.weight") in members:
        return fused((True, "input_linear.weight"), (True, "output_linear.weight"))
    for gate, up, down in _PER_EXPERT_NAMES:
        if (False, f"0.{up}.weight") in members:
            found: list[str] = []
            for expert in range(experts):
                if gated:
                    check(found, (False, f"{expert}.{gate}.weight"), [inter, hidden])
                check(found, (False, f"{expert}.{up}.weight"), [inter, hidden])
                check(found, (False, f"{expert}.{down}.weight"), [hidden, inter])
            return found
    return [f"unrecognized expert layout, for example {sorted(m for _, m in members)[:4]}"]


class PackedCoverage:
    """Which admitted packed modules the model actually bound and processed.

    A renamed or dropped layer would otherwise leave its packed weights unread
    and the layer running on uninitialized or unquantized parameters.
    """

    def __init__(self) -> None:
        self.claimed_linear: set[str] = set()
        self.claimed_moe: set[str] = set()
        self.processed_linear: set[str] = set()
        self.processed_moe: set[str] = set()
        self.verified = False

    def verify(self, admitted_linear: set[str], admitted_moe: set[str]) -> None:
        """Require processed == admitted for both sets."""
        problems = []
        for kind, admitted, claimed, processed in (
            ("linear", admitted_linear, self.claimed_linear, self.processed_linear),
            ("expert", admitted_moe, self.claimed_moe, self.processed_moe),
        ):
            for label, names in (
                ("no model layer asked for", admitted - claimed),
                ("claimed but never processed", (admitted & claimed) - processed),
                ("processed but never admitted", processed - admitted),
            ):
                if names:
                    problems.append(f"{kind} modules {label} ({len(names)}): {sorted(names)[:6]}")
        if problems:
            raise ValueError(
                "hf_int4_g16: packed checkpoint coverage is incomplete, refusing to run: " + "; ".join(problems)
            )
        self.verified = True
