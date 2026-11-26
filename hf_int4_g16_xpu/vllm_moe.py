"""vLLM ``FusedMoEMethodBase`` for packed HF-INT4-G16 routed experts.

vLLM 0.31 gives the quantization method a ``RoutedExperts`` module.  Its
``load_weights`` maps checkpoint keys onto ``w13_*`` / ``w2_*`` parameters
(``gate_proj`` and ``up_proj`` feed the two halves of ``w13``) and calls each
parameter's ``weight_loader`` once per expert.  vLLM does not track
uninitialized weights for quantized models, so this method proves full coverage
itself before the layer can run.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

from .format import EXPERT_TENSOR_SUFFIXES, check_payload, signs_for_k
from .moe import MOE_ACTIVATIONS, PackedExperts, hf_int4_g16_moe
from .moe_admission import MoEModule

logger = init_logger("vllm.hf_int4_g16_xpu")

_ROLE_DTYPES = {"q_pack": torch.int32, "u_pack": torch.int16, "d_row": torch.float32, "bias_corr": torch.float32}
# checkpoint shard id -> (stacked parameter group, checkpoint projection)
_SHARDS = {"w1": ("w13", "gate_proj"), "w3": ("w13", "up_proj"), "w2": ("w2", "down_proj")}


class HFInt4G16MoEMethod(FusedMoEMethodBase):
    """Exact W4A16 routed experts.  Router and shared experts stay with vLLM."""

    def __init__(self, moe: FusedMoEConfig, module: MoEModule, namespace: str, had_seed: int,
                 biascorr_mode: str, linear_mode: str, backend: str, quant_config):
        super().__init__(moe)
        if linear_mode == "w4a8":
            raise ValueError(
                "hf_int4_g16: HF_INT4_G16_MODE=w4a8 is not implemented for MoE experts "
                "(they run exact W4A16); use w4a16 or auto"
            )
        self.module = module
        self.namespace = namespace
        self.had_seed = int(had_seed)
        self.biascorr_mode = biascorr_mode
        self.linear_mode = linear_mode
        self.backend = backend
        self.quant_config = quant_config

    def _fail(self, message: str) -> ValueError:
        return ValueError(f"hf_int4_g16: MoE layer {self.namespace}: {message}")

    def _check_supported(self, layer) -> None:
        moe, parallel, module = self.moe, self.moe.moe_parallel_config, self.module
        if (parallel.tp_size != 1 or parallel.dp_size != 1 or parallel.pcp_size != 1
                or parallel.use_ep or parallel.is_sequence_parallel):
            raise self._fail(
                f"only single-GPU MoE is supported (tp={parallel.tp_size}, dp={parallel.dp_size}, "
                f"pcp={parallel.pcp_size}, ep={parallel.use_ep}, sp={parallel.is_sequence_parallel})"
            )
        if parallel.enable_eplb:
            raise self._fail("expert load balancing (EPLB) is not supported")
        if moe.activation.value not in MOE_ACTIVATIONS:
            raise self._fail(
                f"unsupported activation {moe.activation.value!r}; supported gated activations: {sorted(MOE_ACTIVATIONS)}"
            )
        if moe.has_bias:
            raise self._fail("expert model biases are not part of this format")
        if any(v is not None for v in (moe.swiglu_limit, moe.swiglu_alpha, moe.swiglu_beta)):
            raise self._fail("SwiGLU clamp/alpha/beta variants are not supported")
        if getattr(layer, "apply_router_weight_on_input", False):
            raise self._fail("apply_router_weight_on_input is not supported")
        if moe.num_experts != module.num_experts:
            raise self._fail(f"model has {moe.num_experts} experts, checkpoint declares {module.num_experts}")
        hidden = moe.hidden_dim_unpadded or moe.hidden_dim
        if hidden != module.hidden_size or moe.hidden_dim != module.hidden_size:
            raise self._fail(f"model hidden size {moe.hidden_dim} != checkpoint hidden_size {module.hidden_size}")
        if moe.intermediate_size != module.intermediate_size:
            raise self._fail(
                f"model intermediate size {moe.intermediate_size} != checkpoint intermediate_size "
                f"{module.intermediate_size}"
            )

    def create_weights(self, layer, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        del params_dtype, extra_weight_attrs
        self._check_supported(layer)
        module = self.module
        e, h, i = module.num_experts, module.hidden_size, module.intermediate_size
        if (num_experts, hidden_size, intermediate_size_per_partition) != (e, h, i):
            raise self._fail(
                f"vLLM geometry (E={num_experts}, H={hidden_size}, I={intermediate_size_per_partition}) "
                f"!= checkpoint (E={e}, H={h}, I={i})"
            )
        layer.hf_loaded_experts = set()
        layer.hf_processed = False
        groups = {"w13": (2 * i, h, module.projections["gate_proj"]), "w2": (h, i, module.projections["down_proj"])}
        for group, (n, k, projection) in groups.items():
            shapes = {"q_pack": (e, k // 8, n), "u_pack": (e, k // 64, n), "d_row": (e, n)}
            if projection.bias_corr:
                shapes["bias_corr"] = (e, n)
            for role, shape in shapes.items():
                name = f"{group}_{EXPERT_TENSOR_SUFFIXES[role]}"
                parameter = Parameter(torch.empty(shape, dtype=_ROLE_DTYPES[role]), requires_grad=False)
                parameter.weight_loader = self._make_loader(layer, group, role, name)
                layer.register_parameter(name, parameter)

    def _make_loader(self, layer, group: str, role: str, parameter_name: str) -> Callable:
        def load(param, loaded_weight, weight_name, shard_id=None, expert_id=None, return_success=False):
            what = f"{self.namespace}.{parameter_name} expert={expert_id} shard={shard_id}"
            if shard_id not in _SHARDS or _SHARDS[shard_id][0] != group:
                raise self._fail(f"{parameter_name} cannot receive shard {shard_id!r} ({weight_name})")
            if not isinstance(expert_id, int) or not 0 <= expert_id < self.module.num_experts:
                raise self._fail(f"expert id {expert_id!r} outside [0,{self.module.num_experts}) ({weight_name})")
            projection = _SHARDS[shard_id][1]
            want_shape = self.module.tensor_shape(projection, role)
            if tuple(loaded_weight.shape) != want_shape:
                raise self._fail(f"{what}: shape {tuple(loaded_weight.shape)} != {want_shape}")
            if loaded_weight.dtype != _ROLE_DTYPES[role]:
                raise self._fail(f"{what}: dtype {loaded_weight.dtype} != {_ROLE_DTYPES[role]}; refusing silent cast")
            key = (parameter_name, shard_id, expert_id)
            if key in layer.hf_loaded_experts:
                raise self._fail(f"duplicate load of {what}")
            check_payload(role, loaded_weight, what)
            destination = param.data[expert_id]
            if group == "w13":  # [gate | up] along the output dimension
                width = self.module.intermediate_size
                destination = destination.narrow(-1, 0 if shard_id == "w1" else width, width)
            destination.copy_(loaded_weight)
            layer.hf_loaded_experts.add(key)
            return True if return_success else None

        load.supports_moe_loading = True
        return load

    def _parameter_names(self, layer, group: str) -> list[str]:
        return [f"{group}_{suffix}" for suffix in EXPERT_TENSOR_SUFFIXES.values()
                if hasattr(layer, f"{group}_{suffix}")]

    def process_weights_after_loading(self, layer) -> None:
        if getattr(layer, "hf_processed", False):
            raise self._fail("refusing to process an already-processed layer")
        expected = {
            (name, shard_id, expert)
            for shard_id, (group, _) in _SHARDS.items()
            for name in self._parameter_names(layer, group)
            for expert in range(self.module.num_experts)
        }
        missing = sorted(expected - layer.hf_loaded_experts)
        if missing:
            raise self._fail(f"incomplete expert loading: {len(missing)} missing, first {missing[:6]}")
        for group, k in (("w13", self.module.hidden_size), ("w2", self.module.intermediate_size)):
            device = getattr(layer, f"{group}_qweight").device
            for name in self._parameter_names(layer, group):
                parameter = getattr(layer, name)
                if not parameter.is_contiguous():
                    parameter.data = parameter.data.contiguous()
            layer.register_buffer(f"hf_signs_{group}", signs_for_k(k, self.had_seed).to(device), persistent=False)
            bias_name = f"{group}_{EXPERT_TENSOR_SUFFIXES['bias_corr']}"
            if self.biascorr_mode == "neg_zero" and hasattr(layer, bias_name):
                getattr(layer, bias_name).data.fill_(-0.0)
        layer.hf_processed = True
        self.quant_config.coverage.processed_moe.add(self.namespace)
        if self.linear_mode == "auto":
            logger.info_once("hf_int4_g16: MoE experts always run exact W4A16 (HF_INT4_G16_MODE=auto)")

    def get_fused_moe_quant_config(self, layer):
        # The packed tensors are consumed directly by apply(); vLLM's generic
        # quant config has no meaning for this layout.
        return None

    def _packed(self, layer, group: str) -> PackedExperts:
        def get(role):
            return getattr(layer, f"{group}_{EXPERT_TENSOR_SUFFIXES[role]}", None)

        bias = None if self.biascorr_mode == "off" else get("bias_corr")
        return PackedExperts(
            get("q_pack"), get("u_pack"), get("d_row"), getattr(layer, f"hf_signs_{group}"), bias
        )

    def apply(self, layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts=None, shared_experts_input=None) -> torch.Tensor:
        # The MoERunner runs shared experts itself for non-modular-kernel methods.
        del shared_experts, shared_experts_input
        if not layer.hf_processed:
            raise self._fail("weights were not processed after loading")
        self.quant_config.require_complete_coverage()
        return hf_int4_g16_moe(
            x, topk_weights, topk_ids, self._packed(layer, "w13"), self._packed(layer, "w2"),
            activation=layer.activation.value, backend=self.backend,
        )
