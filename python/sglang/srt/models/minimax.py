from typing import Optional

from torch import nn
import torch
from transformers import MiniMaxConfig

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict

from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK


class MiniMaxText01MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
    

class MiniMaxText01MoE(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: MiniMaxConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "moe",
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
        )
        self.hidden_size = config.hidden_size
        
        self.quant_config = quant_config
        self.activation = config.hidden_act

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # self.gate.weight.weight_loader = MiniMaxText01MoE.gate_weight_loader
        experts_type = get_moe_impl_class()
        extra_kwargs = {}
        if experts_type.__name__ == "FusedMoE":
            quant_config_name = (
                quant_config.get_name() if quant_config is not None else None
            )
        self.experts = experts_type(
            num_experts=config.num_local_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            activation=self.activation,
            prefix=add_prefix("experts", prefix),
            **extra_kwargs,
        )
        return

    # @staticmethod
    # def gate_weight_loader(param: nn.Parameter,
    #                        loaded_weight: torch.Tensor) -> None:
    #     assert param.size() == loaded_weight.size()
    #     param.data.copy_(loaded_weight.to(torch.float32))
    #     return

    def forward(
            self,
            hidden_states: torch.Tensor,
            should_allreduce_rusion: bool = False,
    ) -> torch.Tensor:
        #TODO: implement deepep here
        num_tokens, hidden_dim = hidden_states.shape

        router_logits, _ = self.gate(hidden_states)
        topk_output = self.top_k(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        #TODO: Implement shared experts if necessary here

        if self.tp_size > 1 and not should_allreduce_rusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)
    
class MiniMaxText01LinearAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_hidden_layers: int,
            block_size: int,
            head_dim: int,
            hidden_act: str,
            layer_id: int,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        # TODO: support DP Attention
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.block_size = block_size
        self.hidden_act = hidden_act
        self.layer_id = layer_id
        self.prefix = prefix

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            self.num_heads,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=prefix,
        )
        self.output_gate = ColumnParallelLinear(
            hidden_size,
        )

        attn_backend = global_server_args_dict.get("attention_backend")



    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = torch.tensor(get_slopes(n_attention_heads),
                              dtype=torch.float32).reshape(
                                  n_attention_heads, 1, 1)
        return slopes