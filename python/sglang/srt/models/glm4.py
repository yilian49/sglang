# SPDX-License-Identifier: Apache-2.0
# Adapted from GLM-4 (vLLM) → SGLang
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Glm4Config

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.utils import add_prefix, make_layers
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class Glm4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act.lower() != "silu":
            raise ValueError("Only silu activation is supported for GLM4.")
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        gate_up_out, _ = self.gate_up(x)
        x = self.act(gate_up_out)
        x, _ = self.down(x)
        return x


class Glm4Attention(nn.Module):
    def __init__(
        self,
        config: Glm4Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_heads = num_heads
        assert self.total_heads % tp_size == 0
        self.heads = self.total_heads // tp_size
        self.total_kv = num_kv_heads
        assert (self.total_kv % tp_size == 0) or (tp_size % self.total_kv == 0)
        self.kv_heads = max(1, self.total_kv // tp_size)
        self.head_dim = hidden_size // num_heads
        self.qkv = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_heads,
            self.total_kv,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.heads,
            self.head_dim,
            self.head_dim ** -0.5,
            num_kv_heads=self.kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv_out, _ = self.qkv(hidden_states)
        q, k, v = qkv_out.split([self.heads * self.head_dim] * 3, dim=-1)
        q, k = self.rotary(positions, q, k)
        attn_out = self.attn(q, k, v, forward_batch)
        out, _ = self.o_proj(attn_out)
        return out


class Glm4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Glm4Attention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Glm4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_norm(hidden_states)
        else:
            hidden_states, residual = self.input_norm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states, residual = self.post_mlp_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_batch)
        hidden_states = self.post_mlp_norm(hidden_states)
        return hidden_states, residual


class Glm4Model(nn.Module):
    def __init__(
        self,
        config: Glm4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp = get_pp_group()
        if self.pp.is_first_rank:
            self.embed = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            from sglang.srt.layers.utils import PPMissingLayer
            self.embed = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, pfx: Glm4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=pfx,
            ),
            pp_rank=self.pp.rank_in_group,
            pp_size=self.pp.world_size,
            prefix=add_prefix("model.layers", prefix),
        )
        if self.pp.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            from sglang.srt.layers.utils import PPMissingLayer
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp.is_first_rank:
            hidden = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
            residual = None
        else:
            assert pp_proxy is not None
            hidden = pp_proxy["hidden_states"]
            residual = pp_proxy["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden, residual = layer(positions, hidden, forward_batch, residual)

        if not self.pp.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden, "residual": residual})

        hidden, _ = self.norm(hidden, residual)
        return hidden


class Glm4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Glm4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp = get_pp_group()
        self.model = Glm4Model(config, quant_config, prefix=add_prefix("model", prefix))
        if self.pp.is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            from sglang.srt.layers.utils import PPMissingLayer
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy: Optional[PPProxyTensors] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        hidden = self.model(input_ids, positions, forward_batch, inputs_embeds, pp_proxy)
        # if capturing embeddings only:
        if get_embedding:
            return hidden  # type: ignore
        logits = self.logits_processor(self.lm_head, hidden, forward_batch)
        return logits


# Register the entry class so the server can discover it:
EntryClass = [Glm4ForCausalLM]
