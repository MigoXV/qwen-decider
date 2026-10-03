# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM v0.14.1 qwen3_next and v0.17.1 qwen3_5.
# Copyright contributors to the vLLM project; Copyright 2025 The Qwen Team.
"""Dense text-only Qwen3.5 on the unmodified vLLM 0.14.1 engine."""

from itertools import islice

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextRMSNorm,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
)


class GatedDeltaNet(Qwen3NextGatedDeltaNet):
    def fix_query_key_value_ordering(self, qkvz, ba):
        # Qwen3.5 packs whole Q/K/V/Z projections, not Qwen3Next's head groups.
        q, k, v, z = qkvz.split([self.key_dim, self.key_dim, self.value_dim, self.value_dim], -1)
        b, a = ba.chunk(2, -1)
        return (
            q.reshape(-1, self.num_k_heads, self.head_k_dim),
            k.reshape(-1, self.num_k_heads, self.head_k_dim),
            v.reshape(-1, self.num_v_heads, self.head_v_dim),
            z.reshape(-1, self.num_v_heads, self.head_v_dim),
            b.contiguous(),
            a.contiguous(),
        )


class Decoder(Qwen3NextDecoderLayer):
    def __init__(self, vllm_config, layer_type, prefix):
        nn.Module.__init__(self)
        c = vllm_config.model_config.hf_config
        self.layer_type, self.layer_idx = layer_type, extract_layer_index(prefix)
        kwargs = dict(
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
        )
        if layer_type == "linear_attention":
            kwargs["quant_config"] = None  # recurrent branch stays BF16/FP16
            self.linear_attn = GatedDeltaNet(c, prefix=prefix + ".linear_attn", **kwargs)
        else:
            self.self_attn = Qwen3NextAttention(c, prefix=prefix + ".self_attn", **kwargs)
        self.mlp = Qwen3NextMLP(
            c.hidden_size,
            c.intermediate_size,
            c.hidden_act,
            quant_config=vllm_config.quant_config,
            prefix=prefix + ".mlp",
        )
        self.input_layernorm = Qwen3NextRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.layer_scale = False


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0})
class Backbone(nn.Module):
    def __init__(self, vllm_config, prefix="model"):
        super().__init__()
        c = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(c.vocab_size, c.hidden_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            c.num_hidden_layers,
            lambda prefix: Decoder(vllm_config, c.layer_types[extract_layer_index(prefix)], prefix),
            prefix=prefix + ".layers",
        )
        self.norm = Qwen3NextRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], c.hidden_size
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden, residual = layer(hidden, residual, positions)
        return self.norm(hidden, residual)[0]


class QwenDeciderForCausalLM(nn.Module, HasInnerState, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3NextForCausalLM.get_mamba_state_dtype_from_config.__func__
    )
    get_mamba_state_shape_from_config = classmethod(
        Qwen3NextForCausalLM.get_mamba_state_shape_from_config.__func__
    )

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        if (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("v1 supports a single GPU only")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("prefix caching is not supported")
        if vllm_config.cache_config.cache_dtype != "auto":
            raise ValueError("quantized KV cache is not supported")
        self.config = vllm_config.model_config.hf_config
        # Text-only mRoPE has identical positions on all axes: ordinary partial RoPE is equivalent.
        self.config.rope_parameters = {
            k: v
            for k, v in self.config.rope_parameters.items()
            if k not in {"mrope_section", "mrope_interleaved"}
        }
        self.model = Backbone(vllm_config=vllm_config)
        if not self.config.tie_word_embeddings:
            raise ValueError("v1 requires tied embedding/head weights")
        self.lm_head = self.model.embed_tokens
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
    ):
        if positions.ndim == 2:
            positions = positions[0]
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded, pending = set(), {}
        seen_shards = {}
        mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for name, tensor in weights:
            name = name.replace("model.language_model.", "model.")
            if name == "lm_head.weight":
                continue  # explicitly tied to embed_tokens
            matched = False
            for target, parts in [
                ("in_proj_qkvz", ["in_proj_qkv", "in_proj_z"]),
                ("in_proj_ba", ["in_proj_b", "in_proj_a"]),
            ]:
                for part in parts:
                    if "." + part + "." in name:
                        key = name.replace("." + part + ".", "." + target + ".")
                        pending.setdefault(key, {})[part] = tensor
                        if len(pending[key]) == len(parts):
                            param = params[key]
                            param.weight_loader(
                                param, torch.cat([pending[key][p] for p in parts], dim=0)
                            )
                            loaded.add(key)
                            del pending[key]
                        matched = True
                        break
                if matched:
                    break
            if matched:
                continue
            for target, source, shard in mapping:
                if "." + source + "." in name:
                    key = name.replace("." + source + ".", "." + target + ".")
                    param = params[key]
                    param.weight_loader(param, tensor, shard)
                    seen_shards.setdefault(key, set()).add(shard)
                    loaded.add(key)
                    break
            else:
                if name not in params:
                    raise ValueError(f"unexpected weight: {name}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
        for key, shards in seen_shards.items():
            expected = {"q", "k", "v"} if ".qkv_proj." in key else {0, 1}
            if shards != expected:
                raise ValueError(f"incomplete packed weight {key}: {shards}")
        # compressed-tensors creates optional FP8 attention scale placeholders
        # even for an unquantized KV cache. vLLM finalizes these to unity.
        runtime_scales = {
            name
            for name in params
            if any(
                name.endswith(".self_attn.attn." + suffix)
                for suffix in ("q_scale", "k_scale", "v_scale", "prob_scale")
            )
        }
        missing = set(params) - loaded - runtime_scales
        if pending or missing:
            raise ValueError(f"incomplete weights: {missing}, partial={list(pending)}")
        return loaded
