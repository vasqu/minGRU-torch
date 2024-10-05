import math
from packaging import version
from typing import Optional, Tuple, Dict, Any, Union

import torch
from torch import nn
from torch.functional import F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from .configuration_minGRU import MinGRUConfig
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import (
    ROPE_INIT_FUNCTIONS,
    logging,
    DynamicCache,
    PreTrainedModel,
)
from transformers.utils import get_torch_version, is_flash_attn_greater_or_equal_2_10

logger = logging.get_logger(__name__)


class HybridMinGRUAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the minGRU cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and
    `conv_states` and `gru_states` for minGRU cache. Each of these lists has `num_layers` tensors.

    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_key_value_heads, seq_len, head_dim)`.

    For minGRU layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, hidden_size * gru_expansion_factor, conv_kernel_size)`,
    and `gru_states` represents the gur state and has a shape of `(batch_size, 1, hidden_size * gru_expansion_factor)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.has_previous_state = False

        self.conv_states = []
        self.gru_states = []
        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            if i not in config.attention_layers_idx:
                self.conv_states += [
                    torch.zeros(batch_size, config.hidden_size, config.conv_kernel_size, device=device, dtype=dtype)
                ]
                self.gru_states += [
                    torch.zeros(
                        batch_size, 1, config.hidden_size * config.gru_expansion_factor, device=device, dtype=dtype
                    )
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.gru_states += [torch.tensor([[]] * batch_size, device=device)]
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # MinGRU layers don't need the seq_len either way
        if len(self.transformer_layers) == 0:
            return 0

        # Take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0

        # We also allow seq_len checks on empty tensors
        size_idx = -2 if len(self.key_cache[layer_idx].shape) > 2 else -1

        return self.key_cache[layer_idx].shape[size_idx]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMinGRUAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None) -> "DynamicCache":
        raise NotImplementedError("HybridMinGRUAttentionDynamicCache does not have a legacy cache equivalent.")


class MinGRURMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MinGRURMSNorm is equivalent to LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MinGRURotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[MinGRUConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MinGRULinearScalingRotaryEmbedding(MinGRURotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class MinGRUDynamicNTKScalingRotaryEmbedding(MinGRURotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MinGRUAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MinGRUConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_attention_bias)

        self.rotary_emb = MinGRURotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache: Optional[HybridMinGRUAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        query, key, value = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=cache,
        )

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32, especially important for bigger fp16 models
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        cache: Optional[HybridMinGRUAttentionDynamicCache] = None,
    ):
        bsz, q_len, _ = hidden_states.shape

        # Compute QKV
        # [batch, seq_len, hidde_size]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Split combined hidden dims back into respective attention heads
        # [batch, seq_len, hidden_size] --> [batch, seq_len, num_heads, head_dim]
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_heads_kv, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache KV values
        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx)

        return query_states, key_states, value_states


class MinGRUFlashAttention2(MinGRUAttention):
    """
    MinGRU flash attention module. This module inherits from `MinGRUAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache: Optional[HybridMinGRUAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        query, key, value = self._attn_projections_and_rope(
            hidden_states=hidden_states, position_ids=position_ids, cache=cache
        )

        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        # Permute to get the expected shape for Flash Attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 / bfloat16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        input_dtype = query.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.in_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        # Compute attention
        attn_weights = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            q_len,
            dropout=0.0,
            softmax_scale=None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        # Reshape outputs
        attn_output = attn_weights.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class MinGRUSdpaAttention(MinGRUAttention):
    """
    MinGRU attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MinGRUAttention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache: Optional[HybridMinGRUAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if output_attentions:
            logger.warning_once(
                "`MinGRUSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual "
                "implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                cache=cache,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query, key, value = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache=cache
        )

        key = repeat_kv(key, self.num_groups_kv)
        value = repeat_kv(value, self.num_groups_kv)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


MINGRU_ATTENTION_CLASSES = {
    "eager": MinGRUAttention,
    "flash_attention_2": MinGRUFlashAttention2,
    "sdpa": MinGRUSdpaAttention,
}


class MinGRUBlock(nn.Module):
    def __init__(self, config: MinGRUConfig, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.expansion_factor = config.gru_expansion_factor
        self.intermediate_size = config.hidden_size * self.expansion_factor * 2
        self.kernel_size = config.conv_kernel_size
        self.use_conv_bias = config.use_conv_bias
        self.use_gru_bias = config.use_mingru_bias

        self.norm = MinGRURMSNorm(self.hidden_size, eps=config.rms_norm_epsilon)

        self.act = F.silu
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            bias=self.use_conv_bias,
            kernel_size=self.kernel_size,
            groups=self.hidden_size,
            padding=self.kernel_size-1,
        )

        self.to_hidden_and_gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.use_gru_bias)
        self.to_out = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) if self.expansion_factor != 1. else nn.Identity()

    def forward(self, x, attention_mask, cache):
        seq_len = x.shape[1]

        # Managing cache state
        if cache is not None:
            cached_start = not cache.has_previous_state
            cached_forward = not cached_start
        else:
            cached_start = False
            cached_forward = False

        hidden_states = self.norm(x)

        # necessary to avoid influence of padding
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]

        # TODO: add cuda conv option
        # prefill conv states
        if cached_start:
            hidden_states_transposed = hidden_states.transpose(1, 2)
            cache.conv_states[self.layer_idx].copy_(F.pad(hidden_states_transposed, (self.kernel_size - hidden_states_transposed.shape[-1], 0)))

        # reuse conv states or swipe through them
        if cached_forward:
            cache.conv_states[self.layer_idx].copy_(torch.roll(cache.conv_states[self.layer_idx], shifts=-1, dims=-1))
            cache.conv_states[self.layer_idx][:, :, -1] = hidden_states.squeeze(1)
            hidden_states = torch.sum(cache.conv_states[self.layer_idx] * self.conv1d.weight.squeeze(1), dim=-1)
            if self.conv1d.bias is not None:
                hidden_states = hidden_states + self.conv1d.bias
            hidden_states = self.act(hidden_states)
        else:
            hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        # TODO: after linear proj + mask gate with min dtype
        # necessary to avoid influence of padding
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None]

        # similar to mamba we up project before recurrent ops
        hidden_states, gate = self.to_hidden_and_gate(hidden_states).chunk(2, dim=-1)

        # inference mode
        if cached_forward or seq_len == 1:
            hidden_states = self.g(hidden_states)
            gate = gate.sigmoid()

            # TODO: check if we can cache this for the next iteration
            out = torch.lerp(cache.gru_states[self.layer_idx], hidden_states, gate) if cached_forward else (hidden_states * gate)
        # train mode (or on initial forward)
        else:
            # TODO: cleanup
            log_coefficients = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_hidden_states = self.log_g(hidden_states)
            log_values = log_z + log_tilde_hidden_states

            out2 = self.heinsen_associative_scan_log(log_coefficients, log_values)[:, :seq_len]

            # add empty initial states by adding -inf == 0 log space
            dtype = gate.dtype
            gate = F.pad(gate, (0, 0, 1, 0), value=torch.finfo(dtype).min)
            hidden_states = F.pad(hidden_states, (0, 0, 1, 0))

            log_coefficients = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_hidden_states = self.log_g(hidden_states)
            log_values = log_z + log_tilde_hidden_states

            out_tmp = self.heinsen_associative_scan_log(log_coefficients, log_values)
            out = out_tmp[:, -seq_len:]

            res = torch.allclose(out2, out, atol=1e-5)

            # TODO: check if this is correct
            # optionally save last hidden state
            if cached_start:
                cache.gru_states[self.layer_idx].copy_(out[:, -1, :].unsqueeze(1))

            # cut of until last hidden state
            out = out[:, -seq_len:]

        # residual connection
        return self.to_out(out) + x, None

    def g(self, hidden_states):
        return torch.where(hidden_states >= 0, hidden_states + 0.5, hidden_states.sigmoid())

    def log_g(self, hidden_states):
        return torch.where(hidden_states >= 0, (F.relu(hidden_states) + 0.5).log(), -F.softplus(-hidden_states))

    def heinsen_associative_scan_log(self, log_coefficients, log_values):
        a_star = log_coefficients.cumsum(dim=1)
        log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
        log_h = a_star + log_h0_plus_b_star
        return log_h.exp()


class MinGRUMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.expansion_factor = config.mlp_expansion_factor
        self.intermediate_size = self.hidden_size * self.expansion_factor
        self.act_fn = ACT2FN[config.hidden_act]
        self.use_bias = config.use_mlp_bias

        self.norm = MinGRURMSNorm(self.intermediate_size, eps=config.rms_norm_epsilon)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.use_bias)
        self.down_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=self.use_bias)

    def forward(self, x):
        # pre-norm
        hidden_states = self.norm(x)

        # feed forward
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)

        # residual connection
        return hidden_states + x


class MinGRUDecoderBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.attention_layer = layer_idx in config.attention_layers_idx

        if self.attention_layer:
            self.block = MINGRU_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        else:
            self.block = MinGRUBlock(config, layer_idx)
        self.feed_forward = MinGRUMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache: Optional[HybridMinGRUAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if self.attention_layer:
            hidden_states, attn_weights = self.block(
                hidden_states,
                attention_mask,
                position_ids,
                cache,
                output_attentions,
                use_cache,
            )
        else:
            hidden_states, attn_weights = self.block(
                hidden_states,
                attention_mask,
                cache
            )
        return self.feed_forward(hidden_states), attn_weights


class MinGRUPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MinGRUConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MinGRUDecoderBlock"]
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True  # Note: only supports HybridMinGRUAttentionDynamicCache
    _is_stateful = True

    # TODO: check some sane init weight methods
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)


class MinGRUModel(MinGRUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MinGRUDecoderBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self._attn_implementation = config._attn_implementation
        self._uses_attention_layers = len(config.attention_layers_idx) > 0

        self.gradient_checkpointing = False
        self.norm_f = MinGRURMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMinGRUAttentionDynamicCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # We allow empty caches on initial forward
        if past_key_values is None and use_cache:
            past_key_values = HybridMinGRUAttentionDynamicCache(
                config=self.config,
                batch_size=inputs_embeds.shape[0],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

        # LLama based positions
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_block in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                out = self._gradient_checkpointing_func(
                    decoder_block.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                out = decoder_block(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = out[0]

            if output_attentions:
                if out[1] is not None:
                    # Append attentions only of attention layers. MinGRU layers return `None` as the attention weights
                    all_self_attns += (out[1],)

        hidden_states = self.norm_f(hidden_states)

        # Add hidden states from the last block
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridMinGRUAttentionDynamicCache,
        output_attentions: bool,
    ):
        # TODO: fix attention mask creation, we need 2d in GRU and 4d in llama
        if not self._uses_attention_layers:
            if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
                return None
            return attention_mask

        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if self._attn_implementation == "sdpa" and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
                self._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
