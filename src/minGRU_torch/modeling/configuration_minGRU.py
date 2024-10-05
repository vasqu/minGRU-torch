"""MinGRU configuration"""
from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class MinGRUConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MinGRUModel`]. It is used to instantiate a MinGRU
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a hybrid configuration of MinGRU and Attention layers of roughly 140m parameters.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the MinGRU model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MinGRUModel`].
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        gru_expansion_factor (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the intermediate size in the minGRU layer.
        conv_kernel_size (`int`, *optional*, defaults to 4):
            Size of the convolution kernel.
        mlp_expansion_factor (`int`, *optional*, defaults to 4):
            Expanding factor used to determine the intermediate size in the mlp layer.
        attention_head_dim (`int`, *optional*, defaults to 64):
            Multi-head attention's head dimension.
        num_attention_heads (`int`, *optional*, defaults to 18):
            The number of heads in multi-head attention.
        num_key_value_heads (`int`, *optional*, defaults to 18):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `attention_num_key_value_heads=attention_num_heads`, the model will use Multi Head Attention (MHA), if
            `attention_num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `attention_num_heads`.
        use_mingru_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["to_hidden_and_gate", "to_out"] of the minGRU block.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the minGRU block.
        use_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in the qkv projection of the attention block.
        use_mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in the feed forward projections.
        attention_layers_idx (`List[int]`, *optional*, defaults to `[4, 9]`):
            The specific layers that exchange the minGRU block with the attention equivalent.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the model.
        rms_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the rms normalization layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation used for initializing any torch weights/parameters.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.

            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
    """

    model_type = "minGRU"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50280,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        hidden_size=768,
        gru_expansion_factor=2,
        conv_kernel_size=4,
        mlp_expansion_factor=4,
        attention_head_dim=64,
        num_attention_heads=18,
        num_key_value_heads=18,
        use_mingru_bias=False,
        use_conv_bias=True,
        use_attention_bias=False,
        use_mlp_bias=False,
        attention_layers_idx=None,
        num_hidden_layers=12,
        rms_norm_epsilon=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        use_cache=True,
        **kwargs,
    ):
        attention_layers_idx = [4, 9] if attention_layers_idx is None else attention_layers_idx

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.gru_expansion_factor = gru_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.mlp_expansion_factor = mlp_expansion_factor
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = attention_head_dim if attention_head_dim is not None else self.hidden_size // self.num_attention_heads
        self.use_mingru_bias = use_mingru_bias
        self.use_conv_bias = use_conv_bias
        self.use_attention_bias = use_attention_bias
        self.use_mlp_bias = use_mlp_bias
        self.attention_layers_idx = attention_layers_idx
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_epsilon = rms_norm_epsilon
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        rope_config_validation(self)
        self._attention_layers_idx_validation()

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

    def _attention_layers_idx_validation(self):
        """
        Validate the `attention_layers_idx` configuration.
        """
        if isinstance(self.attention_layers_idx, list) and len(self.attention_layers_idx) == 0:
            return

        if not isinstance(self.attention_layers_idx, List) and all(
                isinstance(x, int) for x in self.attention_layers_idx
        ):
            raise ValueError(
                "`attention_layers_idx` must be a list of integers indicating the attention layers, "
                f"got {self.attention_layers_idx}"
            )

        if min(self.attention_layers_idx) < 0 or max(self.attention_layers_idx) >= self.num_hidden_layers:
            raise ValueError(
                "`attention_layers_idx` has out-of-range indices, "
                f"got {self.attention_layers_idx}, but expected indices in {list(range(self.num_hidden_layers))}"
            )
