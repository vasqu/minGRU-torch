from transformers import AutoTokenizer
from minGRU_torch import MinGRUConfig, MinGRUForCausalLM

# init model where we exchange layer idx 4 and 9 with attention equivalents
config = MinGRUConfig(
    attention_layers_idx=[4, 9],
    num_hidden_layers=12,
)
model = MinGRUForCausalLM(config)

# it is important to use left padding to avoid any influence of padding tokens
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', clean_up_tokenization_spaces=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

# it will produce trash as we have no pretraining :)
input_ids = tokenizer(["Hey how are you doing?", "What is life?"], padding=True, return_tensors="pt")
out = model.generate(**input_ids, max_new_tokens=10, use_cache=True)
print(tokenizer.batch_decode(out))


import torch
from minGRU_torch import MinGRUConfig, MinGRUBlock

# random input
x = torch.randn(size=(4, 10, 256))

# construct a small minGRU block
config = MinGRUConfig(
    hidden_size=256,
    gru_expansion_factor=2,
    conv_kernel_size=4,
)
minGRU_block = MinGRUBlock(config, layer_idx=0)

# output is at 0 as we need to output None attn weights at 1
out = minGRU_block(x)[0]

# ensure it worked
assert x.shape == out.shape
