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
