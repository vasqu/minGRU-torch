from transformers import AutoTokenizer

from src.minGRU_torch.modelling_minGRU import (
    MinGRUConfig,
    MinGRUModel, MinGRUForCausalLM,
    MinGRUBlock, MinGRUAttention
)

config = MinGRUConfig()
model = MinGRUForCausalLM(config)
print(model)
print(model.num_parameters())

gru_block = MinGRUBlock(config, 0)
attn_block = MinGRUAttention(config, 0)

for name, block in zip(["gru", "attn"], [gru_block, attn_block]):
    block_params = 0
    for param in block.parameters():
        block_params += param.numel()
    print(f'{name}: {block_params}')


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', clean_up_tokenization_spaces=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
input_ids = tokenizer(["Hey how are you doing?", "What is life?"], padding=True, return_tensors="pt")

out = model.generate(**input_ids, max_new_tokens=10, use_cache=True)
print(tokenizer.batch_decode(out))
