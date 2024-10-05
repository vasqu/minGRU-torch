import torch

from src.minGRU_torch.modelling_minGRU import (
    MinGRUConfig,
    MinGRUModel, MinGRUForCausalLM,
    MinGRUBlock, MinGRUAttention
)

config = MinGRUConfig()
config.attention_layers_idx = [2, 3]
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



out = model(
    input_ids=torch.tensor([[1,2,3], [1,2,3]], dtype=torch.long),
    attention_mask=torch.tensor([[1,1,1], [1,1,0]], dtype=torch.long),
    use_cache=True
)
print(out.logits.shape)
