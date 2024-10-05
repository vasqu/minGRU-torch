import torch

from src.minGRU_torch.modelling_minGRU import MinGRUConfig, MinGRUModel

config = MinGRUConfig()
config.attention_layers_idx = [2, 3]
model = MinGRUModel(config)


model(
    input_ids=torch.tensor([[1,2,3], [1,2,3]], dtype=torch.long), use_cache=True
)
