# Hybrid minGRU x Attention

## Introduction

This is a highly experimental implementation of minGRU [[1]](#citation) that is compatible 
with `transformers` [[2]](#citation). The core parts of minGRU are based on [[3]](#citation).

Note:
- It is only compatible with the pinned `transformers` version as custom cache support is not given otherwise.
- Attention with RoPE is supported with `eager`, `flash_attention_2`, and `sdpa`.
- Initial state support is not given yet (in parallel mode).
- Potentially deviation from original [[1]](#citation) implementation as I only add the residual connection after 
the convolution + minGRU ops instead of after each one separately.

Don't expect this to work perfectly! I've done this quickly and scrapped parts together - it's a toy project through and through.


## Installation
I won't distribute a pypi package, but you can use it as package by cloning the repo and installing it at root:
```bash
git clone https://github.com/vasqu/minGRU-torch.git
cd minGRU-torch
pip install .
```
I've semi-implemented the path for the cuda causal convolution. Thus, you could install the
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) package separately (for safety with version 1.2.0) which will then be utilized automatically.


## Usage
### MinGRU Block
```python
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

# output is at 0 as we need to output None at 1 for compatibility reasons 
out = minGRU_block(x)[0]

# ensure it worked
assert x.shape == out.shape
```

### MinGRU Language Model
```python
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
```


## Citation

```bibtex
[1]
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}

[2]
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```

[3] No citation, but thanks to lucidrains for his repo [over here](https://github.com/lucidrains/minGRU-pytorch)
which provides most fundamental implementations for minGRU :)
