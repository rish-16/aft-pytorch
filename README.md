# aft-pytorch
Unofficial PyTorch implementation of **Attention Free Transformer**'s layers by [Zhai](https://twitter.com/zhaisf?lang=en), et al. [[abs](https://openreview.net/forum?id=pW--cu2FCHY), [pdf](https://arxiv.org/pdf/2105.14103.pdf)] from Apple Inc.

<img src="https://github.com/rish-16/aft-pytorch/raw/main/pic.png" width=650>

## Installation
You can install `aft-pytorch` via `pip`:

```bash
pip install aft-pytorch
```

## Usage
You can import the **AFT-Full** or **AFT-Simple** layer (as described in the paper) from the package like so:

### `AFTFull`
```python
from aft_pytorch import AFTFull

layer = AFTFull(
    max_seqlen=20,
    dim=512,
    hidden_dim=64
)

# a batch of sequences with 10 timesteps of length 512 each
x = torch.rand(32, 10, 512)
y = layer(x) # [32, 10, 512]
```

### `AFTSimple`
```python
from aft_pytorch import AFTSimple

layer = AFTSimple(
    max_seqlen=20,
    dim=512,
    hidden_dim=64
)

# a batch of sequences with 10 timesteps of length 512 each
x = torch.rand(32, 10, 512)
y = layer(x) # [32, 10, 512]
```
### `AFTLocal`
```python
from aft_pytorch import AFTLocal

layer = AFTLocal(
    max_seqlen=20,
    dim=512,
    hidden_dim=64
)

# a batch of sequences with 10 timesteps of length 512 each
x = torch.rand(32, 10, 512)
y = layer(x) # [32, 10, 512]
```

> This layer wrapper is a 'plug-and-play' with your existing networks / Transformers. You can swap out the Self-Attention layer with the available layers in this package with minimal changes.

## TODO
- Add full AFT architecture
- Add variants like, `AFTConv`, `AFTLocal`

## Contributing
If you like this repo, please leave a star! If there are any amends or suggestions, feel free to raise a PR/issue.

## Credits
```
@misc{attention-free-transformer,
title = {An Attention Free Transformer},
author = {Shuangfei Zhai and Walter Talbott and Nitish Srivastava and Chen Huang and Hanlin Goh and Ruixiang Zhang and Josh Susskind},
year = {2021},
URL = {https://arxiv.org/pdf/2105.14103.pdf}
}
```

## License
[MIT](https://github.com/rish-16/aft-pytorch/blob/main/LICENSE)
