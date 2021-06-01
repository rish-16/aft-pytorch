import torch
from aft_pytorch import AFTAttention

layer = AFTAttention(
    dim=2,
    hidden_dim=64
)

# a batch of 32 sequences with 10 timesteps with embed size 512
x = torch.rand(32, 3, 2)
y = layer(x)

print (y.shape) # [32, 1024]