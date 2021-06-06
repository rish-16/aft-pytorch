import torch
from aft_pytorch import AFTFullAttention

layer = AFTFullAttention(
    dim=512,
    tsteps=15,
    hidden_dim=200,
    heads=8
)

# a batch of 64 sequences with 10 timesteps with embed size 512
x = torch.rand(64, 15, 512)
y = layer(x)

print (y.shape) # [64, 15, 512]