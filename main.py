import torch
from aft_pytorch import AFTFullAttention

layer = AFTFullAttention(
    seqlen=15,
    dim=512,
    hidden_dim=200,
    heads=8
)

# a batch of 64 sequences with 15 timesteps with embed size 512
x = torch.rand(64, 15, 512)
y = layer(x)

print (y.shape) # [64, 15, 512]