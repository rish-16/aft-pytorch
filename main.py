import torch
from aft_pytorch import AFTFull, AFT, AFTEncoderBlock, AFTDecoderBlock, MLP, PositionalEncoding

net = AFT(
    vocab_size=10000,
    dim=200,
    hidden_dim=512,
    depth=6,
    p=0.2
)

# a batch of 32 sequences with 10 timesteps with embed size 200
x = torch.rand(64, 15, 200)
y = net(x)

print (y.shape) # [32, 1024]