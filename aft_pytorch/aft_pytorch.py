import torch
from torch import nn, einsum
import torch.nn.functional as F    

class AFTFull(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        
        self.wbias = nn.Parameter(torch.rand(hidden_dim)) # learnable pair-wise position bias
        self.to_out = nn.Linear(hidden_dim, dim) if dim != hidden_dim else nn.Identity()

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)

        Q_sig = torch.sigmoid(Q)
        numer = torch.exp(K + self.wbias)
        denom = numer.sum(0)
        weighted = torch.mul(numer, V).sum(0) / denom
        Yt = torch.mul(Q_sig, weighted)
        Yt = self.to_out(Yt)

        return Yt

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dp=0.1):
        super().__init__()
        self.l1 = nn.Linear(dim, hidden_dim)
        self.g1 = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, dim)
        self.d1 = nn.Dropout(dp)

    def forward(self, x):
        x = self.l1(x)
        x = self.g1(x)
        x = self.d1(x)
        out = self.l2(x)

        return out

class AFTEncoderBlock(nn.Module):
    def __init__(self, dim, hidden_dim, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = AFTFull(dim, hidden_dim)
        self.mlp = MLP(dim, hidden_dim)
        self.d1 = nn.Dropout(p)
        self.d2 = nn.Dropout(p)

    def forward(self, x):
        x = self.ln(x)
        x = self.attn(x) + x
        x = self.d1(x)
        x = self.ln(x)
        x = self.mlp(x)
        out = self.d2(x) + x

        return out

class AFTDecoderBlock(nn.Module):
    def __init__(self, dim, hidden_dim, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn1 = AFTFull(dim, hidden_dim)
        self.attn2 = AFTFull(dim, hidden_dim)
        self.mlp = MLP(dim, hidden_dim, p=p)
        self.d1 = nn.Dropout(p)
        self.d2 = nn.Dropout(p)
        self.d3 = nn.Dropout(p)

    def forward(self, x):
        x = self.ln(x)
        x = self.attn1(x)
        x = self.d1(x) + x
        x = self.ln(x)
        x = self.attn2(x)
        x = self.dropout_2(x) + x
        x = self.ln(x)
        x = self.mlp(x)
        out = self.d3(x) + x

        return out

class AFT(nn.Module):
    def __init__(self, dim, hidden_dim, depth=6):
        super().__init__()
        self.layers = nn.ModuleList()
        self.enc = nn.ModuleList([AFTEncoderBlock(dim, hidden_dim) for _ in range(depth)])
        self.dec = nn.ModuleList([AFTDecoderBlock(dim, hidden_dim) for _ in range(depth)])
        self.out = nn.Linear() # TODO: create output layer

    def forward(self, x):
        x = self.enc(x)
        out = self.dec(x)

        return out