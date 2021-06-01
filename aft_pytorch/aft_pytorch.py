import torch
from torch import nn, einsum
import torch.nn.functional as F

class AFTAttention(nn.Module):
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

class AFT(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()

    def forward(self, x):
        pass