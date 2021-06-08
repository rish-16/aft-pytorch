import torch, math
from torch import nn, einsum
import torch.nn.functional as F    

class AFTFullAttention(nn.Module):
    def __init__(self, seqlen, dim, hidden_dim, heads):
        super().__init__()
        '''
        seqlen: the number of tokens in a sequence
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        heads: the number of AFT-Full heads
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.to_q = nn.Linear(dim, hidden_dim * heads)
        self.to_k = nn.Linear(dim, hidden_dim * heads)
        self.to_v = nn.Linear(dim, hidden_dim * heads)
        self.wbias = nn.Parameter(torch.rand(self.heads, seqlen, seqlen))
        self.to_out = nn.Linear(heads * hidden_dim, dim) if dim != hidden_dim else nn.Identity()

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, self.heads, T, self.hidden_dim)
        K = self.to_k(x).view(B, self.heads, T, self.hidden_dim)
        V = self.to_v(x).view(B, self.heads, T, self.hidden_dim)

        '''
        From the paper
        '''
        numer = torch.exp(self.wbias).unsqueeze(0) @ torch.exp(K)
        denom = numer.sum(0)

        Q_sig = torch.sigmoid(Q)
        weighted = torch.mul(numer, V).sum(0) / denom
        Yt = torch.mul(Q_sig, weighted)
        Yt = Yt.view(B, T, self.heads * self.hidden_dim)
        Yt = self.to_out(Yt)

        return Yt

'''
Taken from the PyTorch docs for Positional Embeddings only
'''
# class PositionalEncoding(nn.Module):
#     def __init__(self, dim, p=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=p)

#         pe = torch.zeros(max_len, dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# class MLP(nn.Module):
#     def __init__(self, dim, hidden_dim, dp=0.1):
#         super().__init__()
#         self.l1 = nn.Linear(dim, hidden_dim)
#         self.g1 = nn.GELU()
#         self.l2 = nn.Linear(hidden_dim, dim)
#         self.d1 = nn.Dropout(dp)

#     def forward(self, x):
#         x = self.l1(x)
#         x = self.g1(x)
#         x = self.d1(x)
#         out = self.l2(x)

#         return out        

# class AFTEncoderBlock(nn.Module):
#     def __init__(self, dim, hidden_dim, p=0.1):
#         super().__init__()
#         self.ln = nn.LayerNorm(dim)
#         self.attn = AFTFull(dim, hidden_dim)
#         self.mlp = MLP(dim, hidden_dim)
#         self.d1 = nn.Dropout(p)
#         self.d2 = nn.Dropout(p)

#     def forward(self, x):
#         x = self.ln(x)
#         x = self.attn(x) + x
#         x = self.d1(x)
#         x = self.ln(x)
#         x = self.mlp(x)
#         out = self.d2(x) + x

#         return out

# class AFTDecoderBlock(nn.Module):
#     def __init__(self, dim, hidden_dim, p=0.1):
#         super().__init__()
#         self.ln = nn.LayerNorm(dim)
#         self.attn1 = AFTFull(dim, hidden_dim)
#         self.attn2 = AFTFull(dim, hidden_dim)
#         self.mlp = MLP(dim, hidden_dim, p=p)
#         self.d1 = nn.Dropout(p)
#         self.d2 = nn.Dropout(p)
#         self.d3 = nn.Dropout(p)

#     def forward(self, x):
#         x = self.ln(x)
#         x = self.attn1(x)
#         x = self.d1(x) + x
#         x = self.ln(x)
#         x = self.attn2(x)
#         x = self.dropout_2(x) + x
#         x = self.ln(x)
#         x = self.mlp(x)
#         out = self.d3(x) + x

#         return out

# class AFT(nn.Module):
#     def __init__(self, vocab_size, dim, hidden_dim, enc=None, dec=None, depth=6, p=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.pos_embed = PositionalEncoding(dim, p=p)
#         self.embed = nn.Embedding(vocab_size, dim)
#         self.enc = enc if enc else nn.ModuleList([AFTEncoderBlock(dim, hidden_dim) for _ in range(depth)])
#         # self.dec = nn.ModuleList([AFTDecoderBlock(dim, hidden_dim) for _ in range(depth)])
#         self.dec = dec if dec else nn.Linear(dim, vocab_size)
#         self.dim = dim

#     def forward(self, x):
#         x = self.embed(x) * math.sqrt(self.dim)
#         print (x.shape)
#         x = self.pos_embed(x)
#         x = self.enc(x)
#         out = self.dec(x)

#         return out