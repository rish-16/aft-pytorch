import torch, math
from torch import nn, einsum
import torch.nn.functional as F    

class AFTFull(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full

        Number of heads is 1 as done in the paper
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        Yt = torch.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

class AFTSimple(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        
        Number of Heads is 1 as done in the paper.
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)

        '''
        From the paper
        '''
        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

class AFTLocal(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64, s=256):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        s: the window size used for AFT-Local in the paper

        Number of heads is 1 as done in the paper
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        self.max_seqlen = max_seqlen
        self.s = s
        nn.init.xavier_uniform_(self.wbias)


    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)
        self.wbias = nn.Parameter(torch.Tensor([
            [self.wbias[i][j] if math.fabs(i-j) < self.s else 0 for j in range(self.max_seqlen)] 
            for i in range(self.max_seqlen)
            ]))
        temp_wbias = self.wbias[:T, :T].unsqueeze(0) # sequences can still be variable length

        '''
        From the paper
        '''
        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        Yt = torch.mul(Q_sig, weighted)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt

class AFTConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

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