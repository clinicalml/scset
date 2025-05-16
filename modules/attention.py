import torch
import torch.nn as nn
import torch.nn.functional as F

"""
# MAB from official SetTransformer repo, doesn't implement masking i.e. can't handle variably sized sets
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V # dim_V is the hidden dimension used in attention / dimension of output from attention
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V) # dim_Q is the hideen dimension of each "seed"
        self.fc_k = nn.Linear(dim_K, dim_V) # dim_k is the hideen dimension of each input cell
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2) # bmm = batch matrix multiplication
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
"""

class MAB(nn.Module):
    """ Multihead Attention Block """

    # reimplementation of MAB that allows for masked inputs
    # credit: Dan Marthaler, https://github.com/juho-lee/set_transformer/issues/12#issuecomment-957994323
    def __init__(self, embed_dim: int, num_heads: int, ln: bool = False):
        super(MAB, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        if ln:
            self.ln0 = nn.LayerNorm(embed_dim)
            self.ln1 = nn.LayerNorm(embed_dim)

        self.fc_o = nn.Linear(embed_dim, embed_dim)

    #TODO: Warning: This Transformer Block is non-standard. Normal would be to use a FFN with a hidden layer. Moreover no dropout is used.

    def forward(self, X, Y, key_padding_mask: torch.Tensor = None, need_weights=False):
        H = X + self.multihead_attn(X, Y, Y, key_padding_mask=key_padding_mask, need_weights=need_weights)[0] #multihead attn returns a tuple, first element is target sequence, second (if used) is attention weights
        H = H if getattr(self, 'ln0', None) is None else self.ln0(H)
        O = H + F.relu(self.fc_o(H))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    """ Set Attention Block (Self Attention Encoder Block)"""

    def __init__(self, dim_hidden, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_hidden, num_heads, ln=ln)

    def forward(self, X, X_mask=None):
        return self.mab(X, X, key_padding_mask=X_mask)


class PMA(nn.Module):
    """ Pooling by Multihead Attention (Tokens -> Seeds) """

    def __init__(self, dim_hidden, num_seeds, num_heads, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_hidden))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(embed_dim=dim_hidden, num_heads=num_heads, ln=ln)
        #self.mab = MAB(dim_hidden_seeds, dim_input, dim_hidden_attn, num_heads, ln=ln) # dim_Q (seed), dim_K (cell input), dim_V (attention hidden/output size)

    def forward(self, X, X_mask=None):
        # if X_mask is None:
        #     print("Warning! No padding mask was provided to PMA. This is not expected for most of our experiments.")
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, X_mask, need_weights=False)


class ISAB(nn.Module):
    """ Induced Set Attention Block (Tokens -> Induced Tokens -> Tokens) """

    def __init__(self, dim_hidden, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_hidden))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(embed_dim=dim_hidden, num_heads=num_heads, ln=ln)
        self.mab1 = MAB(embed_dim=dim_hidden, num_heads=num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
