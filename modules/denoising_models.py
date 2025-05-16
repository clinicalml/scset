import torch
import torch.nn as nn

from .activation import get_activation
from .mlp import MLP
from .positionalembedding import PositionalEmbedding


def build_denoising_model(args, sample_dim):
    if args.encoder == 'pseudo_bulk':
        if args.pseudobulk_project:
            input_condition_dim = args.model_dim
        else:
            input_condition_dim = sample_dim
    elif args.encoder == 'cell_transformer':
        input_condition_dim = args.model_dim #sample_dim
    elif args.encoder == 'set_transformer':
        input_condition_dim = args.model_dim*args.num_seeds
    else:
        input_condition_dim = args.model_dim

    denoise_model = ConditionalDenosingMLP(sample_dim=sample_dim,
                                time_emb_dim=args.time_emb_dim,
                                input_condition_dim=input_condition_dim, 
                                condition_dim=args.model_dim,
                                num_res_blocks=args.num_res_blocks,
                                activation=args.denoising_act)
    return denoise_model


class DenoisingMLP(nn.Module):

    def __init__(self, sample_dim, time_emb_dim, hidden_dim, num_res_blocks, activation='silu') -> None:
        """
        sample_dim: dimension of the input and output
        time_emb_dim: dimension of the time embedding
        hidden_dim: dimension of the condition vector and the hidden layers
        """
        super().__init__()

        self.time_embedding = PositionalEmbedding(time_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            get_activation(activation),
            nn.Linear(time_emb_dim, hidden_dim),
        )
        self.in_adapter = nn.Linear(sample_dim, hidden_dim)
        self.out_adapter = nn.Linear(hidden_dim, sample_dim)

        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim, activation=activation) for _ in range(num_res_blocks)])

    def forward(self, x, tidx):
        """
        x: tensor of shape (batch_size, sample_shape)
        tidx: tensor of shape (batch_size,)
        """
        time_emb = self.time_embedding(tidx)
        time_condition = self.time_mlp(time_emb)

        x = self.in_adapter(x)
        for res_block in self.res_blocks:
            x = res_block(x, time_condition)
        x = self.out_adapter(x)
        return x
    
class ConditionalDenosingMLP(DenoisingMLP):

    def __init__(self, sample_dim, time_emb_dim, input_condition_dim, condition_dim, num_res_blocks, activation='silu') -> None:
        super().__init__(sample_dim, time_emb_dim, condition_dim, num_res_blocks, activation=activation)

        self.condition_mlp = nn.Sequential(
            nn.Linear(input_condition_dim, condition_dim),
            get_activation(activation),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, x, tidx, condition):
        """
        x: tensor of shape (batch_size, sample_shape)
        tidx: tensor of shape (batch_size,)
        condition: tensor of shape (batch_size, condition_shape)
        """
        time_emb = self.time_embedding(tidx)
        time_condition = self.time_mlp(time_emb)

        condition = self.condition_mlp(condition)

        joint_condition = time_condition + condition

        x = self.in_adapter(x)
        for res_block in self.res_blocks:
            x = res_block(x, joint_condition)
        x = self.out_adapter(x)
        return x


# AdaLN proposed in https://arxiv.org/pdf/2212.09748 and adapted from https://github.com/LTH14/mar
class ResidualBlock(nn.Module):

    def __init__(self, sample_dim, condition_dim, activation='silu') -> None:

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(sample_dim, sample_dim),
            get_activation(activation),
            nn.Linear(sample_dim, sample_dim),
        )
        
        self.modulation = nn.Sequential(
            get_activation(activation),
            nn.Linear(condition_dim, 3*sample_dim),
        )

        self.layer_norm = nn.LayerNorm(sample_dim)

    def forward(self, x, condition):
        """
        x: tensor of shape (batch_size, sample_shape)
        condition: tensor of shape (batch_size, condition_shape)
        """
        bias, scale1, scale2 = self.modulation(condition).chunk(3, dim=-1)
        r = self.layer_norm(x)
        r = r * (1 + scale1) + bias
        r = self.mlp(r)
        return x + r * scale2

# Flow models are adapted from https://bm371613.github.io/conditional-flow-matching/, adapted to be conditional
# We only use the loss function, of the 3 functions below

class ConditionalFlowNeuralNetwork(DenoisingMLP):

    def __init__(self, n_features, cond_dim, time_embedding_size=8, n_blocks=5):
        super().__init__()
        self.time_embedding = PositionalEmbedding(time_emb_dim)

        self.cond_dim = cond_dim

        self.condition_mlp = nn.Sequential(
            nn.Linear(input_condition_dim, condition_dim),
            get_activation(activation),
            nn.Linear(condition_dim, condition_dim),
        )

        input_dim = n_features + time_embedding_size + cond_dim
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(input_dim, n_features)

    def forward(self, X, time, cond):
        # X: (B, n_features), time: (B,), cond: (B, cond_dim)
        t_embed = self.time_embedding(time)  # (B, time_embedding_size)
        X = torch.cat([X, t_embed, cond], dim=1)
        for block in self.blocks:
            X = X + block(X)
        X = self.final(X)
        return X
    
def conditional_flow_matching_loss(model, x0, x1, t, cond):
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    target_v = x1 - x0
    pred_v = model(xt, t, cond)
    return ((pred_v - target_v)**2).mean()


class ZeroToOneTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)

    def forward(self, t):
        emb = self.freqs * t[..., None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb