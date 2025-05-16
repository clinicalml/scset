import torch
import torch.nn as nn
from torch import LongTensor

class PositionalEmbedding(nn.Module):

    def __init__(self, model_dim : int, max_width_or_height : int = 1200, temperature : float = 10000.):
        super(PositionalEmbedding, self).__init__()

        assert model_dim % 2 == 0, 'Embedding dimension must be multiple of 2 for 1D positional embedding'

        possible_positions = torch.arange(max_width_or_height, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim , 2, dtype=torch.float32) * - (torch.log(torch.tensor(temperature)) / model_dim))
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        cos = torch.cos(pos)

        self.register_buffer('positional_embeddings', torch.zeros(max_width_or_height, model_dim))

        self.positional_embeddings[:, 0::2] = sin
        self.positional_embeddings[:, 1::2] = cos

    # positions = (batch_size, seq_len, 2)
    def forward(self, positions : LongTensor):
        """
        Computes positional embeddings corresponding to 2D input positions
        Args:
            x: (..., model_dim) tensor
            positions: (...) tensor tensor of positions
        Returns:
            (..., model_dim) tensor of positional embeddings
        """
        pos_emb = self.positional_embeddings[positions]
        return pos_emb