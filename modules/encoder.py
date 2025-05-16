import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from .activation import get_activation
from .attention import PMA, SAB

def build_encoder_model(args, sample_dim):
    if args.encoder == 'cell_transformer':
        return CellTransformer(input_dim=sample_dim,
                    model_dim=args.model_dim,
                    num_blocks=args.num_blocks,
                    num_heads=args.num_heads,
                    feedforward_dim=args.feedforward_dim,
                    dropout=args.dropout,
                    activation=args.activation)

    elif args.encoder == 'set_transformer':
        return SetTransformer(input_dim=sample_dim,
                    model_dim=args.model_dim,
                    output_dim=args.model_dim,
                    num_seeds=args.num_seeds,
                    num_heads=args.num_heads,
                    activation=args.activation,
                    ln=True)

    elif args.encoder == 'abmil':
        return ABMIL(input_dim=sample_dim,
                    model_dim=args.model_dim,
                    num_layers=args.num_hidden_layers,
                    gated_attn=True,
                    activation=args.activation)

    elif args.encoder == 'pseudo_bulk_classifier':
        return PseudoBulkClassifier(dim_input=sample_dim,
                    dim_output=args.model_dim,
                    num_layers=args.num_hidden_layers,
                    activation=args.activation)

    elif args.encoder == 'pseudo_bulk':
        return PseudoBulk(input_dim=sample_dim,
                    model_dim=args.model_dim,
                    libnorm=args.libnorm,
                    lognorm=args.lognorm,
                    project=args.pseudobulk_project)

    elif args.encoder == 'random_embedding':
        return RandomEmbeddingModel(model_dim=args.model_dim)

    else:
        raise ValueError(f"encoder {args.encoder} not supported")
    
    
class CellTransformer(nn.Module):

    def __init__(self, input_dim, model_dim, num_blocks, num_heads, feedforward_dim, dropout=0.1, activation='gelu'):
        super().__init__()

        self.embedding = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(model_dim, input_dim)
        self.output_layer.weight.data = self.embedding.weight.T # tie weights to input embedding
        #self.output_layer.weight.requires_grad = False

    def forward(self, X, X_mask=None, return_CLS_only=True):
        """
        x: (batch_size, seq_len, input_dim)
        mask: (batch_size, seq_len)
        Returns: if return_CLS_only, (batch_size, input_dim), else (batch_size, n_cells+1, input_dim)
        """
        X = self.embedding(X)

        cls_tokens = self.cls_token.expand(X.shape[0], 1, -1)
        X = torch.cat([cls_tokens, X], dim=1)

        if X_mask is not None:
            cls_mask = torch.zeros(X_mask.shape[0], 1, dtype=torch.bool, device=X_mask.device) #all non-masked
            X_mask = torch.cat([cls_mask, X_mask], dim=1)

        for block in self.blocks:
            X = block(X, src_key_padding_mask=X_mask)

        # project back from transformed space to input space
        #X = self.output_layer(X)

        if return_CLS_only:
            return X[:,0]
        else:
            return X #return all tokens, including cls token in the first column
    
class SetTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_seeds=100, num_heads=4, activation="relu", ln=True):
        super(SetTransformer, self).__init__()
        self.initial_embed = nn.Linear(input_dim, model_dim)
        self.pma = PMA(dim_hidden=model_dim, num_seeds=num_seeds, num_heads=num_heads, ln=ln)
        self.sab = SAB(dim_hidden=model_dim, num_heads=num_heads, ln=ln)

    def forward(self, X, X_mask=None):
        X = self.initial_embed(X) # batch x dim_input -> batch x dim_hidden
        X = self.pma(X, X_mask) # batch x num_seeds x dim_hidden
        X = self.sab(X) # batch x num_seeds x dim_hidden
        X = X.reshape(X.shape[0], -1) # batch x (num_seeds*dim_hidden)
        return X

class ABMIL(nn.Module):

    def __init__(self, input_dim, model_dim, num_layers, gated_attn=True, activation='relu'):
        super().__init__()
        self.gated_attn = gated_attn

        self.V = nn.Linear(input_dim, model_dim)    
        if self.gated_attn:
            self.U = nn.Linear(input_dim, model_dim)
        self.w = nn.Linear(model_dim, 1)

        mlp_layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(last_dim, model_dim))
            mlp_layers.append(get_activation(activation))
            last_dim = model_dim
        mlp_layers.append(nn.Linear(model_dim, model_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, X, X_mask=None, return_attn=False):
        """
        X: batch size x num_cells x input_dim
        X_mask: batch size x num_cells
        """
        if self.gated_attn:
            dot_prod = self.w(torch.tanh(self.V(X)*torch.sigmoid(self.U(X)))) # batch size x num_cells x 1
        else:
            dot_prod = self.w(torch.tanh(self.V(X))) # batch size x num_cells x 1
        if X_mask is not None:
            dot_prod = dot_prod.masked_fill(X_mask.unsqueeze(-1), -1e9)
        attn = F.softmax(dot_prod, dim=1) # batch size x num_cells x 1
        X = torch.sum(attn * X, dim=1) # batch size x input_dim
        if return_attn:
            return self.mlp(X), attn.squeeze(-1)
        else:
            return self.mlp(X) # batch size x num_classes

class PseudoBulk(nn.Module):

    def __init__(self, input_dim, model_dim, libnorm=False, lognorm=False, project=False):
        super().__init__()
        self.libnorm = libnorm
        self.lognorm = lognorm
        self.project = project
        self.highly_expressed_mask = torch.zeros(input_dim)
        
        # define layers for potential projection
        if self.project:
            # define self.projection as a linear + RELU + linear
            self.projection = nn.Sequential(
                nn.Linear(input_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim)
            )

    def set_sizefactor_info(self, highly_expressed_mask):
        self.highly_expressed_mask = highly_expressed_mask  #mask, 1 if highly expressed, 0 otherwise

    def forward(self, X, X_mask=None):
        if X_mask is not None:
            # average X along dimension 1 (cells) where X_mask is False - batch is 0th dim, cells are 1st dim, gene features are 2nd dim
            X = (X * (~X_mask.unsqueeze(-1))).sum(dim=1) / (~X_mask.unsqueeze(-1)).sum(dim=1)        

        else:
            X = X.mean(dim=1)

        if self.libnorm:
            # size factor mask -- exclude highly expressed genes in each psbulk
            #mask = X > self.libnorm_maxfrac*X.sum(dim=1, keepdim=True)
            if self.highly_expressed_mask is not None:
                size_factors = X.masked_fill(self.highly_expressed_mask, 0).sum(dim=1, keepdim=True)
            else:
                size_factors = X.sum(dim=1, keepdim=True)
            X = X / size_factors*1e4

        if self.lognorm:
            X = torch.log1p(X)

        if self.project:
            X = self.projection(X)

        return X 
  
class RandomEmbeddingModel(nn.Module):

    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
    
    def forward(self, x, X_mask=None):
        return torch.randn(x.shape[0], self.model_dim).to(x.device)



