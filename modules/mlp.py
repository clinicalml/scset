import torch
import torch.nn as nn
from .activation import get_activation


def build_classifier_head(args, sample_dim, output_dim):
    """
    Training head for classification tasks (in particular supervised training)
    """
    if args.encoder == 'pseudo_bulk':
        input_dim = sample_dim
    elif args.encoder in ['pseudobulk', 'celltypefracs', 'celltypemeans', 'catfracsmeans', 'kmeans']:
        input_dim = sample_dim
    # elif args.encoder == 'cell_transformer': 
    #     input_dim = sample_dim # cell_transformer projects back to input space
    elif args.encoder == 'set_transformer':
        input_dim = args.model_dim*args.num_seeds
    else:
        input_dim = args.model_dim
    mlp = MLP(input_dim=input_dim,
              hidden_dim=args.model_dim,
              output_dim=output_dim,
              num_hidden_layers=args.classifier_num_hidden_layers,
              activation=args.classifier_activation,
              dropout=args.classifier_dropout,
              layernorm_switch=args.classifier_no_layernorm)
    return mlp

def build_sorter_head(args, sample_dim):
    """
    Training head specifically for the CellSorter task.
    """
    # set input_dim to dim of one sample + one cell concatenated together
    if args.encoder == 'pseudo_bulk':
        input_dim = 2*sample_dim 
    if args.encoder == 'cell_transformer':
        input_dim = 2*sample_dim
    elif args.encoder == 'set_transformer':
        input_dim = (args.model_dim*args.num_seeds)+sample_dim
    else:
        input_dim = args.model_dim+sample_dim

    mlp = MLP(input_dim=input_dim,
              hidden_dim=args.model_dim,
              output_dim=1,
              num_hidden_layers=args.projection_num_hidden_layers,
              activation=args.projection_activation,
              dropout=args.projection_dropout,
              layernorm_switch=args.projection_no_layernorm)
    return mlp


def build_projection_head(args, sample_dim):
    """
    Training head for the contrastive learning task.
    """
    if args.encoder == 'pseudo_bulk':
        input_dim = sample_dim
    if args.encoder == 'cell_transformer':
        input_dim = sample_dim
    elif args.encoder == 'set_transformer':
        input_dim = (args.model_dim*args.num_seeds)
    else:
        input_dim = args.model_dim
    mlp = MLP(
        input_dim=input_dim,
        hidden_dim=args.model_dim,
        output_dim=args.projection_output_dim,
        num_hidden_layers=args.projection_num_hidden_layers,
        activation=args.projection_activation,
        dropout=args.projection_dropout,
        layernorm_switch=args.projection_no_layernorm
    )
    return mlp

def build_clustermask_head(args, sample_dim):
    """
    Training head for the clustermask task.
    """
    # if args.encoder == 'pseudo_bulk':
    #     input_dim = sample_dim
    # elif args.encoder == 'set_transformer':
    #     input_dim = (args.model_dim*args.num_seeds)
    # else:
    #     input_dim = args.model_dim
    input_dim = args.model_dim # right now only using this pretraining for cell_transformer
    mlp = MLP(
        input_dim=input_dim,
        hidden_dim=args.model_dim,
        output_dim=2, # corresponding to mean, n
        num_hidden_layers=args.clustermask_num_hidden_layers,
        activation=args.clustermask_activation,
        dropout=args.clustermask_dropout,
        layernorm_switch=args.clustermask_no_layernorm
    )
    return mlp


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, activation='silu', dropout=0.0, layernorm_switch=True):
        super(MLP, self).__init__()

        layers = []
        last_dim = input_dim

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            if layernorm_switch:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)