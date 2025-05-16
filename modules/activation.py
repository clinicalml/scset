import torch
import torch.nn as nn

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Activation {activation} not supported')
