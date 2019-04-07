import torch
import torch.nn as nn
import numpy as np

class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super(ScaleLayer, self).__init__()
        self.scale = scale
    def forward(self, input):
        return input * self.scale
    def extra_repr(self):
        return 'scale = {}'.format(self.scale)

class NN(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], scale = True):
        super(NN, self).__init__()
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_layers   = len(num_hiddens)
        self.scale        = scale
        self.nn           = self.mlp()
        for l in self.nn:
            if type(l) == nn.Linear:
                nn.init.xavier_uniform_(l.weight)
                nn.init.zeros_(l.bias)
    
    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(nn.Linear(pre_dim, self.num_hiddens[i], bias=True))
            if self.scale:
                layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        if self.scale:
            layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.nn(x)
