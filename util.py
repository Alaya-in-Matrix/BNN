import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NN(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], nout = 2): #XXX: nout = 2, output and logarithm of heteroscedastic noise variance
        super(NN, self).__init__()
        self.dim          = dim
        self.nout         = nout
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_layers   = len(num_hiddens)
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
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(nn.Linear(pre_dim, self.nout, bias = True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.nn(x)
        return out

def stable_noise_var(log_noise_var):
    return 1e-6 + log_noise_var.exp()
