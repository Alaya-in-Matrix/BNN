import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.utils import clamp_probs
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

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

class NoisyNN(NN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], logvar = torch.log(torch.tensor(1e-3))):
        super(NoisyNN, self).__init__(dim, act, num_hiddens, nout = 1)
        self.logvar = nn.Parameter(logvar)
    
    def forward(self, input):
        out     = self.nn(input)
        logvars = torch.clamp(self.logvar, max = 20.) * out.new_ones(out.shape)
        return torch.cat((out, logvars), dim = out.dim() - 1)

class StableRelaxedBernoulli(RelaxedBernoulli):
    """
    Numerical stable relaxed Bernoulli distribution
    """
    def rsample(self, sample_shape = torch.Size()):
        return clamp_probs(super(StableRelaxedBernoulli, self).rsample(sample_shape))

def stable_noise_var(input):
    return F.softplus(torch.clamp(input, min = -10.))

def stable_log_lik(mu, var, y):
    noise_var = stable_noise_var(var)
    return -0.5 * (mu - y)**2 / noise_var - 0.5 * torch.log(noise_var) - 0.5 * np.log(2 * np.pi)

def stable_nn_lik(nn_out, y):
    return stable_log_lik(nn_out[:, 0], nn_out[:, 1], y)
