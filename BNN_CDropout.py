import sys, os
import torch
import torch.optim         as optim
import torch.nn            as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy               as np
from util import *
from BNN  import BNN
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.utils import clamp_probs, probs_to_logits, logits_to_probs

class CDropout(nn.Module):
    def __init__(self, p = 0.5):
        super(CDropout, self).__init__()
        self.p_logit = nn.Parameter(probs_to_logits(torch.as_tensor(p), is_binary = True))
    
    def dropout_rate(self):
        return clamp_probs(torch.sigmoid(self.p_logit))

    def forward(self, input):
        bdist = StableRelaxedBernoulli(probs = 1 - self.dropout_rate(), temperature = 0.1)
        return input * bdist.rsample(input.shape)

    def extra_repr(self):
        return 'dropout_rate = {}'.format(self.dropout_rate())


class CDropoutLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CDropoutLinear, self).__init__()
        self.layer = nn.Sequential(
                CDropout(), 
                nn.Linear(in_features, out_features))

    def forward(self, input):
        return self.layer(input)

    def sample(self):
        dropout_rate        = self.layer[0].dropout_rate()
        linear              = nn.Linear(self.layer[1].in_features, self.layer[1].out_features)
        linear.weight.data  = self.layer[1].weight.data.clone()
        linear.bias.data    = self.layer[1].bias.data.clone()
        linear.weight.data *= StableRelaxedBernoulli(probs = 1 - dropout_rate, temperature = 0.1).sample((linear.in_features, ))
        return linear

class NN_CDropout(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], nout = 2):
        super(NN_CDropout, self).__init__()
        self.dim          = dim
        self.nout         = nout
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_layers   = len(num_hiddens)
        self.nn           = self.mlp()

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(CDropoutLinear(pre_dim, self.num_hiddens[i]))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(CDropoutLinear(pre_dim, self.nout))
        return nn.Sequential(*layers)

    def sample(self):
        layers = []
        for layer in self.nn:
            layers.append(layer.sample() if isinstance(layer, CDropoutLinear) else layer)
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.nn(input)
        return out

class BNN_CDropout(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_CDropout, self).__init__()
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_epochs   = conf.get('num_epochs',   400)
        self.l2_reg       = conf.get('l2_reg',       1e-6)
        self.lr           = conf.get('lr',           1e-2)
        self.batch_size   = conf.get('batch_size',   32)
        self.print_every  = conf.get('print_every',  100)
        self.normalize    = conf.get('normalize',    True)
        self.use_cuda     = conf.get('use_cuda',     False) and torch.cuda.is_available()
        self.nn           = NN_CDropout(dim, self.act, self.num_hiddens, 2)
        if self.use_cuda:
            self.nn = self.nn.cuda()

    def train(self, X, y):
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        self.normalize_Xy(X, y, self.normalize)
        num_train = self.X.shape[0]
        opt       = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        dataset   = TensorDataset(self.X, self.y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            for bx, by in loader:
                opt.zero_grad()
                loss = -1 * stable_nn_lik(self.nn(bx), by).sum()
                loss.backward()
                opt.step()
                epoch_loss += loss
            if (epoch + 1) % self.print_every == 0:
                print("[Epoch %5d, loss = %.2f]" % (epoch + 1, epoch_loss))
        self.nn = self.nn.cpu()
        if self.normalize:
            self.x_mean  = self.x_mean.cpu()
            self.x_std   = self.x_std.cpu()
            self.y_mean  = self.y_mean.cpu()
            self.y_std   = self.y_std.cpu()

    def sample(self, num_samples = 1):
        nns = [self.nn.sample() for i in range(num_samples)]
        return nns

    def sample_predict(self, nns, X):
        num_x     = X.shape[0]
        X         = (X - self.x_mean) / self.x_std
        pred      = torch.zeros(len(nns), num_x)
        prec = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            nn_out    = nns[i](X)
            py        = nn_out[:, 0]
            logvar    = nn_out[:, 1]
            noise_var = stable_noise_var(logvar) * self.y_std**2
            pred[i]   = self.y_mean + py  * self.y_std
            prec[i]   = 1 / noise_var
        return pred, prec

    def report(self):
        print(self.nn.nn)
