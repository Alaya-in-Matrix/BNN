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

    def dropout_rate(self):
        return self.layer[0].dropout_rate()

    def reg(self):
        p       = self.layer[0].dropout_rate()
        entropy = -1 * (p * p.log() + (1-p) * (1-p).log())
        w2      = torch.sum(self.layer[1].weight**2)
        return w2, torch.tensor(1.*self.layer[1].in_features) * entropy

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
        self.dim         = dim
        self.nout        = nout
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_layers  = len(num_hiddens)
        self.hidden      = self.mlp()
        self.pred        = CDropoutLinear(num_hiddens[-1], 1)
        self.logvar      = CDropoutLinear(num_hiddens[-1], 1)

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(CDropoutLinear(pre_dim, self.num_hiddens[i]))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        return nn.Sequential(*layers)

    def sample(self):
        layers = []
        for layer in self.hidden:
            layers.append(layer.sample() if isinstance(layer, CDropoutLinear) else layer)
        pred_layer            = self.pred.sample()
        logvar_layer          = self.logvar.sample()
        out_layer             = nn.Linear(self.num_hiddens[-1], 2)
        out_layer.weight.data = torch.cat((pred_layer.weight.data, logvar_layer.weight.data), dim=0).clone()
        out_layer.bias.data   = torch.cat((pred_layer.bias.data, logvar_layer.bias.data)).clone()
        layers.append(out_layer)
        return nn.Sequential(*layers)

    def forward(self, input):
        out    = self.hidden(input)
        pred   = self.pred(out)
        logvar = self.logvar(out)
        return torch.cat((pred, logvar), dim = input.dim()-1)

class BNN_CDropout(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_CDropout, self).__init__()
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_epochs   = conf.get('num_epochs',   400)
        self.lr           = conf.get('lr',           1e-2)
        self.lscale       = conf.get('lscale',       1e-2)
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
            epoch_lik  = 0.
            epoch_wreg = 0.
            epoch_ent  = 0.
            for bx, by in loader:
                opt.zero_grad()
                log_lik    = stable_nn_lik(self.nn(bx), by).sum()
                wreg, ent  = self.reg()
                wreg      *= bx.shape[0] / num_train
                ent       *= bx.shape[0] / num_train
                loss       = -1 * log_lik + (wreg - ent)
                loss.backward()
                opt.step()
                epoch_lik  += log_lik 
                epoch_wreg += wreg    
                epoch_ent  += ent     
            if (epoch + 1) % self.print_every == 0:
                print("[Epoch %5d, loss = %-6.2f, -log_lik = %-6.2f, wreg = %g, -entropy = %6.2f]" % (
                    epoch + 1,
                    epoch_wreg - epoch_ent - epoch_lik,
                    -1 * epoch_lik,
                    epoch_wreg,
                    -1 * epoch_ent)
                )
        self.nn = self.nn.cpu()
        if self.normalize:
            self.x_mean  = self.x_mean.cpu()
            self.x_std   = self.x_std.cpu()
            self.y_mean  = self.y_mean.cpu()
            self.y_std   = self.y_std.cpu()
    
    def reg(self):
        entropy    = torch.tensor(0.)
        weight_reg = torch.tensor(0.)
        for layer in self.nn.hidden:
            if isinstance(layer, CDropoutLinear):
                w2, ent     = layer.reg()
                entropy    += ent
                weight_reg += 0.5 * self.lscale**2 * (1 - layer.dropout_rate()) * w2
        w2, ent     = self.nn.pred.reg()
        entropy    += ent
        weight_reg += 0.5 * self.lscale**2 * (1 - self.nn.pred.dropout_rate()) * w2
        w2, ent     = self.nn.logvar.reg()
        entropy    += ent
        weight_reg += 0.5 * self.lscale**2 * (1 - self.nn.logvar.dropout_rate()) * w2
        return weight_reg, entropy


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
        print(self.nn)
