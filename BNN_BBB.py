from   torch.distributions import constraints, kl_divergence, Normal
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import NN, stable_noise_var, stable_nn_lik, stable_log_lik
from   BNN                 import BNN
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.distributions as dist
import sys, os
from torch import autograd

class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GaussianLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mu           = Parameter(torch.zeros(out_features, 1 + in_features))
        self.rho          = Parameter(torch.zeros(out_features, 1 + in_features))
        nn.init.xavier_uniform_(self.mu[:, :-1])
        nn.init.constant_(self.rho, val = -5.)

    def rsample(self):
        eps           = torch.randn(self.mu.shape)
        scale         = stable_noise_var(self.rho).sqrt()
        self.wb       = self.mu + scale * eps

    def forward(self, input):
        w_mu    = self.mu[:, :-1]
        b_mu    = self.mu[:, -1]
        w_s2    = stable_noise_var(self.rho[:, :-1])
        b_s2    = stable_noise_var(self.rho[:, -1])
        out_mu  = F.linear(input,    weight = w_mu, bias = b_mu)
        out_std = F.linear(input**2, weight = w_s2, bias = b_s2).sqrt()
        eps     = torch.randn(out_mu.shape)
        return out_mu + eps * out_std

        # self.rsample()
        # w = self.wb[:, :-1]
        # b = self.wb[:, -1]
        # return F.linear(input, weight=w, bias = b)

    def sample_linear(self):
        self.rsample()
        w                 = self.wb[:, :-1]
        b                 = self.wb[:, -1]
        layer             = nn.Linear(self.in_features, self.out_features, bias = True)
        layer.weight.data = w.clone()
        layer.bias.data   = b.clone()
        return layer

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class BayesianNN(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50]):
        super(BayesianNN, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_layers  = len(num_hiddens)
        self.nn          = self.mlp()

    def sample(self):
        layers = []
        for layer in self.nn:
            layers.append(layer.sample_linear() if isinstance(layer, GaussianLinear) else layer)
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.nn(input)
        return out

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(GaussianLinear(pre_dim, self.num_hiddens[i]))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(GaussianLinear(pre_dim, 1))
        return nn.Sequential(*layers)

class BNN_BBB(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_BBB, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_epochs  = conf.get('num_epochs',   2000)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  100)

        self.lr          = conf.get('lr',           1e-2)
        self.weight_std  = conf.get('weight_std',   0.1)
        self.kl_factor   = conf.get('kl_factor',    1e-3)

        self.nn      = BayesianNN(dim, self.act, self.num_hiddens)

    def loss(self, X, y):
        num_x       = X.shape[0]
        X           = X.reshape(num_x, self.dim)
        y           = y.reshape(num_x, 1)
        pred        = self.nn(X)
        mse         = nn.MSELoss(reduction = 'mean')(pred, y)
        kld         = 0.
        for layer in self.nn.nn:
            if isinstance(layer, GaussianLinear):
                kld += kl_divergence(Normal(layer.mu, stable_noise_var(layer.rho).sqrt()), Normal(0., self.weight_std)).sum()
        return mse, kld

    def train(self, X, y):
        num_x       = X.shape[0]
        X           = X.reshape(num_x, self.dim)
        y           = y.reshape(num_x)
        dataset     = TensorDataset(X, y)
        loader      = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        self.nn.train()
        opt = torch.optim.Adam(self.nn.parameters(), self.lr)
        for epoch in range(self.num_epochs):
            epoch_kl  = 0.
            epoch_mse = 0.
            for bx, by in loader:
                opt.zero_grad()
                mse, kl_term  = self.loss(bx, by)
                kl_loss       = self.kl_factor * kl_term
                loss          = mse + kl_loss
                loss.backward()
                opt.step()
                epoch_mse += mse * bx.shape[0]
                epoch_kl   = kl_loss
            epoch_mse /= num_x
            if ((epoch + 1) % self.print_every == 0):
                print("[Epoch %5d, loss = %8.2f (KL = %8.2f, mse = %8.2f), kl_factor = %g]" % (epoch + 1, epoch_mse + epoch_kl, epoch_kl, epoch_mse, self.kl_factor))
        self.nn.eval()               

    def sample(self, num_samples = 1):
        nns = [self.nn.sample() for _ in range(num_samples)]
        return nns

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        pred  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            py      = nns[i](X).squeeze()
            pred[i] = py
        return pred

    def report(self):
        print(self.nn)
