from   torch.distributions import constraints
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import NN, stable_noise_var, stable_nn_lik
from   BNN                 import BNN
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import sys, os
from torch import autograd

class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GaussianLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mu           = Parameter(torch.randn(out_features, 1 + in_features))
        self.rho          = Parameter(torch.randn(out_features, 1 + in_features))

    def rsample(self):
        eps           = torch.randn(self.mu.shape)
        scale         = F.softplus(torch.clamp(self.rho, min = -7.))
        self.wb       = self.mu + scale * eps
        self.log_prob = torch.sum(-0.5 * torch.pow((self.wb - self.mu) / scale, 2) - 0.5 * torch.log(2 * np.pi * scale**2))

    def forward(self, input):
        self.rsample()
        w = self.wb[:, :-1]
        b = self.wb[:, -1]
        return F.linear(input, weight=w, bias = b)

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
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], nout = 2):
        super(BayesianNN, self).__init__()
        self.dim         = dim
        self.nout        = nout
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
        layers.append(GaussianLinear(pre_dim, self.nout))
        return nn.Sequential(*layers)

class BNN_BBB(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_BBB, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_epochs  = conf.get('num_epochs',   4000)
        self.batch_size  = conf.get('batch_size',   64)
        self.print_every = conf.get('print_every',  100)
        self.lr          = conf.get('lr',           1e-2)
        self.normalize   = conf.get('normalize',    True)
        self.w_prior     = torch.distributions.Normal(torch.zeros(1), torch.ones(1))
        self.nn          = BayesianNN(dim, self.act, self.num_hiddens)

    def loss(self, X, y):
        num_x       = X.shape[0]
        X           = X.reshape(num_x, self.dim)
        y           = y.reshape(num_x)
        nn_out      = self.nn(X)
        log_lik     = stable_nn_lik(nn_out, y).sum()
        log_qw      = torch.tensor(0.)
        log_pw      = torch.tensor(0.)
        for layer in self.nn.nn:
            if isinstance(layer, GaussianLinear):
                log_qw += layer.log_prob
                log_pw += self.w_prior.log_prob(layer.wb).sum()
        kl_term = log_qw - log_pw
        return log_lik, kl_term

    def train(self, X, y):
        num_x = X.shape[0]
        X     = X.reshape(num_x, self.dim)
        y     = y.reshape(num_x)
        self.normalize_Xy(X, y, self.normalize)
        dataset   = TensorDataset(self.X, self.y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        # opt       = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        opt       = torch.optim.RMSprop(self.nn.parameters(), lr = self.lr, centered = True)
        for epoch in range(self.num_epochs):
            epoch_kl  = 0.
            epoch_lik = 0.
            for bx, by in loader:
                opt.zero_grad()
                log_lik, kl_term  = self.loss(bx, by)
                kl_term          *= bx.shape[0] / num_x
                loss              = kl_term - log_lik
                loss.backward()
                opt.step()
                epoch_kl  += kl_term
                epoch_lik += log_lik
            if ((epoch + 1) % self.print_every == 0):
                log_lik, kl_term = self.loss(self.X, self.y)
                print("[Epoch %5d, loss = %.4g (KL = %.4g, -log_lik = %.4g)]" % (epoch + 1, epoch_kl - epoch_lik, epoch_kl, -1 * epoch_lik))

    def sample(self, num_samples = 1):
        nns = [self.nn.sample() for i in range(num_samples)]
        return nns

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        X     = (X - self.x_mean) / self.x_std
        pred  = torch.zeros(len(nns), num_x)
        prec  = torch.zeros(len(nns), num_x)
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
