from   torch.distributions import constraints
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import NN, stable_noise_var, stable_nn_lik, stable_log_lik
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
        scale         = stable_noise_var(self.rho).sqrt()
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

        self.lr_weight   = conf.get('lr_weight',    1e-2)
        self.lr_noise    = conf.get('lr_noise',     1e-2)
        self.weight_std  = conf.get('weight_std',   0.2)

        self.w_prior = torch.distributions.Normal(torch.zeros(1), self.weight_std*torch.ones(1))
        self.nn      = BayesianNN(dim, self.act, self.num_hiddens)
        self.X       = None
        self.y       = None
        self.logvar  = nn.Parameter(torch.tensor(-2.))

    def loss(self, X, y):
        num_x       = X.shape[0]
        X           = X.reshape(num_x, self.dim)
        y           = y.reshape(num_x)
        pred        = self.nn(X).squeeze()
        log_lik     = stable_log_lik(pred, self.logvar, y).sum()
        log_qw      = torch.tensor(0.)
        log_pw      = torch.tensor(0.)
        for layer in self.nn.nn:
            if isinstance(layer, GaussianLinear):
                log_qw += layer.log_prob
                log_pw += self.w_prior.log_prob(layer.wb).sum()
        kl_term = log_qw - log_pw
        return log_lik, kl_term

    def train(self, X, y):
        num_x   = X.shape[0]
        X       = X.reshape(num_x, self.dim)
        y       = y.reshape(num_x)
        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        param_dict = dict()
        param_dict['logvar'] = self.logvar
        param_dict['nn'] = self.nn.parameters()

        dict_logvar = {"params": self.logvar,          'lr': self.lr_noise}
        dict_nn     = {"params": self.nn.parameters(), 'lr': self.lr_weight}
        opt         = torch.optim.Adam([dict_logvar, dict_nn])
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
                print("[Epoch %5d, loss = %8.2f (KL = %8.2f, -log_lik = %8.2f), noise_var = %g]" % (epoch + 1, epoch_kl - epoch_lik, epoch_kl, -1 * epoch_lik, stable_noise_var(self.logvar)))

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
