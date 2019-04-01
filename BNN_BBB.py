from   torch.distributions import constraints
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import sys, os

class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GaussianLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.mu           = Parameter(torch.randn(out_features, 1 + in_features))
        self.rho          = Parameter(torch.randn(out_features, 1 + in_features))

    def forward(self, input):
        self.dist           = torch.distributions.Normal(self.mu, F.softplus(self.rho))
        self.wb             = self.dist.rsample()
        w                   = self.wb[:, :-1]
        b                   = self.wb[:, -1]
        self.log_prob       = self.dist.log_prob(self.wb).sum()
        return F.linear(input, weight=w, bias = b)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class NoiseLayer(nn.Module):
    def __init__(self, alpha, beta):
        super(NoiseLayer, self).__init__()
        self.alpha = Parameter(alpha)
        self.beta  = Parameter(beta)

    def forward(self, input):
        self.dist      = torch.distributions.Gamma(F.softplus(self.alpha), F.softplus(self.beta))
        self.precision = self.dist.rsample()
        self.log_prob  = self.dist.log_prob(self.precision).sum()
        noise_level    = 1 / self.precision.sqrt()
        return input, noise_level

class NN(nn.Module):
    def __init__(self, dim, act, num_hidden, num_layers, alpha, beta):
        super(NN, self).__init__()
        self.dim        = dim
        self.act        = act
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.alpha      = alpha
        self.beta       = beta
        self.nn         = self.mlp()
    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(GaussianLinear(pre_dim, self.num_hidden))
            layers.append(self.act)
            pre_dim = self.num_hidden
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        layers.append(NoiseLayer(self.alpha, self.beta))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)

class MixturePrior:
    def __init__(self, factor = 0.5, s1 = 10, s2 = 0.01):
        self.factor = torch.tensor(factor)
        self.dist1  = torch.distributions.Normal(0, s1)
        self.dist2  = torch.distributions.Normal(0, s2)
    def log_prob(self, samples):
        lp1   = self.dist1.log_prob(samples).sum()
        lp2   = self.dist2.log_prob(samples).sum()
        if self.factor   == 1:
            mix_lp = lp1
        elif self.factor == 0:
            mix_lp = lp2
        else:
            log_ratio = torch.log(self.factor / (1 - self.factor)) + (lp1 - lp2)
            if log_ratio > 0:
                mix_lp = lp1 + torch.log(1 + torch.exp(-1 * log_ratio)) + torch.log(self.factor)
            else:
                mix_lp = lp2 + torch.log(1 + torch.exp(log_ratio))   + torch.log(1 - self.factor)
        return mix_lp

# TODO: mixture prior
class BNN_BBB:
    def __init__(self, dim, act = nn.Tanh(), conf = dict()):
        self.dim         = dim
        self.act         = act
        self.num_hidden  = conf.get('num_hidden',   50)
        self.num_layers  = conf.get('num_layers',   3)
        self.num_epochs  = conf.get('num_epochs',   100)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  1)
        self.lr          = conf.get('lr',           1e-3)
        self.pi          = conf.get('pi',           0.25)
        self.s1          = conf.get('s1',           2.)
        self.s2          = conf.get('s2',           1.)
        self.alpha       = torch.tensor(conf.get('alpha', 1000.))
        self.beta        = torch.tensor(conf.get('beta',  5.))
        self.n_samples   = conf.get('n_samples', 1)
        self.normalize   = conf.get('normalize', True)
        self.nn          = NN(dim, self.act, self.num_hidden, self.num_layers, self.alpha, self.beta).nn
        self.w_prior     = MixturePrior(factor = self.pi, s1 = self.s1, s2 = self.s2)
        self.n_prior     = torch.distributions.Gamma(F.softplus(self.alpha), F.softplus(self.beta))

    def loss(self, X, y):
        num_x             = X.shape[0]
        X                 = X.reshape(num_x, self.dim)
        y                 = y.reshape(num_x)
        pred, noise_level = self.nn(X)
        pred              = pred.reshape(num_x)
        log_lik           = torch.distributions.Normal(pred, noise_level).log_prob(y).sum()
        log_qw            = torch.tensor(0.)
        log_pw            = torch.tensor(0.)
        for layer in self.nn:
            if isinstance(layer, GaussianLinear):
                log_qw += layer.log_prob
                log_pw += self.w_prior.log_prob(layer.wb).sum()
            elif isinstance(layer, NoiseLayer):
                log_qw += layer.log_prob
                log_qw += self.n_prior.log_prob(layer.precision).sum()
        kl_term = log_qw - log_pw
        return log_lik, kl_term

    def train(self, X, y):
        num_x = X.shape[0]
        X     = X.reshape(num_x, self.dim)
        y     = y.reshape(num_x)
        if self.normalize:
            self.x_mean  = X.mean(dim = 0)
            self.x_std   = X.std(dim = 0)
            self.y_mean  = y.mean()
            self.y_std   = y.std()
        else:
            self.x_mean  = 0
            self.x_std   = 1
            self.y_mean  = 0
            self.y_std   = 1
        self.train_x = (X - self.x_mean) / self.x_std
        self.train_y = (y - self.y_mean) / self.y_std
        dataset      = TensorDataset(self.train_x, self.train_y)
        loader       = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        opt          = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        num_batch    = len(loader)
        for epoch in range(self.num_epochs):
            batch_cnt = 1
            for bx, by in loader:
                opt.zero_grad()
                log_lik = torch.tensor(0.)
                kl_term = torch.tensor(0.)
                for i in range(self.n_samples):
                    _log_lik, _kl_term  = self.loss(bx, by)
                    log_lik += _log_lik
                    kl_term += _kl_term
                pi   = 2**(num_batch - batch_cnt) / (2**(num_batch) - 1)
                loss = (pi * kl_term - log_lik) / self.n_samples
                loss.backward()
                opt.step()
                batch_cnt += 1
            if ((epoch + 1) % self.print_every == 0):
                log_lik, kl_term = self.loss(self.train_x, self.train_y)
                print("[Epoch %5d, loss = %.4g (KL = %.4g, -log_lik = %.4g)]" % (epoch + 1, kl_term - log_lik, kl_term, -1 * log_lik))

    def sample(self, n_samples = 100):
        pass

    def validate(self, test_X, test_y):
        pass
