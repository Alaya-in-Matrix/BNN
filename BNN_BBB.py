from   torch.distributions import constraints
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import ScaleLayer, NN
from   BNN                 import BNN
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

    def sample_linear(self):
        wb                = self.dist.rsample()
        w                 = wb[:, :-1]
        b                 = wb[:, -1]
        layer             = nn.Linear(self.in_features, self.out_features, bias = True)
        layer.weight.data = w.clone()
        layer.bias.data   = b.clone()
        return layer

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

class BayesianNN(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], scale = True):
        super(BayesianNN, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_layers  = len(num_hiddens)
        self.scale       = scale
        self.nn          = self.mlp()

    def sample(self):
        layers = []
        for layer in self.nn:
            if isinstance(layer, GaussianLinear):
                layers.append(layer.sample_linear())
            else:
                layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input):
        return self.nn(input)

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(GaussianLinear(pre_dim, self.num_hiddens[i]))
            if self.scale:
                layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(GaussianLinear(pre_dim, 1))
        if self.scale:
            layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
        return nn.Sequential(*layers)

class MixturePrior:
    def __init__(self, factor = 0.5, s1 = 10., s2 = 0.01):
        self.factor = torch.tensor(factor).float()
        mu          = torch.tensor(0.).float()
        s1          = torch.tensor(s1).float()
        s2          = torch.tensor(s2).float()
        self.dist1  = torch.distributions.Normal(mu, s1)
        self.dist2  = torch.distributions.Normal(mu, s2)
    def log_prob(self, samples):
        if self.factor == 1:
            return self.dist1.log_prob(samples)
        elif self.factor == 0:
            return self.dist2.log_prob(samples)
        else:
            lp1 = self.dist1.log_prob(samples) + torch.log(self.factor)
            lp2 = self.dist2.log_prob(samples) + torch.log(1 - self.factor)
            return torch.logsumexp(torch.stack((lp1, lp2), dim = 0), dim = 0)


class BNN_BBB(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_BBB, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_epochs  = conf.get('num_epochs',   400)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  50)
        self.n_samples   = conf.get('n_samples',    1)
        self.lr          = conf.get('lr',           1e-2)
        self.pi          = conf.get('pi',           0.75)
        self.s1          = conf.get('s1',           5.)
        self.s2          = conf.get('s2',           0.2)
        self.noise_level = conf.get('noise_level',  0.1)
        self.normalize   = conf.get('normalize',    True)
        self.scale_layer = conf.get('scale_layer',  True)
        self.w_prior     = MixturePrior(factor = self.pi, s1 = self.s1, s2 = self.s2)
        self.nn          = BayesianNN(dim, self.act, self.num_hiddens, self.scale_layer)

    def loss(self, X, y):
        num_x   = X.shape[0]
        X       = X.reshape(num_x, self.dim)
        y       = y.reshape(num_x)
        log_lik = torch.tensor(0.)
        log_qw  = torch.tensor(0.)
        log_pw  = torch.tensor(0.)
        for i in range(self.n_samples):
            pred     = self.nn(X).reshape(num_x)
            log_lik += torch.distributions.Normal(pred, self.noise_level).log_prob(y).sum()
            for layer in self.nn.nn:
                if isinstance(layer, GaussianLinear):
                    log_qw += layer.log_prob
                    log_pw += self.w_prior.log_prob(layer.wb).sum()
            kl_term = log_qw - log_pw
        return log_lik / self.n_samples, kl_term / self.n_samples

    def train(self, X, y):
        num_x = X.shape[0]
        X     = X.reshape(num_x, self.dim)
        y     = y.reshape(num_x)
        self.normalize_Xy(X, y, self.normalize)
        self.noise_level /= self.y_std
        dataset   = TensorDataset(self.X, self.y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        opt       = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        num_batch = len(loader)
        for epoch in range(self.num_epochs):
            batch_cnt = 1
            for bx, by in loader:
                opt.zero_grad()
                log_lik             = torch.tensor(0.)
                kl_term             = torch.tensor(0.)
                _log_lik, _kl_term  = self.loss(bx, by)
                log_lik            += _log_lik
                kl_term            += _kl_term
                pi                  = 2**(num_batch - batch_cnt) / (2**(num_batch) - 1)
                loss                = (pi * kl_term - log_lik) / self.n_samples
                loss.backward()
                opt.step()
                batch_cnt += 1
            if ((epoch + 1) % self.print_every == 0):
                log_lik, kl_term = self.loss(self.X, self.y)
                print("[Epoch %5d, loss = %.4g (KL = %.4g, -log_lik = %.4g)]" % (epoch + 1, kl_term - log_lik, kl_term, -1 * log_lik), flush = True)
        self.noise_level *= self.y_std

    def sample(self, num_samples = 1):
        nns = [self.nn.sample() for i in range(num_samples)]
        return nns, torch.ones(num_samples) / (self.noise_level**2)

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        X     = (X - self.x_mean) / self.x_std
        pred  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            pred[i] = self.y_mean + nns[i](X).squeeze() * self.y_std
        return pred


