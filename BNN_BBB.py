from   torch.distributions import constraints
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import NN, stable_noise_var
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
        self.dist           = torch.distributions.Normal(self.mu, 1e-4 + F.softplus(self.rho))
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
        out[:, 2] = np.log(0.01)
        return out

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(GaussianLinear(pre_dim, self.num_hiddens[i]))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(GaussianLinear(pre_dim, 2))
        return nn.Sequential(*layers)

class BNN_BBB(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_BBB, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_epochs  = conf.get('num_epochs',   400)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  50)
        self.lr          = conf.get('lr',           1e-2)
        self.normalize   = conf.get('normalize',    True)
        self.w_prior     = torch.distributions.Normal(torch.zeros(1), 0.4 * torch.ones(1))
        self.nn          = BayesianNN(dim, self.act, self.num_hiddens)

    def loss(self, X, y):
        num_x       = X.shape[0]
        X           = X.reshape(num_x, self.dim)
        y           = y.reshape(num_x)
        nn_out      = self.nn(X)
        pred        = nn_out[:, 0]
        logvar      = nn_out[:, 1]
        prec        = 1 / stable_noise_var(logvar)
        # log_lik     = torch.sum(-0.5 * prec * (pred - y)**2 - 0.5 * logvar - 0.5 * np.log(2 * np.pi))
        log_lik     = -1 * nn.MSELoss()(pred, y)
        log_qw      = torch.tensor(0.)
        log_pw      = torch.tensor(0.)
        for layer in self.nn.nn:
            if isinstance(layer, GaussianLinear):
                log_qw += layer.log_prob
                log_pw += self.w_prior.log_prob(layer.wb).sum()
        kl_term = log_qw - log_pw
        if torch.isinf(log_lik):
            rec = torch.zeros(logvar.numel(), 4)
            rec[:, 0] = pred
            rec[:, 1] = noise_level
            rec[:, 2] = y
            rec[:, 3] = torch.distributions.Normal(pred, noise_level).log_prob(y)
            print(rec)
            sys.exit(1)
        return log_lik / num_x, kl_term / num_x

    def train(self, X, y):
        num_x = X.shape[0]
        X     = X.reshape(num_x, self.dim)
        y     = y.reshape(num_x)
        self.normalize_Xy(X, y, self.normalize)
        dataset   = TensorDataset(self.X, self.y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        opt       = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        num_batch = len(loader)
        for epoch in range(self.num_epochs):
            batch_cnt = 1
            for bx, by in loader:
                opt.zero_grad()
                log_lik, kl_term  = self.loss(bx, by)
                # pi                = 2**(num_batch - batch_cnt) / (2**(num_batch) - 1)
                pi = 0.
                loss              = (pi * kl_term - log_lik)
                loss.backward()
                for n, p in self.nn.named_parameters():
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(n)
                        print(p)
                        print(p.grad)
                        print('lik = %.2f, kl = %.2f' % (log_lik, kl_term))
                        sys.exit(1)
                opt.step()
                batch_cnt += 1
            if ((epoch + 1) % self.print_every == 0):
                log_lik, kl_term = self.loss(self.X, self.y)
                print("[Epoch %5d, loss = %.4g (KL = %.4g, -log_lik = %.4g)]" % (epoch + 1, kl_term - log_lik, kl_term, -1 * log_lik), flush = True)

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
