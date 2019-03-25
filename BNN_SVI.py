import pyro
import pyro.contrib.autoguide as autoguide
from   pyro.optim import Adam, SGD
import torch
import torch.nn as nn
import numpy    as np
from copy import deepcopy

class NN(nn.Module):
    def __init__(self, dim, act, num_hidden, num_layers):
        super(NN, self).__init__()
        self.dim        = dim
        self.act        = act
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.nn         = self.mlp()
        for l in self.nn:
            if type(l) == nn.Linear:
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(nn.Linear(pre_dim, self.num_hidden, bias=True))
            layers.append(self.act)
            pre_dim = self.num_hidden
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)


class BNN_SVI:
    def __init__(self, dim, act = nn.Tanh(), conf = dict()):
        self.dim         = dim
        self.act         = act
        self.num_hidden  = conf.get('num_hidden',   50)
        self.num_layers  = conf.get('num_layers',   3)
        self.num_iters   = conf.get('num_iters',    400)
        self.lr          = conf.get('lr',           1e-3)
        self.batch_size  = conf.get('batch_size',   128)
        self.print_every = conf.get('print_every',  100)
        self.nn          = NN(dim, self.act, self.num_hidden, self.num_layers).nn

    def model(self, X, y):
        """
        Normal distribution for weights and bias
        Gamma for precision
        """
        num_x         = X.shape[0]
        # prec             = pyro.sample("precision", pyro.distributions.Gamma(5,5))
        # noise_scale      = torch.sqrt(1 / prec)
        log_noise_var = pyro.sample("log_noise_var", pyro.distributions.Normal(0, 1))
        noise_scale   = torch.sqrt(torch.exp(log_noise_var))
        priors        = dict()
        for n, p in self.nn.named_parameters():
            if "weight" in n:
                priors[n] = pyro.distributions.Normal(torch.zeros_like(p), torch.tensor(1.0))
            elif "bias" in n:
                priors[n] = pyro.distributions.Normal(torch.zeros_like(p), torch.tensor(20.0))

        lifted_module    = pyro.random_module("module", self.nn, priors)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(X), subsample_size = min(num_x, self.batch_size)) as ind:
            prediction_mean = lifted_reg_model(X[ind]).squeeze(-1)
            pyro.sample("obs", 
                    pyro.distributions.Normal(prediction_mean, noise_scale), 
                    obs = y[ind])

    def train(self, X, y):
        print(X.shape)
        print(y.shape)
        num_train   = X.shape[0]
        y           = y.reshape(num_train, 1)
        self.x_mean = X.mean(dim = 0)
        self.x_std  = X.std(dim = 0)
        self.y_mean = y.mean()
        self.y_std  = y.std()
        self.X      = (X - self.x_mean) / self.x_std
        self.y      = (y - self.y_mean) / self.y_std
        self.guide = autoguide.AutoDiagonalNormal(self.model)
        optim      = Adam({"lr":self.lr})
        svi        = pyro.infer.SVI(
                self.model,
                self.guide,
                optim,
                loss = pyro.infer.Trace_ELBO())
        pyro.clear_param_store()
        self.rec = []
        for i in range(self.num_iters):
            loss = svi.step(self.X, self.y)
            self.rec.append(loss)
            if i % self.print_every == 0:
                print("[iteration %05d] loss: %.4f" % (i + 1, loss / num_train))
    
    def sample(self):
        params = self.guide(self.X, self.y);
        new_nn = deepcopy(self.nn)
        for n, p in new_nn.named_parameters():
            p.data = params['module$$$' + n]
        return new_nn
