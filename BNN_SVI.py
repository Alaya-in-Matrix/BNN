from   pyro.optim          import Adam, SGD
from   torch.distributions import constraints
from   util                import NN
from   BNN                 import BNN
import os, sys
import pyro
import torch
import torch.nn as nn
import numpy    as np

class BNN_SVI(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_SVI, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_iters   = conf.get('num_iters',    4000)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  100)
        self.lr          = conf.get('lr',           1e-2)
        self.weight_std  = conf.get('weight_std',   1.0)
        self.fixed_noise = conf.get('fixed_noise',  None)
        self.nn          = NN(dim, self.act, self.num_hiddens, nout = 1)

    def model(self, X, y):
        """
        Normal distribution for weights and bias
        """
        num_x  = X.shape[0]
        priors = dict()
        for n, p in self.nn.named_parameters():
            priors[n] = pyro.distributions.Normal(loc = torch.zeros_like(p), scale = self.weight_std * torch.ones_like(p)).to_event(1)

        lifted_module    = pyro.random_module("module", self.nn, priors)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(X), subsample_size = min(num_x, self.batch_size)) as ind:
            pred = lifted_reg_model(X[ind]).squeeze(-1)
            pyro.sample("obs", pyro.distributions.Normal(pred, self.fixed_noise), obs = y[ind])

    def guide(self, X, y):
        priors   = dict()
        softplus = nn.Softplus()
        for n, p in self.nn.named_parameters():
            loc   = pyro.param("mu_"    + n, self.weight_std * torch.randn_like(p))
            scale = pyro.param("sigma_" + n, torch.randn_like(p))
            priors[n] = pyro.distributions.Normal(loc = loc, scale = softplus(scale)).to_event(1)
        lifted_module = pyro.random_module("module", self.nn, priors)
        return lifted_module()

    def train(self, X, y):
        if self.fixed_noise is None:
            print("No noise level provided, use noise_level = 0.05 * y.std()")
            self.fixed_noise = 0.05 * y.std()
        num_train         = X.shape[0]
        y                 = y.reshape(num_train)
        optim             = pyro.optim.Adam({"lr":self.lr})
        svi               = pyro.infer.SVI(self.model, self.guide, optim, loss = pyro.infer.Trace_ELBO())
        pyro.clear_param_store()
        self.rec = []
        for i in range(self.num_iters):
            loss = svi.step(X, y)
            if (i+1) % self.print_every == 0:
                self.rec.append(loss / num_train)
                print("[Iteration %05d/%05d] loss: %-4.3f" % (i + 1, self.num_iters, loss / num_train))

    def sample(self, num_samples = 1):
        nns = [self.guide(None, None) for i in range(num_samples)]
        return nns

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        pred  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            pred[i] = nns[i](X).squeeze()
        precs = torch.ones(pred.shape) / (self.fixed_noise**2)
        return pred, precs

    def report(self):
        print(self.nn)
