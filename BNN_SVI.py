import pyro
from   pyro.optim import Adam, SGD
from   torch.distributions import constraints
import torch
import torch.nn as nn
import numpy    as np

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
        self.dim            = dim
        self.act            = act
        self.num_hidden     = conf.get('num_hidden',   50)
        self.num_layers     = conf.get('num_layers',   3)
        self.num_iters      = conf.get('num_iters',    400)
        self.lr             = conf.get('lr',           1e-3)
        self.batch_size     = conf.get('batch_size',   128)
        self.print_every    = conf.get('print_every',  100)
        self.weight_prior   = conf.get('weight_prior', 1.0)
        self.bias_prior     = conf.get('bias_prior',   1.0)
        self.log_noise_mean = conf.get('log_noise_mean', -2.0)
        self.log_noise_std  = conf.get('log_noise_std', 1.)
        self.nn             = NN(dim, self.act, self.num_hidden, self.num_layers).nn

    def model(self, X, y):
        """
        Normal distribution for weights and bias
        Gamma for precision
        """
        noise_scale = pyro.sample("noise_scale", pyro.distributions.LogNormal(self.log_noise_mean, self.log_noise_std))
        num_x       = X.shape[0]
        priors      = dict()
        for n, p in self.nn.named_parameters():
            if "weight" in n:
                priors[n] = pyro.distributions.Normal(loc = torch.zeros_like(p), scale = self.weight_prior * torch.ones_like(p)).to_event(1)
            elif "bias" in n:
                priors[n] = pyro.distributions.Normal(loc = torch.zeros_like(p), scale = self.bias_prior   * torch.ones_like(p)).to_event(1)

        lifted_module    = pyro.random_module("module", self.nn, priors)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(X), subsample_size = min(num_x, self.batch_size)) as ind:
            prediction_mean = lifted_reg_model(X[ind]).squeeze(-1)
            pyro.sample("obs", 
                    pyro.distributions.Normal(prediction_mean, noise_scale), 
                    obs = y[ind])

    def guide(self, X, y):
        softplus       = nn.Softplus()
        log_noise_mean = pyro.param("log_noise_mean", self.log_noise_mean * torch.ones(1))
        log_noise_std  = pyro.param("log_noise_std",  self.log_noise_std  * torch.ones(1), constraint = constraints.positive)
        noise_scale    = pyro.sample("noise_scale", pyro.distributions.LogNormal(log_noise_mean, log_noise_std))
        priors         = dict()
        for n, p in self.nn.named_parameters():
            if "weight" in n:
                loc   = pyro.param("mu_"    + n, self.weight_prior * torch.randn_like(p))
                scale = pyro.param("sigma_" + n, softplus(torch.randn_like(p)), constraint = constraints.positive)
                priors[n] = pyro.distributions.Normal(loc = loc, scale = scale).to_event(1)
            elif "bias" in n:
                loc       = pyro.param("mu_"    + n, self.bias_prior * torch.randn_like(p))
                scale     = pyro.param("sigma_" + n, softplus(torch.randn_like(p)), constraint = constraints.positive)
                priors[n] = pyro.distributions.Normal(loc = loc, scale = scale).to_event(1)
        lifted_module = pyro.random_module("module", self.nn, priors)
        return lifted_module()
            
    def train(self, X, y):
        num_train   = X.shape[0]
        y           = y.reshape(num_train)
        self.x_mean = X.mean(dim = 0)
        self.x_std  = X.std(dim = 0)
        self.y_mean = y.mean()
        self.y_std  = y.std()
        self.X      = (X - self.x_mean) / self.x_std
        self.y      = (y - self.y_mean) / self.y_std
        optim       = pyro.optim.Adam({"lr":self.lr})
        # optim       = pyro.optim.StepLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': self.lr}, 'step_size': int(self.num_iters/3), 'gamma': 0.1})
        svi         = pyro.infer.SVI(self.model, self.guide, optim, loss = pyro.infer.Trace_ELBO(), num_samples = 128)
        pyro.clear_param_store()
        self.rec = []
        for i in range(self.num_iters):
            loss = svi.step(self.X, self.y)
            if i % self.print_every == 0:
                self.rec.append(loss / num_train)
                print("[Iteration %05d] loss: %.4f" % (i + 1, loss / num_train))
    
    def sample(self):
        net = self.guide(self.X, self.y)
        return net

    def validate(self, test_x, test_y, n_samples = 100):
        num_test  = test_x.shape[0]
        preds     = torch.zeros(n_samples, num_test)
        for i in range(n_samples):
            nn = self.sample()
            preds[i, :] = nn((test_x - self.x_mean) / self.x_std).squeeze() * self.y_std + self.y_mean
        rmse = torch.sqrt(torch.mean((test_y - preds.mean(dim = 0))**2))
        return rmse

    def sample_predict(self, nn, x):
        return nn((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean
