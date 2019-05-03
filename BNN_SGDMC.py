import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from util import NN
from BNN  import BNN
from torch.utils.data import TensorDataset, DataLoader

class NoisyNN(NN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], logvar = torch.tensor(0.)):
        super(NoisyNN, self).__init__(dim, act, num_hiddens, nout = 1)
        self.logvar = nn.Parameter(logvar)
    
    def forward(self, input):
        out     = self.nn(input)
        logvars = self.logvar * torch.ones(out.shape)
        return torch.cat((out, logvars), dim = out.dim() - 1)


class BNN_SGDMC(nn.Module, BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        nn.Module.__init__(self)
        BNN.__init__(self)
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.step_burnin = conf.get('steps_burnin', 10000)
        self.steps       = conf.get('steps', 10000)
        self.keep_every  = conf.get('keep_every', 200)
        self.normalize   = conf.get('normalize',  True)
        self.batch_size  = conf.get('batch_size', 64)
        self.lr          = conf.get('lr', 1e-3)
        self.nn          = NoisyNN(dim, self.act, self.num_hiddens)

    def log_prior(self):
        """
        log(noise_var) \sim N(0, 1)
        w              \sim N(0, 1)
        """
        log_prior = 0.
        for p in self.nn.parameters():
            log_prior += -0.5 * torch.pow(p, 2).sum()
        return log_prior

    def sgld_steps(self, num_steps, opt, loader, num_train):
        step_cnt = 0
        loss     = 0.
        while(step_cnt < num_steps):
            for bx, by in loader:
                log_prior = self.log_prior()
                nout      = self.nn(bx).squeeze()
                py        = nout[:, 0]
                logvar    = nout[:, 1]
                precision = torch.exp(-1 * logvar)
                log_lik   = torch.sum(-0.5 * precision * (by - py)**2 - 0.5 * logvar)
                loss      = -1 * (log_lik * (num_train / bx.shape[0]) + log_prior)
                opt.zero_grad()
                loss.backward()
                opt.step()

                for group in opt.param_groups: # Gaussian noise injection
                    for param in group["params"]:
                        lr     = group["lr"]
                        param.requires_grad = False
                        param += np.sqrt(2 * lr) * torch.randn(param.shape)
                        param.requires_grad = True
                step_cnt += 1
        return loss

    def train(self, X, y):
        self.normalize_Xy(X, y, self.normalize)
        num_train = X.shape[0]
        opt       = optim.SGD(self.parameters(), lr = self.lr)
        loader    = DataLoader(TensorDataset(self.X, self.y), batch_size = self.batch_size, shuffle = True)
        
        _ = self.sgld_steps(self.step_burnin, opt, loader, num_train)
        
        step_cnt = 0
        self.nns = []
        while(step_cnt < self.steps):
            loss      = self.sgld_steps(self.keep_every, opt, loader, num_train)
            step_cnt += self.keep_every
            print('Step %4d, loss = %6.2f' % (step_cnt, loss))
            self.nns.append(deepcopy(self.nn))

    def sample(self, num_samples = 1):
        assert(len(self.nns) >= num_samples)
        return self.nns[:num_samples]

    def sample_predict(self, nns, X):
        num_x     = X.shape[0]
        X         = (X - self.x_mean) / self.x_std
        pred      = torch.zeros(len(nns), num_x)
        prec = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            nn_out    = nns[i](X)
            py        = nn_out[:, 0]
            logvar    = nn_out[:, 1]
            noise_var = torch.exp(logvar) * self.y_std**2
            pred[i]   = self.y_mean + py  * self.y_std
            prec[i]   = 1 / noise_var
        return pred, prec

    def report(self):
        print(self.nn)
