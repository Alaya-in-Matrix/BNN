import sys, os
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
        logvars = torch.clamp(self.logvar, max = 20.) * out.new_ones(out.shape)
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
        self.use_cuda    = conf.get('use_cuda',False) and torch.cuda.is_available()
        self.nn          = NoisyNN(dim, self.act, self.num_hiddens)
        if self.use_cuda:
            self.cuda()

    def log_prior(self):
        """
        log(noise_var) \sim N(0, 1)
        w              \sim N(0, 1)
        """
        log_prior = 0.
        for p in self.nn.parameters():
            log_prior += -0.5 * torch.pow(p, 2).sum()
        return log_prior

    def sgld_steps(self, num_steps, num_train):
        step_cnt = 0
        loss     = 0.
        lr       = 0.
        while(step_cnt < num_steps):
            for bx, by in self.loader:
                log_prior = self.log_prior()
                nout      = self.nn(bx).squeeze()
                py        = nout[:, 0]
                logvar    = nout[:, 1]
                precision = 1 / (1e-8 + torch.exp(logvar))
                log_lik   = torch.sum(-0.5 * precision * (by - py)**2 - 0.5 * logvar)
                loss      = -1 * (log_lik * (num_train / bx.shape[0]) + log_prior)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.scheduler.step()

                for group in self.opt.param_groups: # Gaussian noise injection
                    for param in group["params"]:
                        lr     = group["lr"]
                        param.requires_grad = False
                        param += np.sqrt(2 * lr) * torch.randn(param.shape,device = param.device)
                        param.requires_grad = True
                step_cnt += 1
        return loss, lr

    def calc_ab(self, max_lr, min_lr, gamma, steps):
        ratio = min_lr / max_lr
        b     = steps  / (np.exp(np.log(ratio) / gamma) - 1)
        a     = max_lr / b**gamma
        return a, b

    def train(self, X, y):
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        self.normalize_Xy(X, y, self.normalize)
        num_train      = X.shape[0]
        self.opt       = optim.SGD(self.parameters(), lr = self.lr)

        gamma          = -0.55
        a, b           = self.calc_ab(self.lr, 1e-5, gamma, self.step_burnin + self.steps)
        schu_f         = lambda iter : (a * (b + iter)**gamma) / self.lr
        self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, schu_f)
        self.loader    = DataLoader(TensorDataset(self.X, self.y), batch_size = self.batch_size, shuffle = True)
        
        _ = self.sgld_steps(self.step_burnin, num_train)
        
        step_cnt = 0
        self.nns = []
        self.lrs = []
        while(step_cnt < self.steps):
            loss, lr  = self.sgld_steps(self.keep_every, num_train)
            step_cnt += self.keep_every
            print('Step %4d, loss = %8.2f, lr = %g' % (step_cnt, loss, lr),flush = True)
            self.nns.append(deepcopy(self.nn).cpu())
            self.lrs.append(lr)
        self.nn     = self.nn.cpu()
        self.x_mean = self.x_mean.cpu()
        self.x_std  = self.x_std.cpu()
        self.y_mean = self.y_mean.cpu()
        self.y_std  = self.y_std.cpu()
        print("Number of samples: %d" % len(self.nns))

    def sample(self, num_samples = 1):
        wprobs = torch.tensor(self.lrs) / torch.tensor(self.lrs).sum()
        dist   = torch.distributions.Categorical(wprobs)
        nns    = []
        for i in range(num_samples):
            nns.append(self.nns[dist.sample().item()])
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
            noise_var = (1e-8 + torch.exp(logvar)) * self.y_std**2
            pred[i]   = self.y_mean + py  * self.y_std
            prec[i]   = 1 / noise_var
        return pred, prec

    def report(self):
        print(self.nn)
