import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from BNN  import BNN
from util import NoisyNN
from torch.utils.data import TensorDataset, DataLoader
from pysgmcmc.data.utils import infinite_dataloader
# from pysgmcmc.optimizers.sgld import SGLD
from SGLD import SGLD
from copy import deepcopy
from tqdm import tqdm

class BNN_SMC(nn.Module, BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        nn.Module.__init__(self)
        BNN.__init__(self)
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.batch_size   = conf.get('batch_size',   32)
        self.lr           = conf.get('lr',           1e-3)
        self.wprior       = conf.get('wprior',       1.)
        self.logvar_prior = conf.get('logvar_prior', 1)
        self.num_samples  = conf.get('num_samples',  50)
        self.normalize    = conf.get('normalize',    True) # XXX: only usefull for offline training
        self.mcmc_epochs  = conf.get('mcmc_epochs',  40)
        self.X            = None
        self.y            = None
        self.init_nns()
    
    def init_nns(self):
        self.nns = []
        for i in range(self.num_samples):
            net = NoisyNN(self.dim, self.act, self.num_hiddens)
            net.logvar.data = self.logvar_prior * torch.randn(1)
            for layer in net.nn:
                if isinstance(layer, nn.Linear):
                    layer.weight.data = self.wprior * torch.randn(layer.weight.shape)
                    layer.bias.data   = torch.zeros(layer.bias.shape)
            self.nns.append(net)

    def log_prior(self, nn):
        """
        log(noise_var) \sim N(0, 1)
        w              \sim N(0, 1)
        """
        log_prior = -0.5 * (nn.logvar**2 / self.logvar_prior**2)
        for p in nn.nn.parameters():
            log_prior += -0.5 * (p**2 / self.wprior**2).sum()
        return log_prior

    def log_lik(self, net, x, y):
        nn_out    = net(x)
        py        = nn_out[:, 0]
        logvar    = nn_out[:, 1]
        precision = 1 / (torch.exp(logvar) + 1e-16)
        log_lik   = -0.5 * precision * (py - y.squeeze())**2 - 0.5 * logvar
        return log_lik.sum()
    
    def posterior(self, net, x, y):
        return self.log_lik(net, x, y) + self.log_prior(net)

    def reweighting(self, new_x, new_y):
        """
        Generate weight according to the likelihood of (new_x, new_y)
        new_x: vector with length = self.dim
        new_y: scalar target vector
        """
        log_lik = torch.tensor([self.log_lik(nn, new_x, new_y) for nn in self.nns])
        weights = torch.exp(log_lik - log_lik.max())
        weights = weights / weights.sum()
        return weights, log_lik

    def ess(self, weights):
        return weights.sum()**2 / torch.sum(weights**2)

    def resample(self, weights):
        """
        Resample according to weights
        """
        assert(len(weights) == len(self.nns))
        dist     = torch.distributions.Categorical(probs = weights)
        new_nns  = [deepcopy(self.nns[dist.sample().item()]) for i in range(len(self.nns))]
        self.nns = new_nns

    def sgld_update(self, epoch_factor):
        if not self.X is None:
            bs              = 4 if self.X.shape[0] < self.batch_size else self.batch_size
            loader          = infinite_dataloader(DataLoader(TensorDataset(self.X, self.y), batch_size = bs, shuffle = True))
            self.sgld_steps = int(self.mcmc_epochs * epoch_factor * self.X.shape[0] / bs)
            for nn in tqdm(self.nns):
                opt      = SGLD(nn.parameters(), lr = self.lr / (1 + self.X.shape[0]))
                step_cnt = 0
                for bx, by in loader:
                    log_lik   = self.log_lik(nn, bx, by) * self.X.shape[0] / bx.shape[0]
                    log_prior = self.log_prior(nn)
                    loss      = -1 * log_lik - log_prior
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    step_cnt += 1
                    if step_cnt >= self.sgld_steps:
                        break
    
    def train(self, _X, _y):
        self.normalize_Xy(_X, _y, self.normalize)
        X               = self.X.clone()
        y               = self.y.clone()
        self.X          = None
        self.y          = None
        num_train       = X.shape[0]
        tbar            = tqdm(range(num_train))
        self.sgld_steps = 0
        fid = open('train.log', 'w')
        for i in tqdm(range(num_train)):
            new_x           = X[i].unsqueeze(0)
            new_y           = y[i].unsqueeze(0)
            weights, _      = self.reweighting(new_x, new_y)
            ess             = self.ess(weights)
            self.resample(weights)
            self.sgld_update(self.num_samples / ess)
            if self.X is None:
                self.X = new_x
                self.y = new_y
            else:
                self.X = torch.cat((self.X, new_x))
                self.y = torch.cat((self.y, new_y))
            rmse, nll_g, nll = self.validate(_X, _y)
            tbar.set_description('%d, ESS = %.2f, SGLD steps = %d, NLL = %g, SMSE = %g' % (i, ess, self.sgld_steps, nll, rmse**2 / _y.var()))  
            fid.write('%d, ESS = %.2f, SGLD steps = %d, NLL = %g, SMSE = %g\n' % (i, ess, self.sgld_steps, nll, rmse**2 / _y.var()))  
        fid.close()

    def active_train(self, X, y, max_train = 100):
        pass
        # tbar      = tqdm(range(num_train))
        # for i in tqdm(range(num_train)):
        #     new_x      = X[i].unsqueeze(0)
        #     new_y      = y[i].unsqueeze(0)
        #     weights, _ = self.reweighting(new_x, new_y)
        #     self.resample(weights)
        #     self.sgld_update()
        #     if self.X is None:
        #         self.X = new_x
        #         self.y = new_y
        #     else:
        #         self.X = torch.cat((self.X, new_x))
        #         self.y = torch.cat((self.y, new_y))
        #     rmse, nll_g, nll = self.validate(_X, _y)
        #     tbar.set_description('%d, ESS = %.2f, NLL = %g, SMSE = %g' % (i, self.ess(weights), nll, rmse**2 / _y.var()))  


    def sample(self, num_samples = 1):
        assert(num_samples <= len(self.nns))
        return self.nns[:num_samples]

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
        noise_vars = torch.tensor([nn.logvar.exp() for nn in self.nns]).mean()
        print(self.nns[0])
        print("Number of samples: %d" % len(self.nns))
