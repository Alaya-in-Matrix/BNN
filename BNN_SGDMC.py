import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from util import *
from BNN  import BNN
from torch.utils.data import TensorDataset, DataLoader
from pysgmcmc.optimizers.sgld  import SGLD
from pysgmcmc.optimizers.sghmc import SGHMC
from SGLD import SGLD as MySGLD


class BNN_SGDMC(nn.Module, BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        nn.Module.__init__(self)
        BNN.__init__(self)
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.steps_burnin = conf.get('steps_burnin', 3000)
        self.steps        = conf.get('steps',        10000)
        self.keep_every   = conf.get('keep_every',   100)
        self.normalize    = conf.get('normalize',    True)
        self.batch_size   = conf.get('batch_size',   32)

        self.lr_weight    = conf.get('lr_weight',    1e-3)
        self.lr_noise     = conf.get('lr_noise',     1e-5)
        self.weight_std   = conf.get('weight_std',   1.)
        self.logvar_std   = conf.get('logvar_std',   1.)
        self.logvar_mean  = conf.get('logvar_mean',  -1)

        self.nn           = NoisyNN(dim, self.act, self.num_hiddens)

    def log_prior(self):
        log_prior = -0.5 * torch.pow((self.nn.logvar - self.logvar_mean) / self.logvar_std, 2)
        for n, p in self.nn.nn.named_parameters():
            if "weight" in n:
                log_prior += -0.5 * (p**2 / self.weight_std**2).sum()
            elif "bias" in n: # do not regularize the bias
                log_prior += -0.5 * (p**2 / (100*self.weight_std)**2).sum()
        return log_prior

    def sgld_steps(self, num_steps, num_train):
        step_cnt = 0
        loss     = 0.
        while(step_cnt < num_steps):
            for bx, by in self.loader:
                log_prior = self.log_prior()
                nout      = self.nn(bx).squeeze()
                log_lik   = stable_nn_lik(nout, by.squeeze()).sum()
                loss      = -1 * (log_lik * (num_train / bx.shape[0]) + log_prior)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                step_cnt += 1
        return loss

    def train(self, X, y):
        self.normalize_Xy(X, y, self.normalize)
        num_train      = X.shape[0]
        params      = [{'params': self.nn.nn.parameters(), 'lr': self.lr_weight}, {'params': self.nn.logvar, 'lr': self.lr_noise}]
        self.opt    = SGLD(params, num_burn_in_steps = 0)
        # self.opt    = MySGLD(params)
        self.loader = DataLoader(TensorDataset(self.X, self.y), batch_size = self.batch_size, shuffle = True)
        
        _ = self.sgld_steps(self.steps_burnin, num_train)
        
        step_cnt = 0
        self.nns = []
        self.lrs = []
        while(step_cnt < self.steps):
            loss      = self.sgld_steps(self.keep_every, num_train)
            step_cnt += self.keep_every
            print('Step %4d, loss = %8.2f, noise_var = %.2f' % (step_cnt, loss, stable_noise_var(self.nn.logvar) * self.y_std**2),flush = True)
            self.nns.append(deepcopy(self.nn))
        print('Number of samples: %d' % len(self.nns))

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
            noise_var = stable_noise_var(logvar) * self.y_std**2
            pred[i]   = self.y_mean + py  * self.y_std
            prec[i]   = 1 / noise_var
        return pred, prec

    def report(self):
        print(self.nn)
