import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Gamma
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np
from copy import deepcopy
from util import *
from BNN  import BNN
from torch.utils.data import TensorDataset, DataLoader
from pysgmcmc.optimizers.sgld  import SGLD
from pysgmcmc.optimizers.sghmc import SGHMC

class BNN_SGDMC(nn.Module, BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], nout = 1, conf = dict()):
        nn.Module.__init__(self)
        BNN.__init__(self)
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.nout         = nout
        self.steps_burnin = conf.get('steps_burnin', 2500)
        self.steps        = conf.get('steps',        2500)
        self.keep_every   = conf.get('keep_every',   50)
        self.batch_size   = conf.get('batch_size',   32)

        self.lr_weight = conf.get('lr_weight', 1e-3)
        self.lr_noise  = conf.get('lr_noise',  1e-5)
        self.lr_lambda = conf.get('lr_lambda', 1e-4)
        self.alpha_w   = torch.as_tensor(1.* conf.get('alpha_w', 20.))
        self.beta_w    = torch.as_tensor(1.* conf.get('beta_w',  20.))
        self.alpha_n   = torch.as_tensor(1.* conf.get('alpha_n', 20.))
        self.beta_n    = torch.as_tensor(1.* conf.get('beta_n',  20.))

        self.prior_log_lambda    = TransformedDistribution(Gamma(self.alpha_w, self.beta_w), ExpTransform().inv) # log of gamma distribution
        self.prior_log_precision = TransformedDistribution(Gamma(self.alpha_n, self.beta_n), ExpTransform().inv)

        self.log_lambda = nn.Parameter(torch.tensor(0.))
        self.log_precs  = nn.Parameter(torch.zeros(self.nout))
        self.nn         = NN(dim, self.act, self.num_hiddens, self.nout)
        
        self.init_nn()
    
    def init_nn(self):
        pass
        # self.log_lambda.data = self.prior_log_lambda.sample()
        # self.log_precs.data  = self.prior_log_precision.sample((self.nout, ))

    def log_prior(self):
        log_p  = self.prior_log_lambda.log_prob(self.log_lambda).sum()
        log_p += self.prior_log_precision.log_prob(self.log_precs).sum()

        lambd = self.log_lambda.exp()
        for n, p in self.nn.nn.named_parameters():
            if "weight" in n:
                log_p += -0.5 * lambd * torch.sum(p**2) + 0.5 * p.numel() * (self.log_lambda - np.log(2 * np.pi))
        return log_p

    def log_lik(self, X, y):
        y       = y.view(-1, self.nout)
        nout    = self.nn(X).view(-1, self.nout)
        precs   = self.log_precs.exp()
        log_lik = -0.5 * precs * (y - nout)**2 + 0.5 * self.log_precs - 0.5 * np.log(2 * np.pi)
        return log_lik.sum()

    def sgld_steps(self, num_steps, num_train):
        step_cnt = 0
        loss     = 0.
        while(step_cnt < num_steps):
            for bx, by in self.loader:
                log_prior = self.log_prior()
                log_lik   = self.log_lik(bx, by)
                loss      = -1 * (log_lik * (num_train / bx.shape[0]) + log_prior)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                step_cnt += 1
                # print('log_prior = %g, log_lik = %g' % (log_prior, log_lik))
        return loss

    def train(self, X, y):
        y           = y.view(-1, self.nout)
        num_train   = X.shape[0]
        params      = [
                {'params': self.nn.nn.parameters(), 'lr': self.lr_weight},
                {'params': self.log_precs,          'lr': self.lr_noise}, 
                {'params': self.log_lambda,         'lr': self.lr_lambda}]
        self.opt    = SGLD(params, num_burn_in_steps = 0)
        self.loader = DataLoader(TensorDataset(X, y), batch_size = self.batch_size, shuffle = True)
        
        _ = self.sgld_steps(self.steps_burnin, num_train) # burn-in
        
        step_cnt = 0
        self.nns = []
        self.lrs = []
        while(step_cnt < self.steps):
            loss      = self.sgld_steps(self.keep_every, num_train)
            step_cnt += self.keep_every
            prec      = self.log_precs.exp().mean()
            wstd      = 1 / self.log_lambda.exp().sqrt()
            print('Step %4d, loss = %8.2f, precision = %g, weight_std = %g' % (step_cnt, loss, prec, wstd),flush = True)
            self.nns.append(deepcopy(self.nn))
        print('Number of samples: %d' % len(self.nns))

    def sample(self, num_samples = 1):
        assert(num_samples <= len(self.nns))
        return np.random.permutation(self.nns)[:num_samples]

    def sample_predict(self, nns, input):
        num_samples = len(nns)
        num_x       = input.shape[0]
        pred        = torch.empty(num_samples, num_x, self.nout)
        for i in range(num_samples):
            pred[i] = nns[i](input)
        return pred

    def report(self):
        print(self.nn.nn)
