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
from pybnn.sampler.sgld import SGLD
from pybnn.sampler.preconditioned_sgld import PreconditionedSGLD as pSGLD
from pybnn.sampler.sghmc import SGHMC as SGHMC
from pybnn.sampler.adaptive_sghmc import AdaptiveSGHMC as aSGHMC

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
        self.warm_start   = conf.get('warm_start',   False)

        self.lr_weight   = conf.get('lr_weight', 2e-2)
        self.lr_noise    = conf.get('lr_noise',  1e-1)
        self.alpha_n     = torch.as_tensor(1.* conf.get('alpha_n', 1e-2))
        self.beta_n      = torch.as_tensor(1.* conf.get('beta_n',  1e-1))

        # user can specify a suggested noise value, this will override alpha_n and beta_n
        self.noise_level = conf.get('noise_level', None) 
        if self.noise_level is not None:
            prec         = 1 / self.noise_level**2
            prec_var     = (prec * 0.25)**2
            self.beta_n  = torch.as_tensor(prec / prec_var)
            self.alpha_n = torch.as_tensor(prec * self.beta_n)
            print("Reset alpha_n = %g, beta_n = %g" % (self.alpha_n, self.beta_n))

        self.prior_log_precision = TransformedDistribution(Gamma(self.alpha_n, self.beta_n), ExpTransform().inv)

        self.log_precs = nn.Parameter(torch.zeros(self.nout))
        self.nn        = NN(dim, self.act, self.num_hiddens, self.nout)
        self.gain      = 5./3 # Assume tanh activation
        
        self.init_nn()
    
    def init_nn(self):
        self.log_precs.data  = (self.alpha_n / self.beta_n).log() * torch.ones(self.nout)
        for l in self.nn.nn:
            if type(l) == nn.Linear:
                nn.init.xavier_uniform_(l.weight, gain = self.gain)

    def log_prior(self):
        log_p = self.prior_log_precision.log_prob(self.log_precs).sum()
        for n, p in self.nn.nn.named_parameters():
            if "weight" in n:
                std    = self.gain * np.sqrt(2. / (p.shape[0] + p.shape[1]))
                log_p += torch.distributions.Normal(0, std).log_prob(p).sum()
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
                self.scheduler.step()
                step_cnt += 1
                if step_cnt >= num_steps:
                    break
        return loss

    def train(self, X, y):
        y           = y.view(-1, self.nout)
        num_train   = X.shape[0]
        params      = [
                {'params': self.nn.nn.parameters(), 'lr': self.lr_weight},
                {'params': self.log_precs,          'lr': self.lr_noise}] 

        self.opt       = pSGLD(params)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, lambda iter : np.float32((1 + iter)**-0.33))
        # XXX: learning rate scheduler, as suggested in Teh, Yee Whye,
        # Alexandre H. Thiery, and Sebastian J. Vollmer. "Consistency and
        # fluctuations for stochastic gradient Langevin dynamics." The Journal
        # of Machine Learning Research 17.1 (2016): 193-225.
        
        # XXX: I'm not sure if this scheduler is still optimal for preconditioned SGLD


        self.loader    = DataLoader(TensorDataset(X, y), batch_size = self.batch_size, shuffle = True)
        step_cnt    = 0
        self.nns    = []
        self.lrs    = []
        if not self.warm_start:
            self.init_nn()
        
        _ = self.sgld_steps(self.steps_burnin, num_train) # burn-in
        
        while(step_cnt < self.steps):
            loss      = self.sgld_steps(self.keep_every, num_train)
            step_cnt += self.keep_every
            prec      = self.log_precs.exp().mean()
            print('Step %4d, loss = %8.2f, precision = %g' % (step_cnt, loss, prec), flush = True)
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
