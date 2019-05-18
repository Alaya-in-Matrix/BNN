import sys, os
import torch
import torch.nn as nn
import numpy as np
from BNN_SGDMC import BNN_SGDMC
from platypus import NSGAII, MOEAD, CMAES, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation, Archive
import matplotlib.pyplot as plt

class BO:
    def __init__(self, f, dim, nobj, ncons, max_eval = 2, num_init = 1, act = nn.Tanh(), num_hiddens = [50], conf = {}):
        """
        f: objective function, with (-1, dim) tensor as input, with input range [0, 1]
        """
        self.f        = f
        self.dim      = dim
        self.nobj     = nobj
        self.ncons    = ncons
        self.max_eval = max_eval
        self.lb       = torch.tensor(-1.)
        self.ub       = torch.tensor(1.)
        self.bnn      = BNN_SGDMC(self.dim, act = act, num_hiddens = num_hiddens, conf = conf)
        self.X        = torch.rand(num_init, self.dim) * (self.ub - self.lb) + self.lb
        self.y        = self.f(self.X)
        assert(self.y.dim() == 2)
        assert(self.y.shape[1] == self.nobj + self.ncons)

    def see(self, nn_idxs):
        assert(self.dim == 1)
        assert(self.nobj + self.ncons == 1)
        xs = torch.linspace(self.lb, self.ub, 100).view(-1,1)
        with torch.no_grad():
            pred = self.bnn.sample_predict(self.bnn.nns, xs)
            pred = pred
        plt.plot(xs.numpy(), pred.squeeze().t().numpy(), 'g', alpha = 0.1)
        plt.plot(xs.numpy(), pred[nn_idxs].squeeze().t().numpy(), 'r')

        plt.plot(self.X.numpy(), self.y.numpy(), 'k+')
        plt.show()

    def train(self):
        self.bnn.train(self.X, self.y)
        for nn in self.bnn.nns:
            for p in nn.parameters():
                p.requires_grad = False
    

    def nn_opt(self, nn):
        with torch.no_grad():
            def obj_cons(x):
                tx  = torch.tensor(x)
                out = nn(tx) 
                return out[:self.nobj].numpy().tolist(), out[self.nobj:].numpy().tolist()
            def obj_ucons(x):
                tx  = torch.tensor(x)
                return nn(tx).numpy().tolist()

            arch = Archive()
            if self.ncons == 0:
                prob          = Problem(self.dim, self.nobj)
                prob.function = obj_ucons
            else:
                prob                = Problem(self.dim, self.nobj, self.ncons)
                prob.function       = obj_cons
                prob.constraints[:] = "<=0"
            prob.types[:] = [Real(self.lb, self.ub) for i in range(self.dim)]
            self.algo     = NSGAII(prob, population = 50, archive = arch)
            self.algo.run(5000)

            optimized   = self.algo.result
            rand_idx    = np.random.randint(len(optimized))
            suggested_x = torch.tensor(optimized[rand_idx].variables) 
            suggested_y = nn(suggested_x)
            return suggested_x.view(-1, self.dim), suggested_y.view(-1, self.nobj + self.ncons)
    
    def bo_iter(self, num_samples = 1):
        self.train()
        assert(num_samples <= len(self.bnn.nns))
        nn_idxs   = torch.randperm(len(self.bnn.nns))[:num_samples]
        suggested = torch.zeros(num_samples, self.dim)
        self.see(nn_idxs)
        for i in range(num_samples):
            sx, sy       = self.nn_opt(self.bnn.nns[i])
            suggested[i] = sx
        evaluated = self.f(suggested)
        self.X    = torch.cat((self.X, suggested))
        self.y    = torch.cat((self.y, evaluated))
        print(suggested)
        print(evaluated)
    
    def OSFTA(self):
        """
        One Sample to Find Them All 💍
        """
        for i in range(self.max_eval):
            self.bo_iter()
