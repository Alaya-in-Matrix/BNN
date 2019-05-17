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
        self.bnn      = BNN_SGDMC(self.dim, act = act, num_hiddens = num_hiddens, conf = conf)
        self.X        = torch.rand(num_init, self.dim)
        self.y        = self.f(self.X)
        assert(self.y.dim() == 2)
        assert(self.y.shape[1] == self.nobj + self.ncons)

    def see(self):
        assert(self.dim == 1)
        assert(self.nobj + self.ncons == 1)
        xs = torch.linspace(0.,1.,100).view(-1,1)
        with torch.no_grad():
            pred = self.bnn.sample_predict(self.bnn.nns, (xs - self.x_mean) / self.x_std)
            pred = pred * self.y_std + self.y_mean
        plt.plot(xs.numpy(), pred.squeeze().t().numpy(), 'g', alpha = 0.1)

        plt.plot(self.X.numpy(), self.y.numpy(), 'k+')
        plt.show()

    def normalize(self):
        self.x_mean = torch.zeros(self.dim)
        self.x_std  = torch.ones(self.dim)
        self.y_mean = torch.zeros(self.nobj + self.ncons)
        self.y_std  = torch.ones(self.nobj + self.ncons)
        # if self.X.shape[0] == 1:
        #     self.x_mean = torch.zeros(self.dim)
        #     self.x_std  = torch.ones(self.dim)
        #     self.y_mean = torch.zeros(self.nobj + self.ncons)
        #     self.y_std  = torch.ones(self.nobj + self.ncons)
        # else:
        #     self.x_mean = self.X.mean(dim = 0)
        #     self.x_std  = self.X.std(dim  = 0)
        #     self.y_mean = self.y.mean(dim = 0)
        #     self.y_std  = self.y.std(dim  = 0)
        X           = (self.X - self.x_mean) / self.x_std
        y           = (self.y - self.y_mean) / self.y_std
        self.n_lb   = (torch.zeros(self.dim) - self.x_mean) / self.x_std # normalized lower bound
        self.n_ub   = (torch.ones(self.dim)  - self.x_mean) / self.x_std
        return X, y

    def train(self):
        X, y = self.normalize()
        self.bnn.train(X, y)
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

            lb = (torch.zeros(self.dim) - self.x_mean) / self.x_std
            ub = (torch.ones(self.dim)  - self.x_mean) / self.x_std

            arch = Archive()
            if self.ncons == 0:
                prob          = Problem(self.dim, self.nobj)
                prob.function = obj_ucons
            else:
                prob                = Problem(self.dim, self.nobj, self.ncons)
                prob.function       = obj_cons
                prob.constraints[:] = "<=0"
            prob.types[:] = [Real(lb[i], ub[i]) for i in range(self.dim)]
            self.algo     = NSGAII(prob, population = 50, archive = arch)
            self.algo.run(5000)

            optimized   = self.algo.population
            rand_idx    = np.random.randint(len(optimized))
            suggested_x = torch.tensor(optimized[rand_idx].variables) 
            suggested_y = nn(suggested_x)
            suggested_x = suggested_x * self.x_std + self.x_mean
            suggested_y = suggested_y * self.y_std + self.y_mean
            return suggested_x.view(-1, self.dim), suggested_y.view(-1, self.nobj + self.ncons)
    
    def bo_iter(self, num_samples = 1):
        self.train()
        self.see()
        assert(num_samples <= len(self.bnn.nns))
        nn_idxs   = torch.randperm(len(self.bnn.nns))[:num_samples]
        suggested = torch.zeros(num_samples, self.dim)
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
