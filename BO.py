import torch
import torch.nn
import numpy as np
from BNN_SGDMC import BNN_SGDMC

class BO:
    def __init__(f, lb, ub, nobj, ncons, max_eval = 2, num_init = 1, conf = {}):
        """
        f: objective function, with (-1, dim) tensor as input
        """
        self.f        = f
        self.lb       = torch.as_tensor(lb)
        self.ub       = torch.as_tensor(ub)
        self.dim      = len(lb)
        self.nobj     = nobj
        self.ncons    = ncons
        self.max_eval = max_eval
        self.bnn      = BNN_SGDMC(self.dim, act = nn.Tanh(), [50, 50], conf)
        self.X        = torch.randn(num_init, self.dim) * (self.ub - self.lb) + self.lb
        self.y        = self.f(self.X)
        assert(self.y.dim() == 2)
        assert(self.y.shape[1] == self.nobj + self.ncons)

    def normalize(self):
        self.x_mean = self.X.mean(dim = 1)
        self.x_std  = self.X.std(dim = 1)
        self.y_mean = self.y.mean(dim = 1)
        self.y_std  = self.y.std(dim = 1)
        X           = (self.X - x_mean) / self.x_std
        y           = (self.y - y_mean) / self.y_std
        return X, y

    def train(self):
        X, y = self.normalize(self.X, self.y)
        self.bnn.train(X, y)

    def nn_opt(self, nn):
        pass
    
    def bo_iter(self, num_samples = 1):
        assert(num_samples <= self.bnn.num_samples)
        self.train()
        nn_idxs   = torch.randperm(self.bnn.num_samples)[:num_sample]
        nns       = self.bnn.nns[nn_idxs]
        suggested = torch.tensor([self.nn_opt(nn) for nn in nns])
        evaluated = self.f(suggested)
        self.X    = torch.cat((self.X, suggested))
        self.y    = torch.cat((self.y, evaluated))
    
    def OSFTA(self):
        """
        One Sample to Find Them All 💍
        """
        for i in range(self.max_eval):
            self.bo_iter()
