import torch
import torch.nn
import numpy as np
from BNN_SGDMC import BNN_SGDMC
from platypus import NSGAII, MOEAD, CMAES, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation, Archive

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
        with torch.no_grad():
            def obj_cons(x):
                tx  = torch.tensor(x)
                out = nn(tx)
                return out[:self.nobj].numpy().tolist(), out[self.nobj:].numpy().tolist()
            def obj_ucons(x):
                tx  = torch.tensor(x)
                return nn(tx).numpy().tolist()

            arch          = Archive()
            if self.ncons == 0:
                prob          = Problem(self.dim, self.nobj)
                prob.function = obj_ucons
            else:
                prob                = Problem(self.dim, self.nobj, self.ncons)
                prob.function       = obj_cons
                prob.constraints[:] = "<=0"
            prob.types[:] = [Real(self.lb[i], self.ub[i]) for i in range(self.dim)]
            algo          = CMAES(prob, population = 100, archive = arch)
            def cb(a):
                print(a.nfe, len(a.archive), flush=True)
            algo.run(100, callback = cb)

            optimized = algorithm.population
            rand_idx  = np.random.randint(len(optimized))
            suggested = torch.tensor(optimized[rand_idx].variables)
            return suggested
    
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
