import torch
import torch.nn as nn
import torch.nn.functional as F

class DEMO:
    def __init__(self, f, lb, ub, n_out, n_obj, gen = 100, np = 100, F = 0.8, CR = 0.8):
        """
        f: vector-valued fucntion
        lb: lower bound
        ub: upper bound
        n_out: number of the output values of `f`
        n_obj: number of constraints, `n_out` > `n_obj`, the rest values are constraints
        """
        self.func     = f
        self.lb       = torch.as_tensor(lb)
        self.ub       = torch.as_tensor(ub)
        self.dim      = len(lb)
        self.n_out    = n_out
        self.n_obj    = n_obj
        self.max_eval = max_eval
        self.np       = np
        self.f        = f
        self.cr       = cr
        self.pop_x    = torch.lb + torch.rand(self.np, self.dim) * (self.ub - self.lb)
        self.pop_y    = f(self.pop_x)
        self.pf_x     = None
        self.pf_y     = None
    
    def compare(self, y1, y2):
        """
        Whether or not y1 is better than y2
        Return:
             1:  y1 better than y2
            -1:  y1 worse  than y2
             0:  can't tell which is better
        """
        obj1 = y1[:self.n_obj]
        obj2 = y2[:self.n_obj]
        vio1 = y1[self.n_obj:].clamp(min = 0.).sum()
        vio2 = y2[self.n_obj:].clamp(min = 0.).sum()
        if vio1 == 0 and vio2 == 0: # both satisfy constraints
            cmp = obj1 <= obj2
            if cmp.all():
                return 1 # y1 is strictly better
            elif cmp.any():
                return 0
            else:
                return -1
        return torch.sign(vio2 - vio1)

    def extract_pf(self, ys):
        num_data = len(ys)
        mask     = torch.zeros(num_data)
        pf_idx   = []
        for i in range(num_data):
            yi        = ys[i]
            dominated = False
            if not mask[i]:
                for j in range(num_data):
                    yj  = ys[j]
                    cmp = self.compare(yi, yj)
                    if cmp == -1:
                        dominated = True
                        break
                    if cmp == 1:
                        mask[j] = 1.
            if not dominated:
                pf_idx.append(i)
        return pf_idx

    def nd_sorting(self, ys):
        pass

    def optimize(self):
        for i in range(self.gen):
            mutated = self.mutation()
            child   = self.crossover(mutated)
            res     = self.f(child)
