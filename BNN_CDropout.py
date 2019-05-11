import sys, os
import torch
import torch.optim         as optim
import torch.nn            as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy               as np
from util import StableRelaxedBernoulli
from BNN  import BNN
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.utils import clamp_probs, probs_to_logits, logits_to_probs

class CDropout(nn.Module):
    def __init__(self, p = 0.5):
        super(CDropout, self).__init__()
        self.p_logit = nn.Parameter(probs_to_logits(torch.as_tensor(p), is_binary = True))
    
    def dropout_rate(self):
        return clamp_probs(torch.sigmoid(self.p_logit))

    def forward(self, input):
        # bdist = StableRelaxedBernoulli(probs = 1 - self.dropout_rate(), temperature = 0.1)
        # return input * bdist.rsample(input.shape)
        p = self.dropout_rate()
        eps        = 1e-7
        temp       = 0.1
        unif_noise = torch.rand_like(input)
        drop_prob  = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob      = torch.sigmoid(drop_prob / temp)
        random_tensor  = 1 - drop_prob
        
        return input * random_tensor

    def extra_repr(self):
        return 'dropout_rate = {}'.format(self.dropout_rate())


class CDropoutLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CDropoutLinear, self).__init__()
        self.layer = nn.Sequential(
                CDropout(), 
                nn.Linear(in_features, out_features))

    def forward(self, input):
        return self.layer(input)

    def dropout_rate(self):
        return self.layer[0].dropout_rate()

    def reg(self):
        p       = self.layer[0].dropout_rate()
        entropy = -1 * (p * p.log() + (1-p) * (1-p).log())
        w2      = torch.sum(self.layer[1].weight**2)
        return w2, torch.tensor(1.*self.layer[1].in_features) * entropy

    def sample(self):
        dropout_rate        = self.layer[0].dropout_rate()
        linear              = nn.Linear(self.layer[1].in_features, self.layer[1].out_features)
        linear.weight.data  = self.layer[1].weight.data.clone()
        linear.bias.data    = self.layer[1].bias.data.clone()
        linear.weight.data *= StableRelaxedBernoulli(probs = 1 - dropout_rate, temperature = 0.1).sample((linear.in_features, ))
        return linear

class NN_CDropout(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50]):
        super(NN_CDropout, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_layers  = len(num_hiddens)
        self.nn          = self.mlp()

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(CDropoutLinear(pre_dim, self.num_hiddens[i]))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(CDropoutLinear(pre_dim, 1))
        return nn.Sequential(*layers)

    def sample(self):
        layers = []
        for layer in self.nn:
            layers.append(layer.sample() if isinstance(layer, CDropoutLinear) else layer)
        return nn.Sequential(*layers)

    def forward(self, input):
        return self.nn(input)

class BNN_CDropout(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_CDropout, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_epochs  = conf.get('num_epochs',   400)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  100)
        self.normalize   = conf.get('normalize',    True)
        self.fixed_noise = conf.get('fixed_noise',  None)

        self.lr          = conf.get('lr',       1e-2)
        self.lscale      = conf.get('lscale',   1e-1)  # prior for weight: w ~ N(0, I/lscale^2)
        self.dr          = conf.get('dr',       1.)    # XXX: this hyper-parameter shouldn't exist in a full bayesian setting with fixed noise

        self.nn          = NN_CDropout(dim, self.act, self.num_hiddens)
        self.noise_level = 1. if self.fixed_noise is None else self.fixed_noise

    def train(self, X, y):
        num_train        = X.shape[0]
        opt              = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        dataset          = TensorDataset(X, y)
        loader           = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        crit             = nn.MSELoss(reduction = 'sum')
        for epoch in range(self.num_epochs):
            epoch_mse  = 0.
            epoch_wreg = 0.
            epoch_ent  = 0.
            for bx, by in loader:
                opt.zero_grad()
                py        = self.nn(bx).squeeze()
                mse_loss  = crit(py, by)
                wreg, ent = self.reg()
                wreg      *= bx.shape[0] / num_train
                ent       *= bx.shape[0] / num_train
                loss       = mse_loss + (wreg - ent)
                loss.backward()
                opt.step()
                epoch_mse  += mse_loss
                epoch_wreg += wreg
                epoch_ent  += ent
            if self.fixed_noise is None:
                self.noise_level = np.sqrt(epoch_mse.detach().numpy() / num_train)
            if (epoch + 1) % self.print_every == 0:
                print("Epoch %4d, mse = %g, noise = %g, wreg = %g, -entropy = %g" % (epoch+1, epoch_mse / num_train, self.noise_level, epoch_wreg / num_train, -1 * epoch_ent / num_train))
    
    def reg(self):
        entropy    = 0.
        weight_reg = 0.
        noise_var  = self.noise_level**2
        for layer in self.nn.nn:
            if isinstance(layer, CDropoutLinear):
                prob        = 1 - layer.dropout_rate()
                w2, ent     = layer.reg()
                weight_reg += self.lscale**2 * prob * noise_var * w2
                entropy    += self.dr * 2 * noise_var * ent
        return weight_reg, entropy


    def sample(self, num_samples = 1):
        nns = [self.nn.sample() for i in range(num_samples)]
        return nns

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        pred  = torch.zeros(len(nns), num_x)
        prec  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            pred[i] = nns[i](X).squeeze()
        prec = torch.ones(pred.shape) / (self.noise_level**2)
        return pred, prec

    def report(self):
        print(self.nn.nn)
