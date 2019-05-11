import sys
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import numpy               as np
from BNN import BNN
from torch.utils.data import TensorDataset, DataLoader
from util import *

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer  = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min     = np.log(init_min) - np.log(1. - init_min)
        init_max     = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def extra_repr(self):
        return 'dropout_rate = %g' % torch.sigmoid(self.p_logit)
        
    def forward(self, x, layer):
        p   = torch.sigmoid(self.p_logit)
        out = layer(self._concrete_dropout(x, p))
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer   = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer   = p * torch.log(p) 
        dropout_regularizer  += (1. - p) * torch.log(1. - p)
        
        input_dimensionality  = x[0].numel() # Number of elements of first item in batch
        dropout_regularizer  *= self.dropout_regularizer * input_dimensionality
        
        regularization        = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps        = 1e-7
        temp       = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob  = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob      = torch.sigmoid(drop_prob / temp)
        random_tensor  = 1 - drop_prob
        x              = torch.mul(x, random_tensor)
        retain_prob    = 1 - p
        x             /= retain_prob
        return x

class GalConcreteDropoutImp(nn.Module, BNN):
    # def __init__(self, dimnb_features, weight_regularizer, dropout_regularizer):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(GalConcreteDropoutImp, self).__init__()
        BNN.__init__(self)
        self.dim         = dim
        self.nb_features = num_hiddens[0]
        self.wr          = conf.get('wr', 0.)
        self.dr          = conf.get('dr', 0.)
        self.num_epochs  = conf.get('num_epochs',   400)
        self.lr          = conf.get('lr',           1e-3)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  100)
        self.normalize   = conf.get('normalize',    True)

        self.linear1          = nn.Linear(self.dim, self.nb_features)
        self.linear_mu        = nn.Linear(self.nb_features, 1)
        self.linear_logvar    = nn.Linear(self.nb_features, 1)
        self.conc_drop1       = ConcreteDropout(weight_regularizer=self.wr, dropout_regularizer=self.dr)
        self.conc_drop_mu     = ConcreteDropout(weight_regularizer=self.wr, dropout_regularizer=self.dr)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=self.wr, dropout_regularizer=self.dr)
        self.act = nn.Tanh()
        
    def forward(self, x):
        regularization             = torch.zeros(3, device=x.device)
        x,       regularization[0] = self.conc_drop1(x,       nn.Sequential(self.linear1, self.act))
        mean,    regularization[1] = self.conc_drop_mu(x,     self.linear_mu)
        log_var, regularization[2] = self.conc_drop_logvar(x, self.linear_logvar)
        return mean.squeeze(), log_var.squeeze(), regularization.sum()

    def heteroscedastic_loss(self, true, mean, log_var):
        true      = true.squeeze()
        mean      = mean.squeeze()
        log_var   = log_var.squeeze()
        precision = torch.exp(-log_var)
        return torch.mean(precision * (true - mean)**2 + log_var)

    def train(self, X, Y):
        N = X.shape[0]
        Y = Y.reshape(Y.numel())
        self.normalize_Xy(X, Y, self.normalize)
        batch_size  = 64
        optimizer   = optim.Adam(self.parameters(), lr = self.lr)
        
        for i in range(self.num_epochs):
            old_batch = 0
            for batch in range(int(np.ceil(X.shape[0]/self.batch_size))):
                batch = (batch + 1)
                _x = self.X[old_batch: self.batch_size*batch]
                _y = self.y[old_batch: self.batch_size*batch]
                x  = torch.autograd.Variable(torch.FloatTensor(_x))
                y  = torch.autograd.Variable(torch.FloatTensor(_y))
                mean, log_var, regularization = self.forward(x)
                loss = self.heteroscedastic_loss(y, mean, log_var) + regularization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i % 100 == 0:
                print('Epoch %d, loss = %.2f' % (i, loss))

    def sample(self, num_samples = 1):
        return [None for i in range(num_samples)]

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        X     = (X - self.x_mean) / self.x_std
        pred  = torch.zeros(len(nns), num_x)
        prec  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            py, log_var, _ = self.forward(X)
            noise_var      = torch.exp(log_var) * self.y_std**2
            pred[i]        = self.y_mean + py   * self.y_std
            prec[i]        = 1 / noise_var
        return pred, prec

    def report(self):
        print(self)
