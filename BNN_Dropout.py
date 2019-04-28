import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from BNN import BNN
from torch.utils.data import TensorDataset, DataLoader
from util import NN
from copy import deepcopy

class NN_Dropout(NN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], dropout_rate = 0.05):
        super(NN_Dropout, self).__init__(dim, act, num_hiddens)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        scale_factor = 1 - self.dropout_rate
        for l in self.nn:
            if type(l) == nn.Linear and x.shape[1] > 1:
                x = F.dropout(x, p = self.dropout_rate, training = self.training) * scale_factor
            x = l(x)
        return x

class BNN_Dropout(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        super(BNN_Dropout, self).__init__()
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_epochs   = conf.get('num_epochs',   40)
        self.dropout_rate = conf.get('dropout_rate', 0.05)
        self.tau          = conf.get('tau',          1.0)
        self.lscale       = conf.get('lscale',       1e-2)
        self.lr           = conf.get('lr',           1e-3)
        self.batch_size   = conf.get('batch_size',   128)
        self.print_every  = conf.get('print_every',  100)
        self.normalize    = conf.get('normalize',    True)
        self.nn           = NN_Dropout(dim, self.act, self.num_hiddens, self.dropout_rate)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()
    
    def train(self, X, y):
        num_train = X.shape[0]
        y         = y.reshape(num_train)
        self.normalize_Xy(X, y, self.normalize)
        if torch.cuda.is_available():
            self.X = self.X.cuda()
            self.y = self.y.cuda()
        self.l2_reg   = self.lscale**2 * (1 - self.dropout_rate) / (2. * num_train * self.tau) # XXX: tau should also be normalized!
        criterion     = nn.MSELoss()
        dict_decay    = {'params':[], 'weight_decay': self.l2_reg}
        dict_no_decay = {'params':[], 'weight_decay': 0.}
        for name, param in self.nn.named_parameters():
            if "bias" in name:
                dict_no_decay['params'].append(param)
            else:
                dict_decay['params'].append(param)
        opt     = torch.optim.Adam([dict_decay, dict_no_decay], lr = self.lr)
        dataset = TensorDataset(self.X, self.y)
        loader  = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        for epoch in range(self.num_epochs):
            for bx, by in loader:
                opt.zero_grad()
                pred = self.nn(bx).reshape(by.shape)
                loss = criterion(pred, by)
                loss.backward()
                opt.step()
            if (epoch + 1) % self.print_every == 0:
                print("[Epoch %5d, loss = %g]" % (epoch + 1, loss))
        self.nn = self.nn.cpu()
        if self.normalize:
            self.x_mean  = self.x_mean.cpu()
            self.x_std   = self.x_std.cpu()
            self.y_mean  = self.y_mean.cpu()
            self.y_std   = self.y_std.cpu()

    def sample(self, num_samples = 1):
        nns = []
        bs  = torch.distributions.Bernoulli(1 - self.dropout_rate)
        for i in range(num_samples):
            net = deepcopy(self.nn.nn)
            for layer in net:
                if isinstance(layer, nn.Linear) and layer.weight.shape[1] > 1:
                    layer.weight.data *= F.dropout(torch.ones(layer.weight.shape[1]), p = self.dropout_rate) * (1 - self.dropout_rate)
            nns.append(net)
        return nns, self.tau * torch.ones(num_samples)

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        X     = (X - self.x_mean) / self.x_std
        pred  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            pred[i] = self.y_mean + nns[i](X).squeeze() * self.y_std
        return pred

    def report(self):
        print(self.nn)
        print('Dropout rate: %.2f' % self.dropout_rate)
