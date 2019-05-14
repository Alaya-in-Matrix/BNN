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
        self.num_epochs   = conf.get('num_epochs',   400)
        self.batch_size   = conf.get('batch_size',   32)
        self.print_every  = conf.get('print_every',  100)
        self.dropout_rate = conf.get('dropout_rate', 0.05)
        self.min_noise    = conf.get('min_noise',    0.) # a minimum noise level
        self.l2_reg       = conf.get('l2_reg', 1e-6)
        self.lr           = conf.get('lr',     1e-2)
        self.nn           = NN_Dropout(dim, self.act, self.num_hiddens, self.dropout_rate)
        self.noise_level  = 1.

    def train(self, X, y):
        assert(X.dim() == 2)
        assert(y.dim() == 1)
        num_train = X.shape[0]
        dict_decay    = {'params':[], 'weight_decay': self.l2_reg}
        dict_no_decay = {'params':[], 'weight_decay': 0.}
        for name, param in self.nn.named_parameters():
            if "bias" in name:
                dict_no_decay['params'].append(param)
            else:
                dict_decay['params'].append(param)
        opt     = torch.optim.Adam([dict_decay, dict_no_decay], lr = self.lr)
        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        crit    = nn.MSELoss(reduction = 'sum')
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            for bx, by in loader:
                opt.zero_grad()
                pred_y = self.nn(bx).squeeze()
                loss   = crit(pred_y, by)
                loss.backward()
                opt.step()
                epoch_loss += loss
            self.noise_level = max(self.min_noise, np.sqrt(epoch_loss.detach().numpy() / num_train)) # estimation of noise
            if (epoch + 1) % self.print_every == 0:
                print("[Epoch %5d, loss = %g, noise_level = %g]" % (epoch + 1, epoch_loss / num_train, self.noise_level))

    def sample(self, num_samples = 1):
        nns = []
        bs  = torch.distributions.Bernoulli(1 - self.dropout_rate)
        for i in range(num_samples):
            net = deepcopy(self.nn.nn)
            for layer in net:
                if isinstance(layer, nn.Linear) and layer.weight.shape[1] > 1:
                    layer.weight.data *= F.dropout(torch.ones(layer.weight.shape[1]), p = self.dropout_rate) * (1 - self.dropout_rate)
            nns.append(net)
        return nns

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        pred  = torch.zeros(len(nns), num_x)
        prec  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            pred[i] = nns[i](X).squeeze()
        return pred

    def report(self):
        print(self.nn)
        print('Dropout rate: %g' % self.dropout_rate)
        print('Noise level : %g' % self.noise_level)
