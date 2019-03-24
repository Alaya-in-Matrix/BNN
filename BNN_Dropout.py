import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

class NN_Dropout(nn.Module):

    def __init__(self, dim, act, num_hidden, num_layers, dropout_rate, dropout_input = False):
        super(NN_Dropout, self).__init__()
        self.dim           = dim
        self.act           = act
        self.num_hidden    = num_hidden
        self.num_layers    = num_layers
        self.dropout_rate  = dropout_rate
        self.dropout_input = dropout_input
        self.nn            = self.mlp()
        for l in self.nn:
            if type(l) == nn.Linear:
                nn.init.xavier_uniform_(l.weight)
                nn.init.zeros_(l.bias)
    
    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(nn.Linear(pre_dim, self.num_hidden, bias=True))
            layers.append(self.act)
            pre_dim = self.num_hidden
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        bs = torch.distributions.Bernoulli(1 - self.dropout_rate)
        if self.dim > 1:
            x = F.dropout(x, p = self.dropout_rate, training = self.training) * (1 - self.dropout_rate)
        for l in self.nn:
            x = l(x)
            if type(l) != nn.Linear:
                x = F.dropout(x, p = self.dropout_rate, training = self.training) * (1 - self.dropout_rate)
        return x

class BNN_Dropout:
    def __init__(self, dim, act = nn.ReLU(), conf = dict()):
        self.dim          = dim
        self.act          = act
        self.num_hidden   = conf.get('num_hidden',   50)
        self.num_layers   = conf.get('num_layers',   3)
        self.num_epochs   = conf.get('num_epochs',   40)
        self.dropout_rate = conf.get('dropout_rate', 0.05)
        self.lr           = conf.get('lr',           1e-3)
        self.batch_size   = conf.get('batch_size',   128)
        self.tau          = conf.get('tau',          0.01)
        self.lscale       = conf.get('lscale',       1e-2)
        self.print_every  = conf.get('print_every',  100)
        self.nn           = NN_Dropout(dim, self.act, self.num_hidden, self.num_layers, self.dropout_rate)
        if torch.cuda.is_available():
            self.nn = self.nn.cuda()

    # TODO: logging
    # TODO: normalize input
    def train(self, X, y):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        self.x_mean  = X.mean(dim = 0)
        self.x_std   = X.std(dim = 0)
        self.y_mean  = y.mean()
        self.y_std   = y.std()
        self.train_x = (X - self.x_mean) / self.x_std
        self.train_y = (y - self.y_mean) / self.y_std
        num_train    = self.train_x.shape[0]
        self.l2_reg  = self.lscale**2 * (1 - self.dropout_rate) / (2. * num_train * self.tau)
        criterion    = nn.MSELoss()
        opt          = torch.optim.Adam(self.nn.parameters(), lr = self.lr, weight_decay = self.l2_reg)
        dataset      = TensorDataset(self.train_x, self.train_y)
        loader       = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        scheduler    = torch.optim.lr_scheduler.StepLR(opt, step_size = int(self.num_epochs / 4), gamma = 0.1)
        for epoch in range(self.num_epochs):
            scheduler.step()
            for bx, by in loader:
                def closure():
                    opt.zero_grad()
                    pred = self.nn(bx)
                    loss = criterion(pred, by)
                    loss.backward(retain_graph = True)
                    return loss
                opt.step(closure)
            if (epoch + 1) % self.print_every == 0:
                true_loss = criterion(self.nn(self.train_x), self.train_y)
                print("After %d epochs, loss is %g" % (epoch + 1, true_loss))
        self.nn = self.nn.cpu()
    
    def predict(self, x):
        self.nn.eval()
        pred = self.nn((x - self.x_mean) / self.x_std)
        return pred * self.y_std + self.y_mean

    def predict_mv(self, x, n_samples = 20):
        nns   = [self.sample() for i in range(n_samples)]
        preds = torch.zeros(x.shape[0], n_samples)
        for i in range(n_samples):
            preds[:, i] = nns[i]((x - self.x_mean) / self.x_std).reshape((x.shape[0], ))
        preds = preds * self.y_std + self.y_mean
        return preds.mean(dim = 1), preds.var(dim = 1) + 1 / self.tau

    def validate(self, x_test, y_test, n_samples = 1000):
        num_test = x_test.shape[0]
        y_test   = y_test.squeeze()
        preds    = torch.zeros(n_samples, num_test)
        for i in range(n_samples):
            nn          = self.sample()
            preds[i, :] = nn((x_test - self.x_mean) / self.x_std).squeeze()
        preds = preds * self.y_std + self.y_mean
        rmse  = torch.sqrt(torch.mean((y_test - preds.mean(dim = 0))**2)).item()
        ll    = torch.logsumexp(-0.5 * self.tau * (y_test - preds)**2, dim = 0) - np.log(n_samples) - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau)
        nll   = -1 * ll.mean()
        rmse_ = torch.sqrt(torch.mean((y_test - preds)**2)).item()
        return rmse, nll
        
    def sample(self):
        net = deepcopy(self.nn.nn)
        bs  = torch.distributions.Bernoulli(1 - self.dropout_rate)
        for layer in net:
            if isinstance(layer, nn.Linear):
                if layer.weight.shape[1] > 1:
                    vec                = bs.sample((layer.weight.shape[1], ))
                    layer.weight.data *= vec
        return net
