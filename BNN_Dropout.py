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
    def __init__(self, dim, act, conf):
        self.dim          = dim
        self.act          = act
        self.num_hidden   = conf.get('num_hidden', 50)
        self.num_layers   = conf.get('num_layers', 3)
        self.dropout_rate = conf.get('dropout_rate', 0.05)
        self.lr           = conf.get('lr', 1e-3)
        self.batch_size   = conf.get('batch_size', 128)
        self.num_epochs   = conf.get('num_epochs', 40)
        self.tau          = conf.get('tau', 1)
        self.lscale       = conf.get('lscale', 1e-2)
        self.print_every  = conf.get('print_every', 100)
        self.nn           = NN_Dropout(dim, self.act, self.num_hidden, self.num_layers, self.dropout_rate)

    # TODO: logging
    # TODO: normalize input
    def train(self, X, y):
        self.x_mean     = X.mean(dim = 0)
        self.x_std      = X.std(dim = 0)
        self.y_mean     = y.mean()
        self.y_std      = y.std()
        self.train_x    = (X - self.x_mean) / self.x_std
        self.train_y    = (y - self.y_mean) / self.y_std
        num_train       = self.train_x.shape[0]
        self.l2_reg     = self.lscale**2 * (1 - self.dropout_rate) / (2. * num_train * self.tau)
        criterion       = nn.MSELoss()
        opt             = torch.optim.Adam(self.nn.parameters(), lr = self.lr, weight_decay = self.l2_reg)
        dataset         = TensorDataset(self.train_x, self.train_y)
        loader          = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        scheduler       = torch.optim.lr_scheduler.StepLR(opt, step_size = int(self.num_epochs / 4), gamma = 0.1)
        # scheduler       = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, min_lr = 1e-6)
        # scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = int(self.num_epochs / 4), eta_min = 1e-8)
        self.rec_losses = [np.inf for i in range(self.num_epochs)]
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
            true_loss     = criterion(self.nn(self.train_x), self.train_y)
            self.rec_losses[epoch] = true_loss.item()
            if (epoch + 1) % self.print_every == 0:
                print("After %d epochs, loss is %g" % (epoch + 1, true_loss))
    
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
        return preds.mean(dim = 1), preds.var(dim = 1)

    def validate(self, x_test, y_test, n_samples = 20):
        m, v = self.predict_mv(x_test)
        m    = m.reshape(y_test.shape)
        v    = v.reshape(y_test.shape)
        s    = v.sqrt()
        mse  = torch.nn.MSELoss()(m.reshape(y_test.shape), y_test)
        rmse = torch.sqrt(mse).item()
        nll  = -1 * np.mean([torch.distributions.Normal(m[i], s[i]).log_prob(y_test[i]).item() for i in range(x_test.shape[0])])
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
