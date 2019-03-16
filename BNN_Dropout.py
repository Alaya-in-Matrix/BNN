import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

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
            # layers.append(nn.Dropout(p = self.dropout_rate))
            pre_dim = self.num_hidden
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        return nn.Sequential(*layers)
    
    # Forward function used for training
    def forward(self, x):
        for l in self.nn:
            x = l(x)
            if type(l) != nn.Linear:
                x = F.dropout(x, p = self.dropout_rate, training = self.training)
        return x

class BNN_Dropout:
    def __init__(self, dim, act, conf):
        self.dim          = dim
        self.act          = act
        self.num_hidden   = conf.get('num_hidden', 50)
        self.num_layers   = conf.get('num_layers', 3)
        self.dropout_rate = conf.get('dropout_rate', 0.5)
        self.lr           = conf.get('lr', 1e-3)
        self.batch_size   = conf.get('batch_size', 32)
        self.num_epochs   = conf.get('num_epochs', 100)
        self.l2_reg       = conf.get('l2_reg', 1e-6)
        self.nn           = NN_Dropout(dim, self.act, self.num_hidden, self.num_layers, self.dropout_rate)

    # TODO: logging
    # TODO: normalize input
    def train(self, X, y):
        self.train_x    = X
        self.train_y    = y
        criterion       = nn.MSELoss()
        opt             = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        loss            = torch.tensor(float('inf'));
        dataset         = TensorDataset(self.train_x, self.train_y)
        loader          = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        self.rec_losses = []
        for epoch in range(self.num_epochs):
            for bx, by in loader:
                opt.zero_grad()
                pred     = self.nn(bx)
                mse_loss = criterion(pred, by)
                reg_loss = 0
                for name, param in self.nn.named_parameters():
                    if "bias" not in name:
                        reg_loss += self.lr * param.norm(2) ** 2
                loss = mse_loss + reg_loss
                loss.backward(retain_graph = True)
                opt.step()
            print("After %d epochs, loss is %g" % (epoch + 1, loss))
            self.rec_losses.append(loss.data)
    
    def predict(self, x):
        self.nn.eval()
        pred = self.nn(x)
        return pred

    def sample(self):
        pass
