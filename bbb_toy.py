import torch
import torch.nn as nn
import pyro
import pyro.optim
import numpy as np
from   torch.distributions import constraints
from   BNN_BBB import BNN_BBB
import matplotlib.pyplot as plt

# Toy dataset
noise_level = 0.01
num_data    = 1000
xs          = torch.linspace(-3, 3, num_data).reshape(num_data, 1)
ys          = torch.FloatTensor(np.sinc(xs.numpy()))
ys          = ys + noise_level * torch.randn_like(ys)

conf                = dict()
conf['num_epochs']  = 2000
conf['batch_size']  = 10
conf['num_layers']  = 3
conf['print_every'] = 50
conf['lr']          = 1e-2
conf['n_samples']   = 1
model               = BNN_BBB(dim = 1, conf = conf)

num_train = 50
train_id  = torch.randperm(num_data)[:num_train]
train_x   = xs[train_id]
train_y   = ys[train_id]
model.train(train_x, train_y)

num_plot = 10
plt.plot(xs.numpy(), np.sinc(xs.numpy()), label = 'True function')
plt.plot(train_x.numpy(), train_y.numpy(), 'k+', label = 'Data')
for i in range(num_plot):
    py = model.nn((xs - model.x_mean) / model.x_std) * model.y_std + model.y_mean
    plt.plot(xs.numpy(), py.squeeze().detach().numpy(), 'g', alpha = 0.1)
plt.legend()
plt.show()
