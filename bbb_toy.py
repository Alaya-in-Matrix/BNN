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

conf_bbb                = dict()
conf_bbb['num_epochs']  = 4000
conf_bbb['batch_size']  = 128
conf_bbb['num_layers']  = 3
conf_bbb['print_every'] = 50
conf_bbb['n_samples']   = 1
conf_bbb['lr']          = 1e-2
conf_bbb['pi']          = 0.25
conf_bbb['s1']          = 10.
conf_bbb['s2']          = 0.1
conf_bbb['noise_level'] = noise_level
model                   = BNN_BBB(dim = 1, act = nn.Tanh(), conf = conf_bbb)

num_train = 100
train_id  = torch.randperm(num_data)[:num_train]
train_x   = xs[train_id]
train_y   = ys[train_id]
model.train(train_x, train_y)

num_plot = 50
plt.plot(xs.numpy(), np.sinc(xs.numpy()), label = 'True function')
plt.plot(train_x.numpy(), train_y.numpy(), 'k+', label = 'Data')
for i in range(num_plot):
    py = model.nn((xs - model.x_mean) / model.x_std)
    py = py * model.y_std + model.y_mean
    plt.plot(xs.numpy(), py.squeeze().detach().numpy(), 'g', alpha = 0.1)
plt.legend()
plt.show()
