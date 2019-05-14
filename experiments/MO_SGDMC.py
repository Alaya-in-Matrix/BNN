import os,sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OMP_NUM_THREADS']      = '1'
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from BNN_SGDMC import BNN_SGDMC

dir     = './MultiTask/OpAmp/'
train_x = torch.FloatTensor(np.loadtxt(dir + '/train_x'))
train_y = torch.FloatTensor(np.loadtxt(dir + '/train_y'))
test_x  = torch.FloatTensor(np.loadtxt(dir + '/test_x'))
test_y  = torch.FloatTensor(np.loadtxt(dir + '/test_y'))

print("Number of training data: %d" % train_x.shape[0])
print("Number of testing data: %d" % test_x.shape[0])
print("Dimension: %d" % train_x.shape[1])
print("Output: %d" % train_y.shape[1])

x_mean = train_x.mean(dim = 0)
x_std  = train_x.std(dim = 0)
y_mean = train_y.mean(dim = 0)
y_std  = train_y.std(dim = 0)

train_x = (train_x - x_mean) / x_std
train_y = (train_y - y_mean) / y_std
test_x  = (test_x - x_mean)  / x_std

conf                 = {}
conf['noise_level']  = 0.01
conf['steps_burnin'] = 2500
conf['steps']        = 2500
conf['keep_every']   = 50
conf['lr_weight']    = 1e-2
conf['lr_noise']     = 3e-1

model = BNN_SGDMC(dim = train_x.shape[1], nout = train_y.shape[1], act = nn.Tanh(), num_hiddens = [50, 50], conf = conf)
model.train(train_x, train_y)
model.report()

with torch.no_grad():
    pred = model.sample_predict(model.nns, test_x) * y_std + y_mean

py = pred.mean(dim = 0)
ps = pred.std(dim = 0)

np.savetxt('py', py.numpy())
np.savetxt('ps', py.numpy())

err  = py - test_y
mse  = torch.mean(err**2, dim = 0)
rmse = mse.sqrt()
smse = mse / torch.mean((test_y - y_mean)**2, dim = 0)


ll     = torch.distributions.Normal(py, ps).log_prob(test_y).mean(dim = 0)
ll_ref = torch.distributions.Normal(y_mean, y_std).log_prob(test_y).mean(dim = 0)
nll    = ll_ref - ll # normalized log likelihood
print(smse)
print(nll)
