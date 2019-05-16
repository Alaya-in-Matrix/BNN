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
from util import normalize

torch.set_num_threads(1)

if(len(sys.argv) > 1):
    num_thread = int(sys.argv[1])
else:
    num_thread = 1
print("Num thread: %d" % num_thread)

def get_data(dataset, split_id):
   ds_prefix = '../UCI_Datasets/'
   data = np.loadtxt('../UCI_Datasets/' + dataset + "/data/data.txt")
   xs = data[:,:-1]
   ys = data[:,-1]
   n_splits = np.loadtxt('../UCI_Datasets/' + dataset + "/data/n_splits.txt", dtype = np.int64)
   if split_id >= n_splits:
       train_x  = None
       train_y  = None
       test_x   = None
       test_y   = None
       tau      = None
       n_hidden = None
       n_epochs = None
   else:
       train_id = np.loadtxt('../UCI_Datasets/' + dataset + "/data/index_train_" + str(split_id) + ".txt", dtype = np.int64)
       test_id  = np.loadtxt('../UCI_Datasets/' + dataset + "/data/index_test_" + str(split_id) + ".txt", dtype = np.int64)
       tau      = np.loadtxt('../UCI_Datasets/' + dataset + "/results/test_tau_100_xepochs_1_hidden_layers.txt")[split_id]
       n_hidden = np.loadtxt('../UCI_Datasets/' + dataset + "/data/n_hidden.txt", dtype = np.int64)
       n_epochs = np.loadtxt('../UCI_Datasets/' + dataset + "/data/n_epochs.txt", dtype = np.int64)
       train_x  = torch.FloatTensor(xs[train_id])
       train_y  = torch.FloatTensor(ys[train_id])
       test_x   = torch.FloatTensor(xs[test_id])
       test_y   = torch.FloatTensor(ys[test_id])
   return train_x, train_y, test_x, test_y,tau,n_hidden, n_splits, n_epochs


def uci(dataset, split_id):
    train_x, train_y, test_x, test_y,tau, n_hiddens, n_splits, n_epochs = get_data(dataset, split_id)
    if(split_id >= n_splits):
        print("Invalid split_id")
        return np.nan, np.nan, np.nan, np.nan
    print('Dataset %s, split: %d, n_hiddens: %d, prec: %g' % (dataset, split_id, n_hiddens, tau))
    xm, xs, ym, ys = normalize(train_x, train_y)
    train_x = (train_x - xm) / xs
    test_x  = (test_x  - xm) / xs
    train_y = (train_y - ym) / ys

    conf                = dict()
    conf['num_epochs']  = 100*n_epochs # XXX: 10x, not 100x
    conf['batch_size']  = 128          # XXX: 32, not 128

    conf['steps_burnin'] = 2500
    conf['steps']        = 2500
    conf['keep_every']   = 50

    log10_lr_weight = -2.38928115427777321
    log10_lr_noise  = -1
    log10_lr_lambda = -5
    log10_alpha_w   = 0.346474921454162921
    log10_beta_w    = 3
    log10_alpha_n   = 3
    log10_beta_n    = 1.00109289876555185

    conf['lr_weight'] = 10**log10_lr_weight
    conf['lr_noise']  = 10**log10_lr_noise 
    conf['lr_lambda'] = 10**log10_lr_lambda
    conf['alpha_w']   = 10**log10_alpha_w  
    conf['beta_w']    = 10**log10_beta_w   
    conf['alpha_n']   = 10**log10_alpha_n  
    conf['beta_n']    = 10**log10_beta_n   


    for k in conf:
        print('%-20s:%g' %(k,conf[k]))

    model = BNN_SGDMC(train_x.shape[1], nn.Tanh(), num_hiddens = [n_hiddens, n_hiddens], conf = conf)
    model.train(train_x, train_y)
    model.report()
    with torch.no_grad():
        preds = model.sample_predict(model.nns,test_x) * ys + ym
    py = preds.mean(dim = 0).squeeze()
    pv = preds.var(dim = 0).squeeze() + ys**2 / model.log_precs.exp().squeeze().detach()
    ps = pv.sqrt()
    print(py.shape)
    print(ps.shape)
    print(test_y.shape)

    err = (py - test_y).squeeze()
    mse = torch.mean(err**2)
    rmse = mse.sqrt()
    nll = -1 * torch.distributions.Normal(py,ps).log_prob(test_y.squeeze()).mean()

    #rmse, nll = model.validate(test_x, test_y, num_samples=50)
    smse = rmse**2 / torch.mean((test_y - train_y.mean())**2)
    print('RMSE = %g, SMSE = %g, NLL = %6.3f' % (rmse, smse, nll), flush = True)
    print('Model  precision: %g' % (model.log_precs.exp()))
    print('Weight precision: %g' % (model.log_lambda.exp()))
    return rmse, nll

ds = [
  'bostonHousing'
, 'concrete'
, 'energy'
, 'kin8nm'
, 'naval-propulsion-plant'
, 'power-plant'
, 'protein-tertiary-structure'
, 'wine-quality-red'
, 'yacht'

]

stat = dict()
from multiprocessing import Pool
for d in ds:
    def f(split_id):
        return uci(d, split_id)
    stat[d] = [f(split_id) for split_id in range(20)]
    f = open("./results/stat_SGDMC.pkl","wb")
    pickle.dump(stat,f)
    f.close()
