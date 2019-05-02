import os,sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OMP_NUM_THREADS']      = '1'
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GalConcreteDropoutImp import GalConcreteDropoutImp
import matplotlib.pyplot as plt
import pickle

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
       train_x = None
       train_y = None
       test_x  = None
       test_y = None
       tau = None
       n_hidden = None
       n_epochs = None
   else:
       train_id = np.loadtxt('../UCI_Datasets/' + dataset + "/data/index_train_" + str(split_id) + ".txt", dtype = np.int64)
       test_id  = np.loadtxt('../UCI_Datasets/' + dataset + "/data/index_test_" + str(split_id) + ".txt", dtype = np.int64)
       tau      = np.loadtxt('../UCI_Datasets/' + dataset + "/results/test_tau_100_xepochs_1_hidden_layers.txt")[split_id]
       n_hidden = np.loadtxt('../UCI_Datasets/' + dataset + "/data/n_hidden.txt", dtype = np.int64)
       n_epochs = np.loadtxt('../UCI_Datasets/' + dataset + "/data/n_epochs.txt", dtype = np.int64)

       train_x = xs[train_id]
       train_y = ys[train_id]
       test_x = xs[test_id]
       test_y = ys[test_id]
   return train_x, train_y, test_x, test_y,tau,n_hidden, n_splits, n_epochs


def uci(dataset, split_id):
   train_x, train_y, test_x, test_y,tau, n_hiddens, n_splits, n_epochs = get_data(dataset, split_id)
   if(split_id >= n_splits):
       print("Invalid split_id")
       return np.nan, np.nan, np.nan, np.nan
   print('Dataset %s, split: %d, n_hiddens: %d, prec: %g' % (dataset, split_id, n_hiddens, tau))
   conf = dict()
   conf['num_epochs']  = 100*n_epochs  # XXX: 10x, not 100x
   conf['batch_size']  = 128           # XXX: 32, not 128
   conf['print_every'] = 100
   conf['lr']          = 1e-3
   conf['wr']          = 1e-8  / train_x.shape[0] # 1e-4**2, as in Gal's pytorch implementation
   conf['dr']          = 10.   / train_x.shape[0]
   model = GalConcreteDropoutImp(train_x.shape[1], num_hiddens = [n_hiddens], conf = conf)
   model.train(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
   model.report()
   rmse, nll_gaussian,nll = model.validate(torch.FloatTensor(test_x), torch.FloatTensor(test_y), num_samples=1000)
   print('RMSE = %g, NLL_gaussian = %6.3f, NLL = %6.3f' % (rmse, nll_gaussian, nll), flush = True)
   return rmse, nll_gaussian, nll

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
    with Pool(num_thread) as p:
        stat[d] = p.map(f, list(range(20)))
    f = open("./results/stat_CDropout.pkl","wb")
    pickle.dump(stat,f)
    f.close()
