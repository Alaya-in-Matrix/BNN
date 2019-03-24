import torch
import torch.nn as nn
import numpy as np
import toml
import matplotlib.pyplot as plt
from BNN_Dropout import BNN_Dropout
from BNN_SGDHMC  import BNN_SGDHMC
from sklearn.model_selection import train_test_split

dataset_name = 'concrete'
dataset_dir  = 'UCI_Datasets/' + dataset_name + "/data/"
data         = np.loadtxt(dataset_dir + "data.txt")
feature_id   = np.loadtxt(dataset_dir + "index_features.txt", dtype=int)
target_id    = np.loadtxt(dataset_dir + "index_target.txt", dtype = int)
xs           = data[:,feature_id]
ys           = data[:,target_id][:,None]

split_id = 12
train_id = np.loadtxt(dataset_dir + "index_train_{}".format(split_id) + ".txt", dtype = int)
test_id  = np.loadtxt(dataset_dir + "index_test_{}".format(split_id) + ".txt", dtype = int)

train_x = xs[train_id]
train_y = ys[train_id]
test_x  = xs[test_id]
test_y  = ys[test_id]
dim     = train_x.shape[1]

train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_y)
test_x  = torch.FloatTensor(test_x)
test_y  = torch.FloatTensor(test_y)

conf                 = dict()
conf['dropout_rate'] = 0.01
conf['lr']           = 0.1
conf['tau']          = 0.15
conf['lscale']       = 0.25
conf['batch_size']   = 128
conf['num_epochs']   = 1600
conf['num_layers']   = 3
conf['num_hidden']   = 50
conf['print_every']  = 50

model = BNN_Dropout(dim, act = nn.Tanh(), conf = conf)
model.train(train_x, train_y)
dr_rmse_training, dr_nll_training = model.validate(train_x, train_y)
dr_rmse_testing, dr_nll_testing   = model.validate(test_x, test_y)



# conf_hmc = dict()
# conf_hmc['lr'] = 5e-3
# conf_hmc['batch_size'] = 128
# conf_hmc['num_burnin'] = 5000
# conf_hmc['max_sample'] = 100
# conf_hmc['keep_every'] = 100
# hmc_model    = BNN_SGDHMC(dim, act = nn.Tanh(), conf = conf_hmc)
# hmc_model.train(train_x, train_y)
# hmc_rmse_training, hmc_nll_training = hmc_model.validate(train_x, train_y)
# hmc_rmse_testing, hmc_nll_testing   = hmc_model.validate(test_x, test_y)

print("Split ID: %d" % split_id)
print('Dropout Training: rmse = %g, nll = %g' % (dr_rmse_training, dr_nll_training))
print('Dropout Testing:  rmse = %g, nll = %g' % (dr_rmse_testing,  dr_nll_testing))
# print('SGDHMC  Training: rmse = %g, nll = %g' % (hmc_rmse_training, hmc_nll_training))
# print('SGDHMC  Testing:  rmse = %g, nll = %g' % (hmc_rmse_testing,  hmc_nll_testing))


# dr_py, dr_pv   = model.predict_mv(test_x)
# sgd_py, sgd_pv = hmc_model.predict_mv(test_x)
# test_y = test_y.squeeze()
# dr_py  = dr_py.squeeze()
# dr_pv  = dr_pv.squeeze()
# sgd_py = sgd_py.squeeze()
# sgd_pv = sgd_pv.squeeze()

# dr_normed = (test_y - dr_py)   / dr_pv.sqrt()
# sgd_normed = (test_y - sgd_py) / sgd_pv.sqrt()

# plt.hist(dr_normed.detach().numpy(), label  = 'Dropout', bins = 30)
# plt.hist(sgd_normed.detach().numpy(), label = 'HMC', alpha = 0.5, bins = 30)
# plt.legend()
# plt.show()
