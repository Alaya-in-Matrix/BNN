import torch
import torch.nn as nn
import numpy as np
import toml
from BNN_Dropout import BNN_Dropout
from sklearn.model_selection import train_test_split

dataset_name = 'concrete'
dataset_dir  = 'UCI_Datasets/' + dataset_name + "/data/"
data         = np.loadtxt(dataset_dir + "data.txt")
feature_id   = np.loadtxt(dataset_dir + "index_features.txt", dtype=int)
target_id    = np.loadtxt(dataset_dir + "index_target.txt", dtype = int)
xs           = data[:,feature_id]
ys           = data[:,target_id][:,None]

split_id = 19
train_id = np.loadtxt(dataset_dir + "index_train_{}".format(split_id) + ".txt", dtype = int)
test_id  = np.loadtxt(dataset_dir + "index_test_{}".format(split_id) + ".txt", dtype = int)

train_x = xs[train_id]
train_y = ys[train_id]
test_x  = xs[test_id]
test_y  = ys[test_id]
dim     = train_x.shape[1]

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.1)

train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_y)
valid_x = torch.FloatTensor(valid_x)
valid_y = torch.FloatTensor(valid_y)
test_x  = torch.FloatTensor(test_x)
test_y  = torch.FloatTensor(test_y)

conf                 = dict()
conf['dropout_rate'] = 0.03
conf['lr']           = 1e-3
conf['tau']          = 0.075
conf['batch_size']   = 32
conf['num_epochs']   = 4000
conf['num_layers']   = 1
conf['num_hidden']   = 50
conf['print_every']  = 100

model = BNN_Dropout(dim, act = nn.ReLU(), conf = conf)
model.train(train_x, train_y)

rmse_training, nll_training = model.validate(train_x, train_y)
print('Training: rmse = %g, nll = %g' % (rmse_training, nll_training))

rmse_validation, nll_validation = model.validate(valid_x, valid_y)
print('Validation: rmse = %g, nll = %g' % (rmse_validation, nll_validation))

rmse_testing, nll_testing = model.validate(test_x, test_y)
print('Testing: rmse = %g, nll = %g' % (rmse_testing, nll_testing))

print(conf)

fid = open('result.po', 'w')
fid.write('%g\n' % (rmse_validation + nll_validation))
fid.close()
