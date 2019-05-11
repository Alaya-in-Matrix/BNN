import pickle
import numpy as np
import torch
import sys

file_name = sys.argv[1]
stat      = pickle.load(open(file_name, 'rb'))

for dataset in stat:
    rmse  = torch.tensor([rec[0] for rec in stat[dataset] if not np.isnan(rec[0])])
    nll_g = torch.tensor([rec[1] for rec in stat[dataset] if not np.isnan(rec[1])])
    nll   = torch.tensor([rec[2] for rec in stat[dataset] if not np.isnan(rec[2])])
    print('%-30s (%8.4f %8.4f) (%8.4f %8.4f) (%8.4f %8.4f)' % (dataset, rmse.mean(), rmse.std(), nll_g.mean(), nll_g.std(), nll.mean(), nll_g.std()))
