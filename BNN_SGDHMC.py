import torch
import torch.nn as nn
import numpy as np
from pybnn.bohamiann   import Bohamiann
from pybnn.util.layers import AppendLayer
from util import ScaleLayer

class NN(nn.Module):
    def __init__(self, dim, act, num_hidden, num_layers):
        super(NN, self).__init__()
        self.dim        = dim
        self.act        = act
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.nn         = self.mlp()
        self.log_std    = AppendLayer(noise=1e-3)
        for l in self.nn:
            if type(l) == nn.Linear:
                nn.init.xavier_uniform_(l.weight)
                nn.init.zeros_(l.bias)
    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(nn.Linear(pre_dim, self.num_hidden, bias=True))
            layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
            layers.append(self.act)
            pre_dim = self.num_hidden
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.log_std(self.nn(x))


class BNN_SGDHMC:
    def __init__(self, dim, act = nn.Tanh(), conf = dict()):
        self.dim         = dim
        self.act         = act
        self.num_hidden  = conf.get('num_hidden',   50)
        self.num_layers  = conf.get('num_layers',   3)
        self.lr          = conf.get('lr',           1e-2)
        self.batch_size  = conf.get('batch_size',   128)
        self.num_burnin  = conf.get('num_burnin',   2000)
        self.keep_every  = conf.get('keep_every',   50)
        self.max_sample  = conf.get('max_sample',   50)
        self.print_every = conf.get('print_every',  100)
        def get_default_network(input_dimensionality: int) -> torch.nn.Module:
            return NN(dim=input_dimensionality, act = self.act, num_hidden = self.num_hidden, num_layers = self.num_layers)
        self.model = Bohamiann(get_network = get_default_network, print_every_n_steps = self.print_every)

    def train(self, X, y):
        """
        X: n_train * dim tensor
        y: n_train tensor
        """
        num_train = X.shape[0]
        y         = y.reshape(num_train, 1)
        num_steps = self.num_burnin + (1 + self.max_sample) * self.keep_every
        self.model.train(
                X.clone().detach().numpy(),
                y.clone().detach().numpy(),
                num_steps         = num_steps,
                keep_every        = self.keep_every,
                num_burn_in_steps = self.num_burnin,
                batch_size        = self.batch_size,
                lr                = self.lr,
                verbose = True)
        self.num_samples = len(self.model.sampled_weights)

    def sample(self, n_samples = 20):
        return np.random.permutation(self.num_samples)[:n_samples]

    def predict_mv(self, x):
        m, v = self.model.predict(x.numpy())
        return torch.FloatTensor(m), torch.FloatTensor(v)

    def predict_single(self, x, i):
        pred = torch.FloatTensor(self.model.predict_single(x.numpy(), i))
        return pred[:, 0]

    def validate(self, test_x, test_y):
        test_y   = test_y.squeeze()
        num_test = test_x.shape[0]
        m, v     = self.predict_mv(test_x)
        m        = m.squeeze()
        v        = v.squeeze()
        rmse     = torch.sqrt(torch.mean((test_y - m)**2)).item()
        nll      = 0
        for i in range(test_y.shape[0]):
            nll += -1 * torch.distributions.Normal(m[i], torch.sqrt(v[i])).log_prob(test_y[i])
        nll /= test_y.shape[0]
        return rmse, nll
