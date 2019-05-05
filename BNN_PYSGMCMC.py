import torch
import torch.nn as nn
import torch.nn.functional as F
from BNN import BNN
from util import *
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.models.architectures import simple_tanh_network
from pysgmcmc.optimizers.sgld import SGLD


class NoisyNN(NN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], logvar = torch.log(torch.tensor(1e-3))):
        super(NoisyNN, self).__init__(dim, act, num_hiddens, nout = 1)
        self.logvar = nn.Parameter(logvar)
    
    def forward(self, input):
        out     = self.nn(input)
        logvars = torch.clamp(self.logvar, max = 20.) * out.new_ones(out.shape)
        return torch.cat((out, logvars), dim = out.dim() - 1)

class BNN_PYSGMCMC(nn.Module, BNN):
    """
    Wraper of the bayesian neural network provided by pysgmcmc, 
    with custom architecture(number of layers, acitvations)
    """
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        nn.Module.__init__(self)
        BNN.__init__(self)
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.lr           = conf.get('lr', 1e-2)
        self.steps_burnin = conf.get('steps_burnin', 3000)
        self.steps        = conf.get('steps', 10000)
        self.keep_every   = conf.get('keep_every', 100)
        self.batch_size   = conf.get('batch_size', 32)
        self.normalize    = conf.get('normalize',  True)
        self.bnn          = BayesianNeuralNetwork(
                network_architecture = lambda input_dimensionality : NoisyNN(input_dimensionality, self.act, self.num_hiddens), 
                optimizer            = SGLD,
                lr                   = self.lr,
                batch_size           = self.batch_size,
                burn_in_steps        = self.steps_burnin,
                num_steps            = self.steps_burnin + self.steps,
                keep_every           = self.keep_every)
    
    def train(self, X, y):
        self.normalize_Xy(X, y, self.normalize)
        _X = self.X.numpy()
        _y = self.y.numpy()
        self.bnn.train(_X, _y)

    def sample(self, num_samples):
        assert(num_samples <= len(self.bnn.sampled_weights))
        return [None for i in range(num_samples)]

    def sample_predict(self, nns, X):
        assert(len(nns) <= len(self.bnn.sampled_weights))
        X         = (X - self.x_mean) / self.x_std
        num_x     = X.shape[0]
        _, _, out = self.bnn.predict(X.numpy(), return_individual_predictions = True)
        pred = torch.zeros(len(nns), num_x)
        prec = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            rec     = torch.FloatTensor(out[i])
            pred[i] = rec[:, 0] * self.y_std + self.y_mean
            prec[i] = 1 / (torch.exp(rec[:, 1]) * self.y_std**2)
        return pred, prec


    def report(self):
        print(self.bnn.model)
        print("Number of samples: %d" % len(self.bnn.sampled_weights))
