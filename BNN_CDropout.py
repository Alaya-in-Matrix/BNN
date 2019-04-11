from   torch.distributions import constraints
from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import ScaleLayer, NN
from   BNN                 import BNN
from   BNN_BBB             import BNN_BBB
from   torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import sys, os

class RelaxedBernoulliLinear(nn.Module):
    def __init__(self, in_features, out_features, prob = torch.tensor(0.5), temp = torch.tensor(0.1)):
        """
        in_features: integer, dimension of input
        out_features: integer, dimension of output
        prob: scalar tensor, bernoulli probability
        temp: scalar tensor, temperature of the RelaxedBernoulli distribution
        """
        super(RelaxedBernoulliLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.t            = temp
        self.p            = Parameter(prob)
        self.linear       = nn.Linear(in_features, out_features, bias = True)
        self.dist         = RelaxedBernoulli(temperature = self.t, probs = self.p)

    def forward(self, input):
        input *= self.dist.sample((self.in_features, ))
        return self.linear(input * self.dist.sample((self.in_features, )))

    def sample_linear(self):
        layer             = nn.Linear(self.in_features, self.out_features, bias = True)
        layer.weight.data = self.linear.weight.data.clone() * self.dist.sample((self.in_features, ))
        layer.bias.data   = self.linear.bias.data.clone()
        return layer

    def extra_repr(self):
        return 'in_features=%d, out_features=%d, temp = %4.2f, dropout_rate = %4.2f' % (self.in_features, self.out_features, self.t, self.p)


class BayesianNN(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], scale = True, probs = torch.tensor([0.5, 0.5]), temp = torch.tensor(0.1)):
        self.probs = probs
        self.temp  = temp
        self.nn    = self.mlp()

    def sample(self):
        layers = []
        for layer in self.nn:
            if isinstance(layer, RelaxedBernoulliLinear):
                layers.append(layer.sample_linear())
            else:
                layers.append(layer)
        return nn.Sequential(*layers)

    def mlp(self):
        layers  = []
        # pre_dim = self.dim
        # for i in range(self.num_layers):
        #     layers.append(RelaxedBernoulliLinear(pre_dim, self.num_hiddens[i], prob = self.probs[i], temp = self.temp))
        #     if self.scale:
        #         layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
        #     layers.append(self.act)
        #     pre_dim = self.num_hiddens[i]
        # layers.append(RelaxedBernoulliLinear(pre_dim, 1, prob = self.probs[-1], temp = self.temp))
        # if self.scale:
        #     layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
        return nn.Sequential(*layers)
