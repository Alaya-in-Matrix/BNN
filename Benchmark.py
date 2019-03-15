import torch
import torch.nn as nn

class RandomNN(nn.Module):
    """
    Fully-connected neural network with randomly initialized weight
    """
    def __init__(self, dim, num_layer, num_hidden, act):
        super(RandomNN, self).__init__()
        self.dim        = dim
        self.num_layer  = num_layer
        self.num_hidden = num_hidden
        self.nn         = self.mlp(dim, num_layer, num_hidden, [act for i in range(num_layer)])

    def rand_normal_weight(self, weight_variance = 1):
        weight_sampler = torch.distributions.Normal(0, weight_variance)
        for layer in self.nn:
            if type(layer) == nn.Linear:
                layer.weight.data = weight_sampler.sample(layer.weight.shape)
                layer.bias.data   = weight_sampler.sample(layer.bias.shape)

    def rand_xavier_weight(self):
        for layer in self.nn:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def rand_kaiming_weight(self):
        for layer in self.nn:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

    def rand_orthogonal_weight(self):
        for layer in self.nn:
            if type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight)

    def mlp(self, dim, num_layer, num_hidden, act):
        layers  = []
        pre_dim = dim
        for i in range(num_layer):
            layers.append(nn.Linear(pre_dim, num_hidden, bias=True))
            layers.append(act[i])
            pre_dim = num_hidden
        layers.append(nn.Linear(pre_dim, 1, bias = True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
