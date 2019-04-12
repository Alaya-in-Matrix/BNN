from   torch.nn.parameter  import Parameter
from   torch.utils.data    import TensorDataset, DataLoader
from   util                import ScaleLayer
from   BNN                 import BNN
from   BNN_BBB             import MixturePrior
from   torch.distributions.relaxed_bernoulli        import RelaxedBernoulli
from   torch.distributions.utils                    import clamp_probs, logits_to_probs, probs_to_logits
from   torch.distributions.transformed_distribution import TransformedDistribution
from   torch.distributions.transforms               import AffineTransform
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import sys, os

class StableRelaxedBernoulli(RelaxedBernoulli):
    def rsample(self, sample_shape = torch.Size()):
        return clamp_probs(super(StableRelaxedBernoulli, self).rsample(sample_shape))

class RelaxedDropoutLinear(nn.Module):
    def __init__(self, in_features, out_features, temperature = torch.tensor(1.0), dropout_rate = torch.tensor(0.5)):
        super(RelaxedDropoutLinear, self).__init__()
        self.dropout_logit = nn.Parameter(torch.as_tensor(dropout_rate / (1 - dropout_rate)).log())
        self.weight        = nn.Parameter(torch.randn(out_features, in_features))
        self.bias          = nn.Parameter(torch.randn(out_features))
        self.temperature   = torch.as_tensor(temperature) # XXX: should temp also regarded as variational parameter?
        self.in_features   = in_features
        self.out_features  = out_features
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, input):
        probs            = 1 - torch.sigmoid(self.dropout_logit) * torch.ones(self.in_features)
        base_dist        = StableRelaxedBernoulli(temperature = self.temperature, probs = probs)
        self.dist        = TransformedDistribution(base_dist, AffineTransform(loc = torch.zeros(1), scale = self.weight))
        self.mask_weight = self.dist.rsample()
        self.log_prob    = self.dist.log_prob(self.mask_weight).sum()
        return F.linear(input, weight = self.mask_weight, bias = self.bias)
    
    def extra_repr(self):
        return 'in_features = %d, out_features = %d, temperature = %4.2f, dropout_rate = %4.2f' % (self.in_features, self.out_features, self.temperature, torch.sigmoid(self.dropout_logit))

    def sample_linear(self):
        probs             = 1 - torch.sigmoid(self.dropout_logit) * torch.ones(self.in_features)
        base_dist         = StableRelaxedBernoulli(temperature = self.temperature, probs = probs)
        self.dist         = TransformedDistribution(base_dist, AffineTransform(loc = torch.zeros(1), scale = self.weight))
        layer             = nn.Linear(self.in_features, self.out_features, bias = True)
        layer.weight.data = self.dist.sample().data.clone()
        layer.bias.data   = self.bias.data.clone()
        return layer


class BayesianNN(nn.Module):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], scale = False, dropout_rate = 0.5, temp = 0.1):
        super(BayesianNN, self).__init__()
        self.dim          = dim
        self.act          = act
        self.num_hiddens  = num_hiddens
        self.num_layers   = len(num_hiddens)
        self.scale        = scale
        self.dropout_rate = dropout_rate
        self.temp         = temp
        self.nn           = self.mlp()

    def forward(self, input):
        return self.nn(input)

    def sample(self):
        layers = []
        for layer in self.nn:
            if isinstance(layer, RelaxedDropoutLinear):
                layers.append(layer.sample_linear())
            else:
                layers.append(layer)
        return nn.Sequential(*layers)

    def mlp(self):
        layers  = []
        pre_dim = self.dim
        for i in range(self.num_layers):
            layers.append(RelaxedDropoutLinear(pre_dim, self.num_hiddens[i], temperature = self.temp, dropout_rate = self.dropout_rate))
            if self.scale:
                layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
            layers.append(self.act)
            pre_dim = self.num_hiddens[i]
        layers.append(RelaxedDropoutLinear(pre_dim, 1, temperature = self.temp, dropout_rate = self.dropout_rate))
        if self.scale:
            layers.append(ScaleLayer(1 / np.sqrt(1 + pre_dim)))
        return nn.Sequential(*layers)


class BNN_CDropout_BBB(BNN):
    def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
        """
        Concrete dropout
        Instead of performing concrete dropout, BNN_CDropout_BBB directly perform variational inference using Bayes-by-backprop
        """
        super(BNN_CDropout_BBB, self).__init__()
        self.dim         = dim
        self.act         = act
        self.num_hiddens = num_hiddens
        self.num_epochs  = conf.get('num_epochs',   400)
        self.batch_size  = conf.get('batch_size',   32)
        self.print_every = conf.get('print_every',  50)
        self.n_samples   = conf.get('n_samples',    1)
        self.lr          = conf.get('lr',           1e-2)
        self.pi          = conf.get('pi',           1.)
        self.s1          = conf.get('s1',           1.)
        self.s2          = conf.get('s2',           1.)
        self.noise_level = conf.get('noise_level',  0.1)
        self.temperature = conf.get('temperature',  0.1)
        self.normalize   = conf.get('normalize',    True)
        self.scale_layer = conf.get('scale_layer',  False)
        self.w_prior     = MixturePrior(factor = self.pi, s1 = self.s1, s2 = self.s2)
        self.nn          = BayesianNN(dim, self.act, self.num_hiddens, self.scale_layer, temp = self.temperature)
    
    def loss(self, X, y):
        num_x   = X.shape[0]
        X       = X.reshape(num_x, self.dim)
        y       = y.reshape(num_x)
        log_lik = torch.tensor(0.)
        log_qw  = torch.tensor(0.)
        log_pw  = torch.tensor(0.)
        for i in range(self.n_samples):
            pred     = self.nn(X).reshape(num_x).reshape(y.shape)
            log_lik += torch.distributions.Normal(pred, self.noise_level).log_prob(y).sum()
            for layer in self.nn.nn:
                if isinstance(layer, RelaxedDropoutLinear):
                    log_qw += layer.log_prob
                    log_pw += self.w_prior.log_prob(layer.mask_weight).sum()
                    log_pw += self.w_prior.log_prob(layer.bias).sum()
            kl_term = log_qw - log_pw
        return log_lik / self.n_samples, kl_term / self.n_samples

    def train(self, X, y):
        num_x = X.shape[0]
        X     = X.reshape(num_x, self.dim)
        y     = y.reshape(num_x)
        self.normalize_Xy(X, y, self.normalize)
        self.noise_level /= self.y_std
        dataset   = TensorDataset(self.X, self.y)
        loader    = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        opt       = torch.optim.Adam(self.nn.parameters(), lr = self.lr)
        # opt       = torch.optim.SGD(self.nn.parameters(), lr = self.lr, momentum = 0.1)
        num_batch = len(loader)
        for epoch in range(self.num_epochs):
            batch_cnt = 1
            for bx, by in loader:
                opt.zero_grad()
                log_lik             = torch.tensor(0.)
                kl_term             = torch.tensor(0.)
                _log_lik, _kl_term  = self.loss(bx, by)
                log_lik            += _log_lik
                kl_term            += _kl_term
                pi                  = 2**(num_batch - batch_cnt) / (2**(num_batch) - 1)
                loss                = (pi * kl_term - log_lik)
                loss.backward(retain_graph=True)
                opt.step()
                batch_cnt += 1
            if ((epoch + 1) % self.print_every == 0):
                log_lik, kl_term = self.loss(self.X, self.y)
                print("[Epoch %5d, loss = %.4g (KL = %.4g, -log_lik = %.4g)]" % (epoch + 1, kl_term - log_lik, kl_term, -1 * log_lik), flush = True)
        self.noise_level *= self.y_std

    def sample(self, num_samples = 1):
        nns = [self.nn.sample() for i in range(num_samples)]
        return nns, torch.ones(num_samples) / (self.noise_level**2)

    def sample_predict(self, nns, X):
        num_x = X.shape[0]
        X     = (X - self.x_mean) / self.x_std
        pred  = torch.zeros(len(nns), num_x)
        for i in range(len(nns)):
            pred[i] = self.y_mean + nns[i](X).squeeze() * self.y_std
        return pred
