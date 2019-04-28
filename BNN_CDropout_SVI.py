import torch
import pyro
import pyro.optim
import numpy    as np
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from pyro.distributions.relaxed_straight_through import RelaxedBernoulli
from torch.distributions.transforms              import AffineTransform
from torch.distributions.utils import clamp_probs, probs_to_logits, logits_to_probs
from pyro.distributions import TransformedDistribution
from BNN import BNN

# class StableRelaxedBernoulli(RelaxedBernoulli):
#     """
#     Numerical stable relaxed Bernoulli distribution
#     """
#     def rsample(self, sample_shape = torch.Size()):
#         return clamp_probs(super(StableRelaxedBernoulli, self).rsample(sample_shape))

# class BNN_CDropout_SVI(BNN):

#     def __init__(self, dim, act = nn.ReLU(), num_hiddens = [50], conf = dict()):
#         """
#         Concrete dropout
#         Instead of performing concrete dropout, BNN_CDropout_SVI directly perform variational inference using pyro's SVI inference
#         """
#         super(BNN_CDropout_SVI, self).__init__()
#         self.dim          = dim
#         self.act          = act
#         self.num_hiddens  = num_hiddens
#         self.num_epochs   = conf.get('num_epochs',   400)
#         self.batch_size   = conf.get('batch_size',   32)
#         self.print_every  = conf.get('print_every',  50)
#         self.normalize    = conf.get('normalize',    True)
#         self.lr           = conf.get('lr',           1e-2)
#         self.weight_prior = conf.get('weight_prior', 1.)
#         self.temperature  = conf.get('temperature',  0.1)
#         self.dropout_rate = conf.get('dropout_rate', 0.5)
#         self.alpha        = conf.get('alpha', 0.1) # XXX: precison corespond to (possibly) normalized data, be careful
#         self.beta         = conf.get('beta',  0.01)
#         self.init_prec    = conf.get('init_prec', self.alpha / self.beta)

#     def model(self, X, y):
#         num_x       = X.shape[0]
#         y           = y.reshape(num_x)
#         wp          = self.weight_prior
#         para        = dict()
#         in_feature  = self.dim
#         precision   = pyro.sample("precision", pyro.distributions.Gamma(self.alpha, self.beta))
#         noise_level = 1 / precision.sqrt()
#         for i in range(len(self.num_hiddens)):
#             para['w{}'.format(i)] = pyro.sample("w{}".format(i), pyro.distributions.Normal(torch.tensor(0.), wp * torch.ones(self.num_hiddens[i], in_feature)))
#             para['b{}'.format(i)] = pyro.sample("b{}".format(i), pyro.distributions.Normal(torch.tensor(0.), wp * torch.ones(self.num_hiddens[i])))
#             in_feature            = self.num_hiddens[i]
#         para['wout'] = pyro.sample("wout", pyro.distributions.Normal(torch.tensor(0.), wp * torch.ones(1, in_feature)))
#         para['bout'] = pyro.sample("bout", pyro.distributions.Normal(torch.tensor(0.), wp * torch.ones(1)))
#         with pyro.plate("map", len(X), subsample_size = min(num_x, self.batch_size)) as ind:
#             out         = X[ind]
#             in_feature  = self.dim
#             noise_level = 1 / precision.sqrt()
#             for i in range(len(self.num_hiddens)):
#                 out = F.linear(input = out, weight = para['w{}'.format(i)], bias = para['b{}'.format(i)])
#                 out = self.act(out)
#                 in_feature = self.num_hiddens[i]
#             out = F.linear(input = out, weight = para['wout'], bias = para['bout']).squeeze()
#             pyro.sample("obs", pyro.distributions.Normal(out, noise_level), obs = y[ind])

#     def xavier(self, sample_shape = torch.Size()):
#         weight = torch.zeros(sample_shape)
#         nn.init.xavier_uniform_(weight)
#         return weight

#     def guide(self, X, y):
#         in_feature     = self.dim
#         precision_para = pyro.param("precision_para", torch.tensor(self.init_prec)) # XXX: SNR = 10, if y is normalized
#         precision      = pyro.sample("precision", pyro.distributions.Delta(v = precision_para))
#         for i in range(len(self.num_hiddens)):
#             p_logit = pyro.param("p_logit_{}".format(i),      probs_to_logits(torch.as_tensor(self.dropout_rate), is_binary = True))
#             bias    = pyro.param("bias_param_{}".format(i),   torch.zeros(self.num_hiddens[i]))
#             weight  = pyro.param("weight_param_{}".format(i), self.xavier((self.num_hiddens[i], in_feature)))

#             probs = 1. - torch.sigmoid(p_logit) * torch.ones(in_feature)
#             mask  = pyro.sample("mask{}".format(i), StableRelaxedBernoulli(temperature = torch.tensor(self.temperature), probs = probs))
#             pyro.sample("w{}".format(i), pyro.distributions.Delta(v = weight * mask))
#             pyro.sample("b{}".format(i), pyro.distributions.Delta(v = bias))
#             in_feature = self.num_hiddens[i]

#         p_logit = pyro.param("p_logit_out",      probs_to_logits(torch.as_tensor(self.dropout_rate), is_binary = True))
#         bias    = pyro.param("bias_param_out",   torch.zeros(1))
#         weight  = pyro.param("weight_param_out", self.xavier((1, in_feature)))

#         probs = 1. - torch.sigmoid(p_logit) * torch.ones(in_feature)
#         mask  = pyro.sample("mask_out", StableRelaxedBernoulli(temperature = torch.tensor(self.temperature), probs = probs))
#         pyro.sample("wout", pyro.distributions.Delta(v = weight * mask))
#         pyro.sample("bout", pyro.distributions.Delta(v = bias))

#     def train(self, X, y):
#         num_train         = X.shape[0]
#         y                 = y.reshape(num_train)
#         self.normalize_Xy(X, y, self.normalize)
#         optim             = pyro.optim.Adam({"lr":self.lr})
#         svi               = pyro.infer.SVI(self.model, self.guide, optim, loss = pyro.infer.Trace_ELBO())
#         pyro.clear_param_store()
#         self.rec  = []
#         num_iters = self.num_epochs * (1 + int(num_train / self.batch_size))
#         for i in range(num_iters):
#             loss = svi.step(self.X, self.y)
#             if (i+1) % self.print_every == 0 or i == 0:
#                 self.rec.append(loss / num_train)
#                 print("[Iteration %05d/%05d] loss: %-4.3f, precision = %4.3f" % (i + 1, num_iters, loss / num_train, self.sample_prec()), flush = True)
    
#     def sample_one(self):
#         layers = []
#         in_feature = self.dim
#         for i in range(len(self.num_hiddens)):
#             p_logit           = pyro.param("p_logit_{}".format(i))
#             bias              = pyro.param("bias_param_{}".format(i))
#             weight            = pyro.param("weight_param_{}".format(i))
#             probs             = 1. - torch.sigmoid(p_logit) * torch.ones(in_feature)
#             mask              = pyro.sample("mask{}".format(i), StableRelaxedBernoulli(temperature = torch.tensor(self.temperature), probs = probs))
#             weight            = pyro.distributions.Delta(v = weight * mask).sample()
#             bias              = pyro.distributions.Delta(v = bias).sample()
#             layer             = nn.Linear(in_feature, self.num_hiddens[i])
#             layer.weight.data = weight
#             layer.bias.data   = bias
#             in_feature        = self.num_hiddens[i]
#             layers.append(layer)
#             layers.append(self.act)

#         p_logit           = pyro.param("p_logit_out")
#         bias              = pyro.param("bias_param_out")
#         weight            = pyro.param("weight_param_out")
#         probs             = 1. - torch.sigmoid(p_logit) * torch.ones(in_feature)
#         mask              = pyro.sample("mask_out", StableRelaxedBernoulli(temperature = torch.tensor(self.temperature), probs = probs))
#         wout              = pyro.distributions.Delta(v = weight * mask).sample()
#         bout              = pyro.distributions.Delta(v = bias).sample()
#         layer             = nn.Linear(in_feature, 1)
#         layer.weight.data = wout
#         layer.bias.data   = bout
#         layers.append(layer)
#         return nn.Sequential(*layers)

#     def report(self):
#         for i in range(len(self.num_hiddens)):
#             p_logit = pyro.param("p_logit_{}".format(i))
#             print("Layer %2d, dropout_rate = %3.2f, temp = %4.3f" % (i, torch.sigmoid(p_logit), self.temperature))
#         p_logit = pyro.param("p_logit_out")
#         print("Out Layer, dropout_rate = %3.2f, temp = %4.3f" % (torch.sigmoid(p_logit), self.temperature))
#         print("Model precision: %4.3f" % self.sample_prec())

#     def sample_prec(self):
#         return pyro.param("precision_para").item() / self.y_std**2
    
#     def sample(self, num_samples = 1):
#         nns   = [self.sample_one()  for i in range(num_samples)]
#         precs = torch.tensor([self.sample_prec() for i in range(num_samples)])
#         return nns, precs

#     def sample_predict(self, nns, X):
#         num_x = X.shape[0]
#         X     = (X - self.x_mean) / self.x_std
#         pred  = torch.zeros(len(nns), num_x)
#         for i in range(len(nns)):
#             pred[i] = self.y_mean + nns[i](X).squeeze() * self.y_std
#         return pred
