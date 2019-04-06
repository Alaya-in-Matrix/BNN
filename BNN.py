import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BNN(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(X, y):
        """
        X: num_train * dim matrix
        y: num_train vector
        """
        pass

    @abstractmethod
    def sample(num_samples = 1):
        """
        Generate `num_sample` samples from the posterior, return a list of neural networks and posterior precisions
        """
        pass

    @abstractmethod
    def sample_predict(self, nns, input):
        pass

    def validate(self, X, y, num_samples):
        num_test     = X.shape[0]
        y            = y.reshape(num_test)
        post_samples = self.sample(num_samples)
        nn_samples   = [s[0] for s in post_samples]
        prec_samples = [s[1] for s in post_samples]
        preds        = self.sample_predict(nn_samples, X)
        noise_vars   = 1 / prec_samples
        py           = preds.mean(dim = 0)
        pv           = preds.var(dim = 0) + noise_vars
        rmse         = torch.sqrt(torch.mean((py - y)**2))
        nll_gaussian = -1 * torch.distributions.Normal(py, pv.sqrt()).log_prob(y).mean()
        normed       = (y - preds) / noise_vars.sqrt()
        lls          = torch.logsumexp(-0.5 * normed**2 - 0.5 * torch.log(2 * np.pi * noise_var.unsqueeze(1)), dim = 0) - np.log(num_samples)
        return rmse, nll_gaussian, -1 * lls.mean()

    def predict_mv(self, input, num_samples = 20):
        num_test   = input.shape[0]
        nn_samples = [s[0] for s in self.sample(num_samples)]
        preds      = self.sample_predict(nn_samples, input)
        return preds.mean(dim = 0), preds.var(dim = 0)
