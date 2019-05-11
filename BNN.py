import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BNN(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X, y):
        """
        X: num_train * dim matrix
        y: num_train vector
        """
        pass

    @abstractmethod
    def sample(self, num_samples = 1):
        """
        Generate `num_sample` samples from the posterior, return a list of neural networks and posterior precisions
        """
        pass

    @abstractmethod
    def sample_predict(self, nns, input):
        pass

    def validate(self, X, y, num_samples = 20):
        with torch.no_grad():
            nn_samples   = self.sample(num_samples)
            num_test     = X.shape[0]
            y            = y.reshape(num_test)
            preds, precs = self.sample_predict(nn_samples, X)
            noise_vars   = 1 / precs
            py           = preds.mean(dim = 0)
            pv           = preds.var(dim = 0) + noise_vars.mean()
            rmse         = torch.sqrt(torch.mean((py - y)**2))
            nll_gaussian = -1 * torch.distributions.Normal(py, pv.sqrt()).log_prob(y).mean()
            normed       = (y - preds) / noise_vars.sqrt().mean()
            lls          = torch.logsumexp(-0.5 * normed**2 - 0.5 * torch.log(2 * np.pi * noise_vars.unsqueeze(1)), dim = 0) - np.log(num_samples)
        return rmse, nll_gaussian, -1 * lls.mean()

    def predict_mv(self, input, nn_samples):
        num_test     = input.shape[0]
        preds, precs = self.sample_predict(nn_samples, input)
        noise_vars   = 1 / precs
        return preds.mean(dim = 0), preds.var(dim = 0) + noise_vars.mean(dim=0)
