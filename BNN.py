import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BNN(ABC):
    def __init__(self):
        pass

    def normalize_Xy(self, X, y, normalize):
        if normalize:
            self.x_mean = X.mean(dim = 0)
            self.x_std  = X.std(dim = 0)
            self.y_mean = y.mean()
            self.y_std  = y.std()
            self.x_std[self.x_std == 0] = torch.tensor(1.)
            if self.y_std == 0:
                self.y_std = torch.tensor(1.)
        else:
            self.x_mean = torch.ones(X.shape[1])
            self.x_std  = torch.ones(X.shape[1])
            self.y_mean = torch.tensor(0.)
            self.y_std  = torch.tensor(1.)
        self.X = (X - self.x_mean) / self.x_std
        self.y = (y - self.y_mean) / self.y_std

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

    def validate(self, X, y, num_samples = 20):
        with torch.no_grad():
            nn_samples, prec_samples = self.sample(num_samples)
            num_test     = X.shape[0]
            y            = y.reshape(num_test)
            preds        = self.sample_predict(nn_samples, X)
            noise_vars   = 1 / prec_samples
            py           = preds.mean(dim = 0)
            pv           = preds.var(dim = 0) + noise_vars.mean()
            rmse         = torch.sqrt(torch.mean((py - y)**2))
            nll_gaussian = -1 * torch.distributions.Normal(py, pv.sqrt()).log_prob(y).mean()
            normed       = (y - preds) / noise_vars.sqrt().mean()
            lls          = torch.logsumexp(-0.5 * normed**2 - 0.5 * torch.log(2 * np.pi * noise_vars.unsqueeze(1)), dim = 0) - np.log(num_samples)
        return rmse, nll_gaussian, -1 * lls.mean()

    def predict_mv(self, input, nn_samples, prec_samples):
        num_test   = input.shape[0]
        preds      = self.sample_predict(nn_samples, input)
        noise_vars = 1 / prec_samples
        return preds.mean(dim = 0), preds.var(dim = 0) + noise_vars.mean()
