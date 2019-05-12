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

    def validate(self, X, y, num_samples = 20):
        y = y.view(X.shape[0], -1)
        with torch.no_grad():
            nns    = self.sample(num_samples)
            py, pv = self.predict_mv(X, nns)
            rmse   = torch.mean((py - y)**2, dim = 0).sqrt()
            nll    = -1 * torch.distributions.Normal(py, pv.sqrt()).log_prob(y).mean(dim = 0)
        return rmse, nll

    def predict_mv(self, input, nn_samples):
        num_test = input.shape[0]
        preds    = self.sample_predict(nn_samples, input).view(len(nn_samples), num_test, -1)
        return preds.mean(dim = 0), preds.var(dim = 0)
