import torch
import torch.nn as nn

class BNN:
    def __init__(self, act = nn.ReLU(), num_hiddens = [50], conf = {}):
        pass

    def train(X, y):
        """
        X: num_train * dim matrix
        y: num_train vector
        """
        pass

    def sample(num_sample = 1):
        """
        Generate `num_sample` samples from the posterior, return a list of neural networks
        """
        pass

    def sample_predict(nns, input):
        pass


    def validate(X, y, num_samples):
        pass
