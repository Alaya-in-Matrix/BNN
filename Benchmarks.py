import torch
import numpy as np
import hpolib

## Single-objective benchmarks from hpolib
def quad(x):
    return torch.sum((10*(x - 0.5))**2, dim = 1).view(-1, 1)

## Constrained optimization problems from CEC

## Multi-objective optimization problems from ZDT
