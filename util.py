import torch
import torch.nn as nn

class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super(ScaleLayer, self).__init__()
        self.scale = scale
    def forward(self, input):
        return input * self.scale
    def extra_repr(self):
        return 'scale = {}'.format(self.scale)
