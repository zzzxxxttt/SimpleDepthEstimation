import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-3, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.stddev = 0.5
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        var, mean = torch.var_mean(input, dim=[2, 3], keepdim=True)
        if self.training:
            mean *= np.clip(1.0 + np.random.normal(loc=0.0, scale=self.stddev), a_min=0.0, a_max=None)
            var *= np.clip(1.0 + np.random.normal(loc=0.0, scale=self.stddev), a_min=0.0, a_max=None)
        out = (input - mean.detach()) / (var.detach() + self.eps)
        out = self.weight.view(1, -1, 1, 1) * out + self.bias.view(1, -1, 1, 1)
        return out


if __name__ == '__main__':
    norm = LayerNorm(num_channels=32)

    input = torch.randn(3, 32, 16, 16)
    out1 = norm(input)
    out2 = norm(input)

    pass
