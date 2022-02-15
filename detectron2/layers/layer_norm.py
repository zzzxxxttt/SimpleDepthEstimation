import numpy as np

import torch
import torch.nn as nn


class RandLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-3, elementwise_affine=True):
        super(RandLayerNorm, self).__init__()
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
            mean *= 1.0 + torch.fmod(torch.randn_like(mean) * self.stddev, self.stddev * 2)
            var *= 1.0 + torch.fmod(torch.randn_like(var) * self.stddev, self.stddev * 2)
        out = (input - mean.detach()) / (var.detach() + self.eps)
        out = self.weight.view(1, -1, 1, 1) * out + self.bias.view(1, -1, 1, 1)
        return out


if __name__ == '__main__':
    norm = RandLayerNorm(num_channels=32)

    input = torch.randn(3, 32, 16, 16)
    out1 = norm(input)
    out2 = norm(input)

    pass
