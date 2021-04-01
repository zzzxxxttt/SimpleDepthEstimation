import torch.nn as nn


class LayerNorm(nn.Module):

    def forward(self, x):
        x, stddev = x
        return x, stddev
