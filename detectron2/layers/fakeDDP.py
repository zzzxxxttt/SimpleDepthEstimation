import torch.nn as nn


class FakeDDP(nn.Module):
    def __init__(self, model):
        super(FakeDDP, self).__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)
