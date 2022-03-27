import torch.nn as nn
import torch.nn.functional as F


class Conv2dTF(nn.Conv2d):
    def get_padding(self, x):
        # formulas below come form https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        _, _, in_height, in_width = x.shape
        filter_height, filter_width = self.kernel_size
        stride_height, stride_width = self.stride

        if in_height % stride_height == 0:
            pad_along_height = max(filter_height - stride_height, 0)
        else:
            pad_along_height = max(filter_height - (in_height % stride_height), 0)

        if in_width % stride_width == 0:
            pad_along_width = max(filter_width - stride_width, 0)
        else:
            pad_along_width = max(filter_width - (in_width % stride_width), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return pad_left, pad_right, pad_top, pad_bottom

    def forward(self, x):
        assert all(p == 0 for p in self.padding)
        paddings = self.get_padding(x)
        out = F.pad(x, paddings)
        return super().forward(out)


class ConvTranspose2dTF(nn.ConvTranspose2d):
    def forward(self, x):
        out = super().forward(x)
        return out[:, :, :-1, :-1]


class MaxPool2dTF(nn.MaxPool2d):
    def get_padding(self, x):
        # formulas below come form https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        _, _, in_height, in_width = x.shape

        if in_height % self.stride == 0:
            pad_along_height = max(self.kernel_size - self.stride, 0)
        else:
            pad_along_height = max(self.kernel_size - (in_height % self.stride), 0)

        if in_width % self.stride == 0:
            pad_along_width = max(self.kernel_size - self.stride, 0)
        else:
            pad_along_width = max(self.kernel_size - (in_width % self.stride), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return pad_left, pad_right, pad_top, pad_bottom

    def forward(self, x):
        assert self.padding == 0
        paddings = self.get_padding(x)
        out = F.pad(x, paddings)
        return super().forward(out)
