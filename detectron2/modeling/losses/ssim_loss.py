import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
        """
        Structural SIMilarity (SSIM) distance between two images.

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        w : torch.Tensor [B,1,H,W]
        C1,C2 : float
            SSIM parameters
        kernel_size,stride : int
            Convolutional parameters

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM distance
        """
        super(SSIM, self).__init__()

        self.C1 = C1
        self.C2 = C2

        self.pool2d = nn.AvgPool2d(kernel_size, stride=stride)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x, y):
        x, y = self.pad(x), self.pad(y)
        mu_x = self.pool2d(x)
        mu_y = self.pool2d(y)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = self.pool2d(x.pow(2)) - mu_x_sq
        sigma_y = self.pool2d(y.pow(2)) - mu_y_sq
        sigma_xy = self.pool2d(x * y) - mu_x_mu_y
        v1 = 2 * sigma_xy + self.C2
        v2 = sigma_x + sigma_y + self.C2

        ssim_n = (2 * mu_x_mu_y + self.C1) * v1
        ssim_d = (mu_x_sq + mu_y_sq + self.C1) * v2
        ssim = ssim_n / ssim_d

        return torch.clamp((1. - ssim) / 2., 0., 1.)


class WeightedSSIM(nn.Module):
    def __init__(self, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
        """
        Structural SIMilarity (SSIM) distance between two images.

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        w : torch.Tensor [B,1,H,W]
        C1,C2 : float
            SSIM parameters
        kernel_size,stride : int
            Convolutional parameters

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM distance
        """
        super(WeightedSSIM, self).__init__()

        self.C1 = float(C1)
        self.C2 = float(C2)

        self.pool2d = nn.AvgPool2d(kernel_size, stride=stride)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x, y, w):
        avg_w = F.avg_pool2d(w, kernel_size=3, stride=1, padding=1)
        w = w + 1e-2
        inverse_avg_w = 1.0 / (avg_w + 1e-2)

        mu_x = self.pool2d(self.pad(x * w)) * inverse_avg_w
        mu_y = self.pool2d(self.pad(y * w)) * inverse_avg_w

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = self.pool2d(self.pad(x.pow(2) * w)) * inverse_avg_w - mu_x_sq
        sigma_y = self.pool2d(self.pad(y.pow(2) * w)) * inverse_avg_w - mu_y_sq
        sigma_xy = self.pool2d(self.pad(x * y * w)) * inverse_avg_w - mu_x_mu_y

        if self.C1 == float('inf'):
            ssim_n = 2 * sigma_xy + self.C2
            ssim_d = sigma_x + sigma_y + self.C2
        elif self.C2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + self.C1
            ssim_d = mu_x ** 2 + mu_y ** 2 + self.C1
        else:
            ssim_n = (2 * sigma_xy + self.C2) * (2 * mu_x * mu_y + self.C1)
            ssim_d = (sigma_x + sigma_y + self.C2) * (mu_x ** 2 + mu_y ** 2 + self.C1)

        ssim = ssim_n / ssim_d

        return torch.clamp((1. - ssim) / 2., 0., 1.), avg_w
