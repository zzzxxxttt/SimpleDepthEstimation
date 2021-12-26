import torch
import torch.nn as nn


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt):
        mask = depth_gt > 1.0
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def variance_loss_fn(depth):
    depth_var_loss = 1 / ((depth / depth.mean() - 1.0) ** 2).mean()
    return depth_var_loss
