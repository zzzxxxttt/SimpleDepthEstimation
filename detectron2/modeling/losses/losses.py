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


def variance_loss(depth):
    depth_var_loss = 1 / ((depth / depth.mean() - 1.0) ** 2).mean()
    return depth_var_loss


def sparsity_loss(motion_map):
    abs_motion = motion_map.abs()
    mean_abs_motion = abs_motion.mean([2, 3], keepdim=True)
    # We used L0.5 norm here because it's more sparsity encouraging than L1.
    # The coefficients are designed in a way that the norm asymptotes to L1 in
    # the small value limit.
    return (2 * mean_abs_motion * torch.sqrt(abs_motion / (mean_abs_motion + 1e-5) + 1)).mean()
