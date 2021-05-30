import torch
import torch.nn.functional as F

from .smoothness_loss import gradient_x, gradient_y


def motion_consistency_loss(coords_A_in_B, mask, R_A2B, R_B2A, t_A2B, t_B2A):
    B, _, H, W = t_A2B.shape

    # sample translation map given the projected reference points
    sampled_t_B2A = F.grid_sample(t_B2A, coords_A_in_B,
                                  mode='bilinear', padding_mode='zeros', align_corners=True)

    # Building a 4D transform matrix from each rotation and translation, and
    # multiplying the two, we'd get:
    #
    # (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
    # (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
    #
    # Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
    # a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
    # total translation is R2*t1 + t2.

    R2R1 = R_B2A @ R_A2B  # [B, 3, 3]
    R2t1 = R_B2A[:, None, :, :] @ t_A2B.view(B, 3, 1, -1).permute(0, 3, 1, 2)  # [B, HxW, 3, 1]

    rot_unit = R2R1
    trans_zero = R2t1[..., 0] + sampled_t_B2A.view(B, -1, 3)

    eyes = torch.eye(3, device=coords_A_in_B.device)[None, :, :].repeat(B, 1, 1)

    rot_error = ((rot_unit - eyes) ** 2).mean(dim=[1, 2])
    rot1_scale = ((R_A2B - eyes) ** 2).mean(dim=[1, 2])
    rot2_scale = ((R_B2A - eyes) ** 2).mean(dim=[1, 2])
    rot_error = (rot_error / (rot1_scale + rot2_scale + 1e-5)).mean()

    # Here again, we normalize by the magnitudes, for the same reason.
    trans_error = (trans_zero ** 2).sum(2).view(B, H, W)
    trans1_scale = (t_A2B ** 2).sum(1)
    trans2_scale = (sampled_t_B2A ** 2).sum(1)

    trans_error = trans_error / (trans1_scale + trans2_scale + 1e-5)
    trans_error = (mask[:, 0, :, :] * trans_error).mean()

    return rot_error, trans_error


def motion_smoothness_loss(motion_field, warp_around=False):
    motion_gradients_x = gradient_x(motion_field)[:, :, :-1, :]
    motion_gradients_y = gradient_y(motion_field)[:, :, :, :-1]

    return torch.sqrt(1e-5 + motion_gradients_x ** 2 + motion_gradients_y ** 2).mean()


def motion_sparsity_loss(motion_map):
    abs_motion = motion_map.abs()
    mean_abs_motion = abs_motion.mean([2, 3], keepdim=True)
    # We used L0.5 norm here because it's more sparsity encouraging than L1.
    # The coefficients are designed in a way that the norm asymptotes to L1 in
    # the small value limit.
    return (2 * mean_abs_motion * torch.sqrt(abs_motion / (mean_abs_motion + 1e-5) + 1)).mean()
