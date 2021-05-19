import torch


def gradient_x(image):
    """
    Calculates the gradient of an image in the x dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_x : torch.Tensor [B,3,H,W-1]
        Gradient of image with respect to x
    """
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def gradient_y(image):
    """
    Calculates the gradient of an image in the y dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_y : torch.Tensor [B,3,H-1,W]
        Gradient of image with respect to y
    """
    return image[:, :, :-1, :] - image[:, :, 1:, :]


def cal_smoothness_loss(depth, image):
    """
    Calculate smoothness values for inverse depths

    Parameters
    ----------
    depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps
    images : list of torch.Tensor [B,3,H,W]
        Inverse depth maps
    num_scales : int
        Number of scales considered

    Returns
    -------
    smoothness_x : list of torch.Tensor [B,1,H,W]
        Smoothness values in direction x
    smoothness_y : list of torch.Tensor [B,1,H,W]
        Smoothness values in direction y
    """
    inv_depth = 1. / depth.clamp(min=1e-6)

    mean_inv_depth = inv_depth.mean(2, True).mean(3, True)
    inv_depths_norm = inv_depth / mean_inv_depth.clamp(min=1e-6)

    inv_depth_gradients_x = gradient_x(inv_depths_norm)
    inv_depth_gradients_y = gradient_y(inv_depths_norm)

    image_gradients_x = gradient_x(image)
    image_gradients_y = gradient_y(image)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    # Note: Fix gradient addition
    smoothness_x = inv_depth_gradients_x * weights_x
    smoothness_y = inv_depth_gradients_y * weights_y

    return smoothness_x.abs().mean() + smoothness_y.abs().mean()


def cal_motion_smoothness_loss(motion_field, warp_around=False):
    motion_gradients_x = gradient_x(motion_field)
    motion_gradients_y = gradient_y(motion_field)

    return torch.sqrt(1e-5 + motion_gradients_x ** 2 + motion_gradients_y ** 2).mean()
