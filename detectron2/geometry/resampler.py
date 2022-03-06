import torch
import numpy as np
from numpy.linalg import matrix_rank


def safe_gather_nd(data, indices):
    """Gather slices from params into a Tensor with shape specified by indices.
    Similar functionality to tf.gather_nd with difference: when index is out of
    bound, always return 0.
    Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
        tensor.
    Returns:
    A Tensor. Has the same type as params. Values from params gathered from
    specified indices (if they exist) otherwise zeros, with shape
    indices.shape[:-1] + params.shape[indices.shape[-1]:].
    """
    params_shape = data.shape
    indices_shape = indices.shape
    slice_dimensions = indices_shape[-1]

    max_index = params_shape[:slice_dimensions] - 1
    min_index = torch.zeros_like(max_index, dtype=torch.int32)

    clipped_indices = torch.clamp(indices, min_index, max_index)

    # Check whether each component of each index is in range [min, max], and
    # allow an index only if all components are in range:
    mask = torch.all(
        torch.logical_and(indices >= min_index, indices <= max_index), dim=1)
    mask = torch.unsqueeze(mask, -1)

    return (mask.type(dtype=data.dtype) *
            torch_gather_nd(data, clipped_indices))


def torch_gather_nd(data, indices):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...], 
    which represents the location of the elements.
    '''
    data = data.permute(0, 2, 3, 1)
    indices = indices.type(torch.LongTensor)
    output = torch.zeros_like(data, device=data.device)
    ind_i = torch.arange(0, end=indices.shape[0], dtype=torch.long)
    ind_j = torch.arange(0, end=indices.shape[1], dtype=torch.long)
    ind_k = torch.arange(0, end=indices.shape[2], dtype=torch.long)
    ind = torch.meshgrid(ind_i, ind_j, ind_k)
    index = torch.unbind(indices[ind], dim=-1)
    output[index] = data[index]
    output = output.permute(0, 3, 1, 2)
    return output


def torch_gather_nd_v2(data, indices):
    B, H, W, _ = indices.shape
    indices = indices.long()
    ind_b = indices[..., 0].flatten()
    ind_h = indices[..., 1].flatten()
    ind_w = indices[..., 2].flatten()
    return data[ind_b, :, ind_h, ind_w].view(B, H, W, -1).permute(0, 3, 1, 2)


def resampler_with_unstacked_warp(data,
                                  warp_x,
                                  warp_y,
                                  safe=True):
    """Resamples input data at user defined coordinates.
    Args:
    data: [B, C, H, W]
    warp_x: [B, H, W]
    warp_y: [B, H, W]
    safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
        Disable only if you know they are within bounds, otherwise a runtime
        exception will be thrown.
    Returns:
        [B, C, H, W]
    """

    B, H, W = warp_x.shape

    # Compute the four points closest to warp with integer value.
    warp_floor_x = torch.floor(warp_x)
    warp_floor_y = torch.floor(warp_y)

    # Compute the weight for each point.
    right_warp_weight = (warp_x - warp_floor_x)[:, None, :, :]
    down_warp_weight = (warp_y - warp_floor_y)[:, None, :, :]

    warp_floor_x = warp_floor_x.int()
    warp_floor_y = warp_floor_y.int()

    warp_ceil_x = torch.ceil(warp_x).int()
    warp_ceil_y = torch.ceil(warp_y).int()

    left_warp_weight = 1 - right_warp_weight
    up_warp_weight = 1 - down_warp_weight

    # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
    # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
    # dimension being the batch index.
    warp_batch = torch.arange(start=0, end=B, dtype=torch.int32, device=warp_x.device).view(-1, 1, 1)

    # Broadcast to match shape:
    warp_batch = warp_batch + torch.zeros_like(warp_x, dtype=torch.int32, device=warp_x.device)

    up_left_warp = torch.stack([warp_batch, warp_floor_y, warp_floor_x], dim=-1)
    up_right_warp = torch.stack([warp_batch, warp_floor_y, warp_ceil_x], dim=-1)
    down_left_warp = torch.stack([warp_batch, warp_ceil_y, warp_floor_x], dim=-1)
    down_right_warp = torch.stack([warp_batch, warp_ceil_y, warp_ceil_x], dim=-1)

    gather_nd = safe_gather_nd if safe else torch_gather_nd_v2

    # gather data then take weighted average to get resample result.
    result = ((gather_nd(data, up_left_warp) * left_warp_weight +
               gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
              (gather_nd(data, down_left_warp) * left_warp_weight +
               gather_nd(data, down_right_warp) * right_warp_weight) * down_warp_weight)

    result = result.view(B, -1, H, W)
    return result
