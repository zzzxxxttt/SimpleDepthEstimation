import cv2
import numpy as np


def write_depth(depth, save_path):
    pred_depth_scaled = (depth * 255).astype(np.uint16)
    cv2.imwrite(save_path, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return
