import os
import argparse
import pickle
from glob import glob

import cv2
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np
from multiprocessing import Pool

import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection, \
    convert_range_image_to_point_cloud

parser = argparse.ArgumentParser(description="extract waymo data")
parser.add_argument("--src_dir", default='/data3/data/waymo_raw')
parser.add_argument("--dst_dir", default='/data2/data/waymo')
parser.add_argument("--split", default='validation')
args = parser.parse_args()

T = np.array([[0, -1, 0, 0],
              [0, 0, -1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])


def points2img(points, extrinsics, intrinsics):
    T = intrinsics @ extrinsics
    proj = (T[:3, :3] @ points.T + T[:3, [3]]).T
    proj[:, :2] /= proj[:, [2]]
    return proj


def process(idx):
    global args
    global file_list

    segment = file_list[idx].split('/')[-1].replace('.tfrecord', '')
    os.makedirs(os.path.join(args.dst_dir, args.split, 'image', segment), exist_ok=True)

    info = {'cams': {},
            'frames': {}}

    dataset = tf.data.TFRecordDataset(file_list[idx], compression_type='')
    for i, data in enumerate(dataset):

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        frame_time = frame.timestamp_micros
        info['frames'][frame_time] = {'cams': {},
                                      'ego2world': np.array(frame.pose.transform).reshape([4, 4])}

        range_imgs, cam_projs, range_img_top_pose = parse_range_image_and_camera_projection(frame)
        points, cam_proj_points = \
            convert_range_image_to_point_cloud(frame, range_imgs, cam_projs, range_img_top_pose, 0)

        # use top lidar
        points = points[0]
        cam_proj_points = cam_proj_points[0]

        calibs = {f.name: f for f in frame.context.camera_calibrations}

        frame.images.sort(key=lambda x: x.name)
        for image in frame.images:
            cam_id = image.name
            cam_name = open_dataset.CameraName.Name.Name(cam_id)
            cam_timestamp = str(image.camera_trigger_time).replace('.', '').ljust(17, '0')

            img = tf.image.decode_jpeg(image.image).numpy()

            cam_calib = calibs[cam_id]
            extrinsic = np.array(cam_calib.extrinsic.transform).reshape([4, 4])
            f_u, f_v, c_u, c_v, k_1, k_2, p_1, p_2, k_3 = cam_calib.intrinsic
            intrinsic = np.array([[f_u, 0, c_u, 0],
                                  [0, f_v, c_v, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

            info['cams'][cam_name] = {'extrinsics': extrinsic,
                                      'intrinsics': intrinsic,
                                      'height': cam_calib.height,
                                      'width': cam_calib.width}
            info['frames'][frame_time]['cams'][cam_name] = cam_timestamp

            cp_points_tensor = tf.constant(cam_proj_points, dtype=tf.int32)
            mask = tf.equal(cp_points_tensor[..., 0], frame.images[cam_id - 1].name)

            cp_points_tensor = tf.cast(tf.gather_nd(cp_points_tensor, tf.where(mask)), dtype=tf.int32).numpy()
            points_tensor = tf.gather_nd(points, tf.where(mask))

            pts = points_tensor.numpy()
            proj_ours = points2img(pts, np.linalg.inv(extrinsic), intrinsic @ T)

            os.makedirs(os.path.join(args.dst_dir, args.split, 'image', segment, cam_name), exist_ok=True)
            cv2.imwrite(os.path.join(args.dst_dir, args.split, 'image', segment, cam_name, str(cam_timestamp) + '.jpg'),
                        img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

            os.makedirs(os.path.join(args.dst_dir, args.split, 'depth', segment, cam_name), exist_ok=True)
            depth = np.zeros([img.shape[0], img.shape[1]])
            depth[cp_points_tensor[:, 2], cp_points_tensor[:, 1]] = proj_ours[:, 2]
            depth = (depth * 255).astype(np.uint16)
            cv2.imwrite(os.path.join(args.dst_dir, args.split, 'depth', segment, cam_name, cam_timestamp + '.png'),
                        depth, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    return {segment: info}


if __name__ == '__main__':
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, args.split), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, args.split, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, args.split, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'infos'), exist_ok=True)

    file_list = glob(f'{args.src_dir}/{args.split}/*.tfrecord')

    # process(0)

    with Pool(5) as p:  # change according to your cpu
        r = list(tqdm(p.imap(process, range(len(file_list))), total=len(file_list)))

    infos = {}
    for r_ in r:
        infos.update(r_)

    with open(os.path.join(args.dst_dir, 'infos', f'{args.split}_infos.pkl'), 'wb') as handle:
        pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
