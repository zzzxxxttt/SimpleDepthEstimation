import os
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.memory import to_numpy
from detectron2.utils.logger import setup_logger
from detectron2.data.preprocess.build import build_preprocess
from detectron2.modeling.meta_arch.build import build_model

parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
parser.add_argument("--cfg", default='output/Supervised_resnet_r50/config.yaml')
parser.add_argument("--input", default='/data2/data/kitti/kitti_raw/2011_09_26/'
                                       '2011_09_26_drive_0022_sync/image_02/data')
parser.add_argument("--output", default='./imgs')
parser.add_argument("--opts",
                    help="Modify config options using the command-line 'KEY VALUE' pairs",
                    default=['MODEL.WEIGHTS', 'output/Supervised_resnet_r50/model_0000020.pth'],
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

setup_logger(name="detectron2")
logger = setup_logger()
logger.info("Arguments: " + str(args))

if __name__ == "__main__":
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.freeze()

    preprocesses = []
    for preprocess_cfg in [{"NAME": 'LoadImg'}, {"NAME": "KBCrop"}, {"NAME": "ToTensor"}]:
        preprocesses.append(build_preprocess(preprocess_cfg))

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))] \
        if os.path.isdir(args.input) else [args.input]

    # use PIL, to be consistent with evaluation
    plots = []
    for i, file in enumerate(tqdm(files)):
        with torch.no_grad():
            # Apply pre-processing to image.
            data = {'metadata': {'img_dir': file}}
            for preprocess in preprocesses:
                data = preprocess.forward(data)

            data['img'] = data['img'][None, ...]

            predictions = to_numpy(model(data))

            data['depth_pred'] = predictions['depth_pred'][0]
            for postprocess in preprocesses:
                data = postprocess.backward(data)

            plt.close('all')
            fig = plt.figure(figsize=(data['depth_pred'].shape[1] / 100, data['depth_pred'].shape[0] / 100))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(data['depth_pred'] ** 0.8, cmap='magma', vmin=0.0, vmax=cfg.MODEL.MAX_DEPTH ** 0.8)
            # plt.savefig(os.path.join(args.output, 'depth_' + os.path.split(args.input)[-1]),
            #             dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plots.append((plot, cv2.imread(file)))

    if len(plots) == 1:
        plt.close('all')
        f, ax = plt.subplots(2, 1)
        ax[0].imshow(plots[0][1][..., ::-1])
        ax[1].imshow(plots[0][0])
        ax[0].axis('off')
        ax[1].axis('off')
        plt.show()
    else:
        video = cv2.VideoWriter(os.path.join(os.path.dirname(cfg.MODEL.WEIGHTS), 'vis.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                10, (plots[0][0].shape[1], plots[0][0].shape[0] * 2))

        for plot, img in tqdm(plots):
            canvas = np.zeros([plot.shape[0] * 2, plot.shape[1], 3], dtype=np.uint8)
            canvas[:plot.shape[0]] = cv2.resize(img, (plot.shape[1], plot.shape[0]))
            canvas[plot.shape[0]:] = plot[..., ::-1]
            video.write(canvas)

        cv2.destroyAllWindows()
        video.release()
