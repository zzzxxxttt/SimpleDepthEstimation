import argparse
import multiprocessing as mp
import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.memory import to_numpy
from detectron2.utils.logger import setup_logger
from detectron2.data.preprocess.build import build_preprocess
from detectron2.data.preprocess.loading import read_img
from detectron2.modeling.meta_arch.build import build_model

parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
parser.add_argument("--cfg", default="configs/Supervised/res_R50_50epoch.yaml")
parser.add_argument("--input", default='/data/data/kitti/kitti_raw/2011_09_26/'
                                       '2011_09_26_drive_0022_sync/image_02/data/0000000000.png')
parser.add_argument("--output", default='')
parser.add_argument("--opts",
                    help="Modify config options using the command-line 'KEY VALUE' pairs",
                    default=['MODEL.WEIGHTS', 'output/res_R50_50epoch_sup/model_0000029.pth'],
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

setup_logger(name="detectron2")
logger = setup_logger()
logger.info("Arguments: " + str(args))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.freeze()

    preprocesses = []
    for preprocess_cfg in cfg.DATASETS.TEST.get('PREPROCESS', []):
        preprocesses.append(build_preprocess(preprocess_cfg))

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # use PIL, to be consistent with evaluation
    img = read_img(args.input)
    with torch.no_grad():
        # Apply pre-processing to image.
        data = {'image': img, 'metadata': {}}
        for preprocess in preprocesses:
            data = preprocess.forward(data)

        data['image'] = data['image'][None, ...]

        predictions = to_numpy(model(data))

        data['depth_pred'] = predictions['depth_pred'][0]
        for postprocess in preprocesses:
            data = postprocess.backward(data)

        plt.imshow(data['depth_pred'], cmap='plasma')
        plt.axis('off')
        plt.show()
