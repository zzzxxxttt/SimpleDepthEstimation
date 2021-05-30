#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import math
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import (build_detection_test_loader,
                             build_detection_train_loader)

from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (build_evaluator,
                                   DatasetEvaluators,
                                   inference_on_dataset)
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

from detectron2.layers.fakeDDP import FakeDDP

logger = logging.getLogger("detectron2")


def add_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.SOLVER.DEPTH_LR = 1e-5
    _C.SOLVER.POSE_LR = 1e-5
    _C.SOLVER.LR_STEPS = (10,)
    _C.SOLVER.GAMMA = 0.5
    _C.SOLVER.CLIP_GRAD = 10.0

    # `True` if cropping is used for data augmentation during training
    _C.DATASETS.TRAIN.DEPTH_ROOT = ""
    _C.DATASETS.TRAIN.KB_CROP = False
    _C.DATASETS.TRAIN.RESIZE = False
    _C.DATASETS.TRAIN.DEPTH_TYPE = "none"
    _C.DATASETS.TRAIN.FORWARD_CONTEXT = 0
    _C.DATASETS.TRAIN.BACKWARD_CONTEXT = 0
    _C.DATASETS.TRAIN.STRIDE = 0
    _C.DATASETS.TRAIN.WITH_POSE = False
    _C.DATASETS.TRAIN.WITH_CONTEXT_DEPTH = False

    _C.DATASETS.TEST.DEPTH_ROOT = ""
    _C.DATASETS.TEST.KB_CROP = False
    _C.DATASETS.TEST.RESIZE = False
    _C.DATASETS.TEST.DEPTH_TYPE = "refined"
    _C.DATASETS.TEST.FORWARD_CONTEXT = 0
    _C.DATASETS.TEST.BACKWARD_CONTEXT = 0
    _C.DATASETS.TEST.STRIDE = 0
    _C.DATASETS.TEST.WITH_POSE = False
    _C.DATASETS.TEST.WITH_CONTEXT_DEPTH = False

    _C.MODEL.DATASET = "kitti"
    _C.MODEL.DEPTH_NET.ENCODER_NAME = "resnet50_bts"

    # BTS config
    _C.MODEL.DEPTH_NET.BTS_SIZE = 512
    _C.MODEL.DEPTH_NET.BN_NO_TRACK = False
    _C.MODEL.DEPTH_NET.FIX_1ST_CONV = False
    _C.MODEL.DEPTH_NET.FIX_1ST_CONVS = False

    _C.MODEL.DEPTH_NET.UPSAMPLE_DEPTH = False
    _C.MODEL.DEPTH_NET.LEARN_SCALE = False
    _C.MODEL.DEPTH_NET.FLIP_PROB = 0.5
    _C.MODEL.DEPTH_NET.NOISE_STDDEV = 0.5
    _C.MODEL.DEPTH_NET.RAMPUP_ITERS = -1

    _C.MODEL.POSE_NET = CN()
    _C.MODEL.POSE_NET.NAME = ''
    _C.MODEL.POSE_NET.NUM_CONTEXTS = 0
    _C.MODEL.POSE_NET.USE_DEPTH = False
    _C.MODEL.POSE_NET.GROUP_NORM = False
    _C.MODEL.POSE_NET.MASK_MOTION = False
    _C.MODEL.POSE_NET.LEARN_SCALE = False
    _C.MODEL.POSE_NET.BURN_IN_ITERS = -1

    _C.LOSS.SSIM_WEIGHT = 0.0
    _C.LOSS.C1 = 0.0
    _C.LOSS.C2 = 0.0
    _C.LOSS.CLIP = 0.0
    _C.LOSS.SMOOTHNESS_WEIGHT = 0.0
    _C.LOSS.SUPERVISED_WEIGHT = 0.0
    _C.LOSS.VARIANCE_FOCUS = 0.85
    _C.LOSS.VAR_LOSS_WEIGHT = 0.0
    _C.LOSS.MOTION_SMOOTHNESS_WEIGHT = 0.0
    _C.LOSS.MOTION_SPARSITY_WEIGHT = 0.0
    _C.LOSS.ROT_CYCLE_WEIGHT = 0.0
    _C.LOSS.TRANS_CYCLE_WEIGHT = 0.0
    _C.LOSS.SCALE_NORMALIZE = False

    _C.TEST.GT_SCALE = False


def get_evaluator(cfg, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = build_evaluator(cfg, output_folder)
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    data_loader = build_detection_test_loader(cfg)
    evaluator = get_evaluator(cfg, os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST.NAME))
    if cfg.MODEL.DEPTH_NET.RAMPUP_ITERS > 0:
        model.module.depth_net.set_stddev(0.0)
    results = inference_on_dataset(model, data_loader, evaluator)
    # if comm.is_main_process():
    #   logger.info("Evaluation results for {} in csv format:".format(dataset_name))
    #   logger.info(results_i)
    return results


def do_train(cfg, model, resume=False):
    model.train()

    data_loader = build_detection_train_loader(cfg)

    optimizer = torch.optim.Adam([{'name': 'Depth',
                                   'params': model.module.depth_net.parameters(),
                                   'lr': cfg.SOLVER.DEPTH_LR,
                                   'weight_decay': 0.01},
                                  {'name': 'Pose',
                                   'params': model.module.pose_net.parameters(),
                                   'lr': cfg.SOLVER.POSE_LR,
                                   'weight_decay': 0.0}], )
    # Training parameters

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.SOLVER.LR_STEPS,
                                                     gamma=cfg.SOLVER.GAMMA)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD,
                                                 max_iter=cfg.SOLVER.MAX_EPOCHS)

    start_epoch = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    max_iter = cfg.SOLVER.MAX_EPOCHS * len(data_loader)

    writers = default_writers(cfg.OUTPUT_DIR, max_iter=max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    logger.info("Starting training from iteration {}".format(start_epoch))

    global_step = start_epoch * len(data_loader)
    with EventStorage(start_epoch) as storage:
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
            storage.epoch = epoch
            storage.max_epoch = cfg.SOLVER.MAX_EPOCHS
            for epoch_iter, data in enumerate(data_loader):
                global_step += 1
                storage.iter = global_step
                storage.epoch_iter = epoch_iter
                storage.max_epoch_iter = len(data_loader)

                if cfg.MODEL.DEPTH_NET.RAMPUP_ITERS > 0:
                    noise_stddev = cfg.MODEL.DEPTH_NET.NOISE_STDDEV * \
                                   (min(global_step / float(cfg.MODEL.DEPTH_NET.RAMPUP_ITERS), 1.0)) ** 2
                    model.module.depth_net.set_stddev(noise_stddev)

                if cfg.MODEL.POSE_NET.BURN_IN_ITERS > 0:
                    motion_weight = np.clip(2 * global_step / cfg.MODEL.POSE_NET.BURN_IN_ITERS - 1, 0.0, 1.0)
                    model.module.pose_net.motion_weight = motion_weight

                output = model(data)

                loss_dict = {k: v for k, v in output.items() if 'loss' in k}
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if (epoch_iter + 1) % cfg.LOG_PERIOD == 0:
                    for writer in writers:
                        writer.write()

            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0:
                eval_results = do_test(cfg, model)
                for tag in eval_results:
                    storage.put_scalars(**{f"{tag}/k": v for k, v in eval_results[tag].items()})

                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            periodic_checkpointer.step(epoch)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    run_name = [] if "RUN_NAME" in args.opts \
        else ["RUN_NAME", os.path.splitext(args.config_file.split('/')[-1])[0]]
    cfg.merge_from_list(args.opts + run_name)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.RUN_NAME)
    cfg.freeze()
    default_setup(cfg, args)  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    # logger.info("Model:\n{}".format(model)) # note: uncomment this to see the model structure
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )
    else:
        model = FakeDDP(model)

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
