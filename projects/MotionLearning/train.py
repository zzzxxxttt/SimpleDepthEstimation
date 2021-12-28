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

import os
import sys
import logging
from functools import partial

sys.path.insert(0, '/root/data/det2')

os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # todo

import torch
import torch.nn as nn

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import build_evaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.events import EventStorage
from detectron2.utils.setup import simple_main

from projects.Supervised.train import get_evaluator

logger = logging.getLogger("detectron2")


def do_test(cfg, model, data_loader):
    if data_loader is None:
        return {}
    evaluator = get_evaluator(cfg, os.path.join(cfg.OUTPUT_DIR, "inference", cfg.DATASETS.TEST.NAME))
    if cfg.MODEL.DEPTH_NET.RAMPUP_ITERS > 0:
        model.module.depth_net.set_stddev(0.0)
    results = inference_on_dataset(model, data_loader, evaluator)
    return results


def do_train(cfg, model, resume=False):
    model.train()

    data_loader = build_detection_train_loader(cfg)
    data_loader_test = build_detection_test_loader(cfg)

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

    checkpointer = DetectionCheckpointer(model.module, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
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

            periodic_checkpointer.step(epoch)

            if cfg.TEST.EVAL_PERIOD > 0 and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0:
                eval_results = do_test(cfg, model, data_loader_test)
                for tag in eval_results:
                    storage.put_scalars(**{f"{tag}/k": v for k, v in eval_results[tag].items()})

                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(partial(simple_main, train_fn=do_train, test_fn=do_test),
           args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           args=(args,))
