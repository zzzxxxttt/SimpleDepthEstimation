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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch

import detectron2.utils.comm as comm

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.engine import default_argument_parser, default_writers, launch
from detectron2.evaluation import build_evaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.events import EventStorage
from detectron2.utils.setup import simple_main

logger = logging.getLogger("detectron2")


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
    results = inference_on_dataset(model, data_loader, evaluator)
    # if comm.is_main_process():
    #   logger.info("Evaluation results for {} in csv format:".format(dataset_name))
    #   logger.info(results_i)
    return results


def do_train(cfg, model, resume=False):
    model.train()

    data_loader = build_detection_train_loader(cfg)

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.depth_net.encoder.parameters(),
                                    'weight_decay': 1e-2},
                                   {'params': model.module.depth_net.decoder.parameters(),
                                    'weight_decay': 0}],
                                  lr=cfg.SOLVER.DEPTH_LR, eps=1e-6)

    checkpointer = DetectionCheckpointer(model.module, cfg.OUTPUT_DIR, optimizer=optimizer)
    periodic_checkpointer = \
        PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=cfg.SOLVER.MAX_EPOCHS)

    start_epoch = \
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("epoch", -1) + 1

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
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                curr_lr = (cfg.SOLVER.DEPTH_LR - cfg.SOLVER.DEPTH_END_LR) * (1 - global_step / max_iter) ** 0.9
                curr_lr += cfg.SOLVER.DEPTH_END_LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr

                if (epoch_iter + 1) % cfg.LOG_PERIOD == 0:
                    for writer in writers:
                        writer.write()

            periodic_checkpointer.step(epoch)

            if cfg.TEST.EVAL_PERIOD > 0 and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0:
                do_test(cfg, model)
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
