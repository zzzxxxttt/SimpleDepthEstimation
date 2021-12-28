# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import os
import cv2
import logging
import numpy as np
from tabulate import tabulate

import detectron2.utils.comm as comm

from detectron2.utils.file_utils import write_depth
from .evaluator import DatasetEvaluator, EVALUATOR_REGISTRY
from ..utils.memory import to_numpy


def garg_crop(pred, gt):
    h, w = gt.shape[:2]
    pred = pred[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)]
    gt = gt[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)]
    return pred, gt


def eigen_crop(pred, gt):
    h, w = gt.shape[:2]
    pred = pred[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)]
    gt = gt[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)]
    return pred, gt


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2 + 1e-8) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


@EVALUATOR_REGISTRY.register()
class kitti_evaluator(DatasetEvaluator):
    def __init__(self, cfg, output_folder):
        super(kitti_evaluator, self).__init__(cfg)

        self._logger = logging.getLogger(__name__)
        self._distributed = comm.get_world_size() > 1

        self.min_depth = 1e-3
        self.max_depth = 80
        self.garg_crop = True
        self.eigen_crop = False
        self.use_gt_scale = cfg.TEST.GT_SCALE

        self.tag = 'kitti evaluator'
        self.metrics = []

    def reset(self):
        self.metrics = []

    def process(self, inputs, outputs):
        inputs, outputs = to_numpy(inputs), to_numpy(outputs)

        for gt, pred, metadata in zip(inputs['depth_orig'], outputs['depth_pred'], inputs['metadata']):
            gt, pred = gt.squeeze(), pred.squeeze()

            data = {'depth_pred': pred, 'metadata': metadata}
            for postprocess in self.postprocesses:
                data = postprocess.backward(data)
            pred = data['depth_pred']

            if self.garg_crop:
                pred, gt = garg_crop(pred, gt)
            elif self.eigen_crop:
                pred, gt = eigen_crop(pred, gt)

            valid_mask = np.logical_and(gt > 1e-3, gt < 80)
            if self.use_gt_scale:
                pred = pred * np.median(gt[valid_mask]) / np.median(pred[valid_mask])

            # pred[pred < self.min_depth] = self.min_depth
            # pred[pred > self.max_depth] = self.max_depth
            # pred[np.isinf(pred)] = self.max_depth
            # pred[np.isnan(pred)] = self.min_depth

            valid_mask = np.logical_and(gt > self.min_depth, gt < self.max_depth)

            if valid_mask.sum() > 0:
                self.metrics.append(compute_errors(gt[valid_mask], pred[valid_mask]))

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            metrics = comm.gather(self.metrics, dst=0)
            metrics = list(itertools.chain(*metrics))

            if not comm.is_main_process():
                return {}
        else:
            metrics = self.metrics

        if len(metrics) == 0:
            self._logger.warning("[DepthEvaluator] Did not receive valid predictions.")
            return {}

        self._logger.info(f'{self.tag}{" w/ gt scale" if self.use_gt_scale else ""}')

        results = np.mean(metrics, axis=0)

        self._logger.info(f"{'abs_rel':>7}, {'sq_rel':>7}, {'rms':>7}, {'log_rms':>7}, "
                          f"{'d1':>7}, {'d2':>7}, {'d3':>7}")
        self._logger.info(f'{results[2]:7.3f}, {results[3]:7.3f}, {results[4]:7.3f}, {results[5]:7.3f}, '
                          f'{results[6]:7.3f}, {results[7]:7.3f}, {results[8]:7.3f}')

        # Copy so the caller can do whatever with results
        return {self.tag: {'abs_rel': results[2], 'sq_rel': results[3],
                           'rms': results[4], 'log_rms': results[5],
                           'd1': results[6], 'd2': results[7], 'd3': results[8]}}


@EVALUATOR_REGISTRY.register()
class kitti_evaluator_0_30(kitti_evaluator):
    def __init__(self, cfg, output_folder):
        super(kitti_evaluator_0_30, self).__init__(cfg, output_folder)

        self.min_depth = 1e-3
        self.max_depth = 30
        self.tag = 'kitti evaluator (0-30m)'


@EVALUATOR_REGISTRY.register()
class kitti_evaluator_30_50(kitti_evaluator):
    def __init__(self, cfg, output_folder):
        super(kitti_evaluator_30_50, self).__init__(cfg, output_folder)

        self.min_depth = 30
        self.max_depth = 50
        self.tag = 'kitti evaluator (30-50m)'


@EVALUATOR_REGISTRY.register()
class kitti_evaluator_50_80(kitti_evaluator):
    def __init__(self, cfg, output_folder):
        super(kitti_evaluator_50_80, self).__init__(cfg, output_folder)

        self.min_depth = 50
        self.max_depth = 80
        self.tag = 'kitti evaluator (50-80m)'


@EVALUATOR_REGISTRY.register()
class kitti_depth_saver(DatasetEvaluator):
    def __init__(self, cfg, output_folder):
        super(kitti_depth_saver, self).__init__(cfg)

        self._logger = logging.getLogger(__name__)
        self._distributed = comm.get_world_size() > 1

        self.use_gt_scale = cfg.TEST.GT_SCALE
        self.output_folder = output_folder

    def process(self, inputs, outputs):
        inputs, outputs = to_numpy(inputs), to_numpy(outputs)

        metadatas = [{} for _ in outputs['depth_pred']]
        for k in inputs['metadata']:
            for i, v in enumerate(inputs['metadata'][k]):
                metadatas[i][k] = v

        for i, (pred, metadata) in enumerate(zip(outputs['depth_pred'], inputs['metadata'])):
            pred = pred.squeeze()

            data = {'depth_pred': pred, 'metadata': metadata}
            for postprocess in self.postprocesses:
                data = postprocess(data)
            pred = data['depth_pred']

            if self.use_gt_scale and 'depth_gt_orig' in inputs:
                gt = inputs['depth_gt_orig']
                valid_mask = np.logical_and(gt > 1e-3, gt < 80)
                pred = pred * np.median(gt[valid_mask]) / np.median(pred[valid_mask])

            save_dir = os.path.join(self.output_folder,
                                    f"{metadata['date']}_{metadata['drive']}_{metadata['img_id']}.png")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            write_depth(pred, save_dir)

    def evaluate(self):
        self._logger.info(f'depth saved to {self.output_folder}{" w/ gt scale" if self.use_gt_scale else ""}')
        return
