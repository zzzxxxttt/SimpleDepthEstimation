# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import numpy as np
import torch.utils.data
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from detectron2.utils.env import seed_all_rng
from detectron2.utils.comm import get_world_size
from detectron2.utils.registry import Registry

from .samplers import InferenceSampler, TrainingSampler
from ..utils import comm

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""

__all__ = [
    "DATASET_REGISTRY",
    "DatasetBase",
    "build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader"
]

"""
This file contains the default logic to build a dataloader for training or testing.
"""


class DatasetBase(data.Dataset):
    def batch_collator(self, batch):
        return default_collate(batch)


def build_batch_data_loader(
        dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0
):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )  # drop_last so the batch always have the same size
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=dataset.batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
def build_detection_train_loader(cfg):
    dataset = DATASET_REGISTRY.get(cfg.DATASETS.TRAIN.NAME)(cfg.DATASETS.TRAIN, cfg)
    assert isinstance(dataset, DatasetBase)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == 'DDPSampler':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=comm.get_world_size(),
                                                                  rank=comm.get_rank())
    elif sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_test_loader(cfg):
    dataset = DATASET_REGISTRY.get(cfg.DATASETS.TEST.NAME)(cfg.DATASETS.TEST, cfg)
    assert isinstance(dataset, DatasetBase)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=dataset.batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
