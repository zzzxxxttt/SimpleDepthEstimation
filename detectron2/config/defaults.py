# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = ""

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR).
# To train on images of different number of channels, just set different mean & std.
# Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
_C.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]

# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
_C.MODEL.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

_C.INPUT = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASETS = CN()

# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = ""
_C.DATASETS.TRAIN.SPLIT = ""
_C.DATASETS.TRAIN.DATA_ROOT = ""
_C.DATASETS.TRAIN.IMG_WIDTH = 768
_C.DATASETS.TRAIN.IMG_HEIGHT = 384
_C.DATASETS.TRAIN.PREPROCESS = []

# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = ""
_C.DATASETS.TEST.SPLIT = ""
_C.DATASETS.TEST.DATA_ROOT = ""
_C.DATASETS.TEST.IMG_WIDTH = 768
_C.DATASETS.TEST.IMG_HEIGHT = 384
_C.DATASETS.TEST.PREPROCESS = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 6
# Options: TrainingSampler, RepeatFactorTrainingSampler
_C.DATALOADER.SAMPLER_TRAIN = "DDPSampler"

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #

_C.MODEL.DEPTH_NET = CN()
_C.MODEL.DEPTH_NET.NAME = ""
_C.MODEL.MAX_DEPTH = 80

# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #

_C.LOSS = CN()

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 10
_C.SOLVER.DEPTH_LR = 0.001

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 1

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.SOLVER.IMS_PER_BATCH = 16

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# The period (in terms of epochs) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 1

_C.EVALUATORS = ("",)

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Directory where output files are written
_C.OUTPUT_DIR = "./output"

# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0
_C.LOG_PERIOD = 20

_C.RUN_NAME = ""
# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
