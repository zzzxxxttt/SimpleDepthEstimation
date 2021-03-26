# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# File:


from .checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

__all__ = ["Checkpointer", "PeriodicCheckpointer", "DetectionCheckpointer"]
