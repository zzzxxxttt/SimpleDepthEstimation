# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip
from .semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head


__all__ = list(globals().keys())
