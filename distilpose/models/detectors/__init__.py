# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.detectors import *
from .top_down_dist import TopDownDistil
from .token_self_dist import TokenSelfDistil
from .top_down_epoch import TopDownE


__all__ = [
    'TopDown', 'AssociativeEmbedding', 'ParametricMesh', 'MultiTask',
    'PoseLifter', 'Interhand3D', 'PoseWarper', 'TopDownDistil','TokenSelfDistil',
    'TopDownE'
]
