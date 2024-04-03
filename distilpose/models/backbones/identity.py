# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.models.builder import BACKBONES
from mmpose.models.backbones.base_backbone import BaseBackbone

@BACKBONES.register_module()
class IdentityNet(BaseBackbone):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    def forward(self, x):
        """Forward function."""
        x = self.net(x)
        return x 
