#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:59:27 2019

@author: usrp1
"""

#! python3
# -*- coding: utf-8 -*-
"""
################################################################################################
Implementation of 'PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION'##
https://arxiv.org/pdf/1710.10196.pdf                                                          ##
################################################################################################
https://github.com/shanexn
Created: 2018-06-11
################################################################################################
"""
import os

os.environ["PATH"] = os.environ["PATH"] + ";" + r"E:\dev\python36\for_pytorch\Library\bin"

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, calculate_gain


class EqualizedLearningRateLayer(nn.Module):
    """
    Mentioned in '4.1 EQUALIZED LEARNING RATE'
    Applies equalized learning rate to the preceding layer.
    *'To initialize all bias parameters to zero and all weights
    according to the normal distribution with unit variance'
    """

    def __init__(self, layer):
        super(EqualizedLearningRateLayer, self).__init__()
        self.layer_ = layer

        # He's Initializer (He et al., 2015)
        kaiming_normal_(self.layer_.weight, a=calculate_gain('conv2d'))
        # Cause mean is 0 after He-kaiming function
        self.layer_norm_constant_ = (torch.mean(self.layer_.weight.data ** 2)) ** 0.5
        self.layer_.weight.data.copy_(self.layer_.weight.data / self.layer_norm_constant_)

        self.bias_ = self.layer_.bias if self.layer_.bias else None
        self.layer_.bias = None

    def forward(self, x):
        self.layer_norm_constant_ = self.layer_norm_constant_.type(torch.cuda.FloatTensor)
        x = self.layer_norm_constant_ * x
        if self.bias_ is not None:
            # x += self.bias.view(1, -1, 1, 1).expand_as(x)
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x