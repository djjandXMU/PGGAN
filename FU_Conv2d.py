#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:39:59 2019

@author: usrp1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, calculate_gain
import numpy as np

class FUConv2d(nn.Module):
    def __init__(self, in_channels,out_channels, 
                 pixelnorm=False, wscale=True, act = None,gain = 1):
        super(FUConv2d, self).__init__()

        if wscale:
            init = lambda x: nn.init.kaiming_normal(x)
        else:
            init = lambda x: x
        self.conv = nn.Linear(in_channels, out_channels)
        init(self.conv.weight)
        if wscale:
            self.c = (gain/np.sqrt((self.conv.weight).shape[1]))
            self.conv.weight.data /= self.c
        else:
            self.c = 1.
        self.eps = 1e-8

        self.pixelnorm = pixelnorm
        if act is not None:
            self.act = nn.LeakyReLU(0.2) if act == 'lrelu' else nn.ReLU()
        else:
            self.act = None
        self.conv.cuda(1)

    def forward(self, x):
        h = x * (self.c)
        h = self.conv(h)
        if self.act is not None:
            h = self.act(h)
        if self.pixelnorm:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        return h
    