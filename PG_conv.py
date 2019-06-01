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

#class PGConv2d(nn.Module):
#    def __init__(self, in_channels,out_channels, kernel_size=3, stride=1, padding=1,
#                 pixelnorm=True, wscale=True, act='lrelu'):
#        super(PGConv2d, self).__init__()
#
#        if wscale:
#            init = lambda x: nn.init.kaiming_normal(x)
#        else:
#            init = lambda x: x
#        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,  stride=stride, padding=padding,kernel_size=kernel_size)
#        init(self.conv.weight)
#        if wscale:
#            self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2))
#            self.conv.weight.data /= self.c
#        else:
#            self.c = 1.
#        self.eps = 1e-8
#
#        self.pixelnorm = pixelnorm
#        if act is not None:
#            self.act = nn.LeakyReLU(0.2) if act == 'lrelu' else nn.ReLU()
#        else:
#            self.act = None
#        self.conv.cuda(1)
#
#    def forward(self, x):
#        h = x * (self.c).cuda(0)
#        h = self.conv(h)
#        if self.act is not None:
#            h = self.act(h)
#        if self.pixelnorm:
#            mean = torch.mean(h * h, 1, keepdim=True)
#            dom = torch.rsqrt(mean + self.eps)
#            h = h * dom
#        return h


class PGConv2d(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=3, stride=1, padding=1,
                 pixelnorm=True, wscale=True, act='lrelu',gain = np.sqrt(2)):
        super(PGConv2d, self).__init__()

        if wscale:
            init = lambda x: nn.init.kaiming_normal(x)
        else:
            init = lambda x: x
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,  stride=stride, padding=padding,kernel_size=kernel_size)
        init(self.conv.weight)
        if wscale:
            self.c = (gain/np.sqrt((self.conv.weight).shape[1]*(self.conv.weight).shape[2]*(self.conv.weight).shape[3]))
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