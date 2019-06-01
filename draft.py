# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:03:28 2019

@author: djj
"""
import torch
import tensorflow as tf

b = torch.ones((1,2,2,3))
c = b.repeat(1,2,5,2)