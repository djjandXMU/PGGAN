# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:26:16 2019

@author: djj
"""
import cv2 
import numpy as np
np.transpose

y_img_input_combine = np.zeros((1,3,256,256))
djj = np.tile(y_img_input_combine,[1,2,1,1])
#djj = y_img_input_combine.transpose([0,2,1,3])
#y_img_input_combine[0,:,:,:] = (cv2.imread('/home/usrp1/djj_cycle_GAN/img/man2woman/man/000001.jpg' )).transpose([0,3,1,2])
#djj= (cv2.imread('/home/usrp1/djj_cycle_GAN/img/man2woman/man/000001.jpg')).transpose((2,0,1))
#print(loss3)