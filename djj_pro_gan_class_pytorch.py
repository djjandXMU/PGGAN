# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:52:03 2019

@author: djj
"""
from FU_Conv2d import FUConv2d
from PG_conv import PGConv2d
from SN import  SpectralNorm
import torch.nn as nn
import cv2
import os
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
#nn.LeakyReLU
#nn.Upsample
class GENERATOR(nn.Module):
    def __init__(self):
        super(GENERATOR, self).__init__()
###################################生成器各层结构############################################################################
        self.upsample = torch.nn.Upsample(size=None, scale_factor=[2,2], mode='nearest', align_corners=None)
#        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh() 
#        nn.AvgPool3d
        self.g_conv1_1_1 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,gain = np.sqrt(2)/4))
#        self.g_conv1_1_1 = torch.nn.init.xavier_normal(self.g_conv1_1_1.weight, gain=1)
        self.g_conv1_2_1 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv1_3_1 =( PGConv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
        
        self.g_conv2_1_2 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv2_2_2 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv2_3_2 = (PGConv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
#        self.g_conv2_assist =( nn.Conv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0))
        
        self.g_conv3_1_3 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv3_2_3 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv3_3_3 = (PGConv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
#        self.g_conv3_assist = (nn.Conv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0))
        
        self.g_conv4_1_4 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv4_2_4 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1))
        self.g_conv4_3_4 = (PGConv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
#        self.g_conv4_assist =( nn.Conv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0))
        
        self.g_conv5_1_5 = (PGConv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1))
        self.g_conv5_2_5 = (PGConv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))
        self.g_conv5_3_5 = (PGConv2d(in_channels=256,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
#        self.g_conv5_assist =( nn.Conv2d(in_channels=512,out_channels=3,kernel_size=1,stride=1,padding=0))
        
        self.g_conv6_1_6 =( PGConv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1))
        self.g_conv6_2_6 = (PGConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1))
        self.g_conv6_3_6 =( PGConv2d(in_channels=128,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
#        self.g_conv6_assist =( nn.Conv2d(in_channels=256,out_channels=3,kernel_size=1,stride=1,padding=0))
        
        self.g_conv7_1_7 = (PGConv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1))
        self.g_conv7_2_7 =( PGConv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1))
        self.g_conv7_3_7 =( PGConv2d(in_channels=64,out_channels=3,kernel_size=1,stride=1,padding=0,gain = 1,pixelnorm=False,act=None))
#        self.g_conv7_assist = (nn.Conv2d(in_channels=128,out_channels=3,kernel_size=1,stride=1,padding=0))

#######################################生成器各层结构###################################################################################


    def Pixl_norm(self,x, eps=1e-8):
        if len(x.shape) > 2:
            axis_ = 3
        else:
            axis_ = 1
    #    with tf.variable_scope('PixelNorm'):
        return x * torch.rsqrt(torch.mean(x.mul(x), dim=axis_, keepdim=True) + eps)
    
    def generator(self,x,block_num,alpha):
        x = self.Pixl_norm(x)
        if block_num==1:
            x=((self.g_conv1_1_1(x)))
            x=((self.g_conv1_2_1(x)))
            x=self.g_conv1_3_1(x)
            return (x)
        else:
            x=((self.g_conv1_1_1(x)))
            x_block_1=((self.g_conv1_2_1(x)))
        ############################block_2#######################################  
        if block_num==2:
            x =self.upsample(x_block_1)
            x = ((self.g_conv2_1_2(x)))
            x = ((self.g_conv2_2_2(x)))
            x_rgb = (self.g_conv2_3_2(x))
            
            x_rgb_assis = self.upsample((self.g_conv1_3_1(x_block_1)))
            
            return x_rgb_assis*(1-alpha) + x_rgb*alpha
        else:
            x =self.upsample(x_block_1)
            x = ((self.g_conv2_1_2(x)))
            x_block_2 = ((self.g_conv2_2_2(x)))
        ############################block_2###########################################
        
        ############################block_3###########################################
        if block_num==3:
            x =self.upsample(x_block_2)
            x = ((self.g_conv3_1_3(x)))
            x = ((self.g_conv3_2_3(x)))
            x_rgb =(self.g_conv3_3_3(x))
            
            x_rgb_assis = self.upsample((self.g_conv2_3_2(x_block_2)))
            
            return (x_rgb_assis*(1-alpha) + x_rgb*alpha)
        else:
            x =self.upsample(x_block_2)
            x = ((self.g_conv3_1_3(x)))
            x_block_3 = ((self.g_conv3_2_3(x)))
        ############################block_3###########################################
        
        
        ############################block_4###########################################
        if block_num==4:
            x =self.upsample(x_block_3)
            x = ((self.g_conv4_1_4(x)))
            x = ((self.g_conv4_2_4(x)))
            x_rgb = (self.g_conv4_3_4(x))
            
            x_rgb_assis = self.upsample((self.g_conv3_3_3(x_block_3)))
            
            return (x_rgb_assis*(1-alpha) + x_rgb*alpha)
        else:
            x =self.upsample(x_block_3)
            x = ((self.g_conv4_1_4(x)))
            x_block_4 = ((self.g_conv4_2_4(x)))
        ############################block_4###########################################
        
        ############################block_5###########################################
        if block_num==5:
            x =self.upsample(x_block_4)
            x = ((self.g_conv5_1_5(x)))
            x = ((self.g_conv5_2_5(x)))
            x_rgb = (self.g_conv5_3_5(x))
            
            x_rgb_assis = self.upsample((self.g_conv4_3_4(x_block_4)))
            
            return (x_rgb_assis*(1-alpha) + x_rgb*alpha)
        else:
            x =self.upsample(x_block_4)
            x = ((self.g_conv5_1_5(x)))
            x_block_5 = ((self.g_conv5_2_5(x)))
        ############################block_5###########################################
        
        ############################block_6###########################################
        if block_num==6:
            x =self.upsample(x_block_5)
            x = ((self.g_conv6_1_6(x)))
            x = ((self.g_conv6_2_6(x)))
            x_rgb = (self.g_conv6_3_6(x))
            
            x_rgb_assis = self.upsample((self.g_conv5_3_5(x_block_5)))
            return (x_rgb_assis*(1-alpha) + x_rgb*alpha)
        else:
            x =self.upsample(x_block_5)
            x = ((self.g_conv6_1_6(x)))
            x_block_6 = ((self.g_conv6_2_6(x)))
        ############################block_6###########################################
        
        ############################block_7###########################################
        if block_num==7:
            x =self.upsample(x_block_6)
            x = ((self.g_conv7_1_7(x)))
            x = ((self.g_conv7_2_7(x)))
            x_rgb = (self.g_conv7_3_7(x))
            
            x_rgb_assis = self.upsample((self.g_conv6_3_6(x_block_6)))
            return (x_rgb_assis*(1-alpha) + x_rgb*alpha)
        else:
            print('error')
        ############################block_7###########################################
class DISCRIMINATOR(nn.Module):
    def __init__(self):
        super(DISCRIMINATOR, self).__init__()

#######################################判决器各层结构###################################################################################
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.lrelu_d = nn.LeakyReLU(negative_slope=0.2)
        
        self.d_conv1_1_1 = ((PGConv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0,pixelnorm=False)))
        self.d_conv1_2_1 =  (PGConv2d(in_channels=513,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv1_3_1 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=4,stride=1,padding=0,pixelnorm=False))
        self.d_linear = (FUConv2d(in_channels = 512 , out_channels = 3))
        
        self.d_conv2_1_2 = (PGConv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0,pixelnorm=False))
        self.d_conv2_2_2 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv2_3_2 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
#        self.d_conv2_4_2 = (nn.Conv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0))
        
        self.d_conv3_1_3 = (PGConv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0,pixelnorm=False))
        self.d_conv3_2_3 =( PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv3_3_3 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
#        self.d_conv3_4_3 = (nn.Conv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0))
        
        self.d_conv4_1_4 = (PGConv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0,pixelnorm=False))
        self.d_conv4_2_4 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv4_3_4 =( PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
#        self.d_conv4_4_4 = (nn.Conv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0))
        
        self.d_conv5_1_5 = (PGConv2d(in_channels=3,out_channels=256,kernel_size=1,stride=1,padding=0,pixelnorm=False))
        self.d_conv5_2_5 =( PGConv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv5_3_5 = (PGConv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,pixelnorm=False))
#        self.d_conv5_4_5 = (nn.Conv2d(in_channels=3,out_channels=512,kernel_size=1,stride=1,padding=0))
        
        self.d_conv6_1_6 =(PGConv2d(in_channels=3,out_channels=128,kernel_size=1,stride=1,padding=0,pixelnorm=False))
        self.d_conv6_2_6 =(PGConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv6_3_6 =(PGConv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,pixelnorm=False))
#        self.d_conv6_4_6 = (nn.Conv2d(in_channels=3,out_channels=256,kernel_size=1,stride=1,padding=0))
        
        self.d_conv7_1_7 = (PGConv2d(in_channels=3,out_channels=64,kernel_size=1,stride=1,padding=0,pixelnorm=False))
        self.d_conv7_2_7 = (PGConv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,pixelnorm=False))
        self.d_conv7_3_7 =( PGConv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,pixelnorm=False))
#        self.d_conv7_4_7 = (nn.Conv2d(in_channels=3,out_channels=128,kernel_size=1,stride=1,padding=0))
        
        
    def MinibatchstateConcat(self,input, averaging='all'):
        s = input.shape
    #        adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) **2, **kwargs) + 1e-8)
        vals = torch.sqrt(torch.mean((input - torch.mean(input, dim=0,keepdim=True)) **2, dim=0,keepdim=True) + 1e-8)
        print(vals.size())
    #        vals = adjusted_std(input, axis=0, keep_dims=True)
        if averaging == 'all':
    #            vals = torch.mean(vals, keepdim=True)
            vals = torch.mean(vals)
        else:
            print ("nothing")
    
        vals = vals.repeat(s[0], 1,s[2], s[3])
        return torch.cat([input, vals], 1)
    
    
    def discriminator(self,image,block_num,alpha):
    
        ############################block_7###########################################
        if block_num==7:    
            x = (self.d_conv7_1_7(image))
            x = (self.d_conv7_2_7(x))
            x = (self.d_conv7_3_7(x))
            x = self.avg_pool(x)
            
            x_down_7 = self.avg_pool(image)
            x_down_7 = (self.d_conv6_1_6(x_down_7))
            
            x_down_7 =  alpha*x+(1-alpha)*x_down_7
        else:
            djj=1
        ############################block_7###########################################
        
        ############################block_6###########################################
        if block_num==7:
            x = (self.d_conv6_2_6(x_down_7))
            x = (self.d_conv6_3_6(x))
            x_down_6 = self.avg_pool(x)
            
        elif block_num==6:
            x = (self.d_conv6_1_6(image))
            x = (self.d_conv6_2_6(x))
            x = (self.d_conv6_3_6(x))
            x = self.avg_pool(x)
            
            x_down_6 = self.avg_pool(image)
            x_down_6 = (self.d_conv5_1_5(x_down_6))
            
            x_down_6 =  alpha*x+(1-alpha)*x_down_6
        else:
            djj=1
        ############################block_6###########################################
        
        ############################block_5###########################################
        if ((block_num==6)|(block_num==7)):
            x = (self.d_conv5_2_5(x_down_6))
            x = (self.d_conv5_3_5(x))
            x_down_5 = self.avg_pool(x)
        elif block_num==5:
            x = (self.d_conv5_1_5(image))
            x = (self.d_conv5_2_5(x))
            x = (self.d_conv5_3_5(x))
            x = self.avg_pool(x)
            
            x_down_5 = self.avg_pool(image)
            x_down_5 = (self.d_conv4_1_4(x_down_5))
            
            x_down_5 =  alpha*x+(1-alpha)*x_down_5
        else:
            djj=1
        ############################block_5###########################################
    
        ############################block_4###########################################
        if ((block_num==6)|(block_num==7)|(block_num==5)):
            x = (self.d_conv4_2_4(x_down_5))
            x = (self.d_conv4_3_4(x))
            x_down_4 = self.avg_pool(x)
        elif block_num==4:
            x = (self.d_conv4_1_4(image))
            x = (self.d_conv4_2_4(x))
            x = (self.d_conv4_3_4(x))
            x = self.avg_pool(x)
            
            x_down_4 = self.avg_pool(image)
            x_down_4 = (self.d_conv3_1_3(x_down_4))
            
            x_down_4 =  alpha*x+(1-alpha)*x_down_4
        else:
            djj=1
        ############################block_4###########################################
        
        ############################block_3###########################################
        if ((block_num==6)|(block_num==7)|(block_num==5)|(block_num==4)):
            x = (self.d_conv3_2_3(x_down_4))
            x = (self.d_conv3_3_3(x))
            x_down_3 = self.avg_pool(x)
        elif block_num==3:
            x = (self.d_conv3_1_3(image))
            x = (self.d_conv3_2_3(x))
            x = (self.d_conv3_3_3(x))
            x = self.avg_pool(x)
            
            x_down_3 = self.avg_pool(image)
            x_down_3 = (self.d_conv2_1_2(x_down_3))
            
            x_down_3 =  alpha*x+(1-alpha)*x_down_3
        else:
            djj=1
        ############################block_3###########################################
        
        ############################block_2###########################################
        if ((block_num==6)|(block_num==7)|(block_num==5)|(block_num==4)|(block_num==3)):
            x = (self.d_conv2_2_2(x_down_3))
            x = (self.d_conv2_3_2(x))
            x_down_2 = self.avg_pool(x)
        elif block_num==2:
            x = (self.d_conv2_1_2(image))
            x = (self.d_conv2_2_2(x))
            x = (self.d_conv2_3_2(x))
            x = self.avg_pool(x)
            
            x_down_2 = self.avg_pool(image)
            x_down_2 = (self.d_conv1_1_1(x_down_2))
            
            x_down_2 =  alpha*x+(1-alpha)*x_down_2
        else:
            djj=1
        ############################block_2###########################################
        
        ############################block_1###########################################
        if ((block_num==6)|(block_num==7)|(block_num==5)|(block_num==4)|(block_num==3)|(block_num==2)):
#                def MinibatchstateConcat(input, averaging='all'):
            x = self.MinibatchstateConcat(x_down_2)
            x = (self.d_conv1_2_1(x))
            x = (self.d_conv1_3_1(x))
            x = x.view(16,512)
            x_output = self.d_linear(x)
        elif block_num==1:
            x = (self.d_conv1_1_1(image))
            x = self.MinibatchstateConcat(x)
            x = (self.d_conv1_2_1(x))
            x = (self.d_conv1_3_1(x))
            x = x.view(16,512)
            x_output = self.d_linear(x)
        else:
            djj=1
        return x_output
        ############################block_1###########################################

        
        
    def L_adv_loss(self,x_img,target_img,alpha,i):    
        rands = (torch.rand(16,1,1,1)).cuda(1)
        interpolated = rands*x_img + (1. - rands)*target_img
        interpolated = Variable(interpolated, requires_grad=True).cuda(1)
    ####      （1，128，128，3）
    ####  （1，2，2，1）
    #    discriminator(y_img,num_block=block,max_iter=self.max_iters,step=step_input[0],reuse=True)
        logit = D.discriminator(interpolated,block_num=i+1,alpha=alpha)
        gradients = torch_grad(outputs=logit, inputs=interpolated,
                               grad_outputs=torch.ones(logit.size()).cuda(1) ,
                               create_graph=True, retain_graph=True)[0]
    #########################这是改进的WAGN#####################################333
        gradients = gradients.view(16, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return 10 * ((gradients_norm - 1) ** 2).mean()
        
        
        


        
#loss_func = torch.nn.MSELoss()
#
#gpus = [1]   #使用哪几个GPU进行训练，这里选择0号GPU
#cuda_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
##cnn = net()
##max_iter =100000
##generator(self,x,block_num,alpha)
##optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
##loss_func = nn.CrossEntropyLoss()   
#
#
#D = DISCRIMINATOR()
#G = GENERATOR()
#D = D.cuda(1)
#G = G.cuda(1)
#D.to(device)
#G.to(device)

#optimizer_d = torch.optim.Adam(D.parameters(), lr=0.001,betas=(0.0, 0.99))
#optimizer_g = torch.optim.Adam(G.parameters(), lr=0.001,betas=(0.0, 0.99))




#######################parameter initialnizer---discriminator#########################

for i in range(6):
    i=i+3
    loss_func = torch.nn.MSELoss()

    gpus = [1]   #使用哪几个GPU进行训练，这里选择0号GPU
    cuda_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #cnn = net()
    #max_iter =100000
    #generator(self,x,block_num,alpha)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    #loss_func = nn.CrossEntropyLoss()   
    
    
    D = DISCRIMINATOR()
    G = GENERATOR()
    D = D.cuda(1)
    G = G.cuda(1)
    D.to(device)
    G.to(device)

#    i=1
#    net.load_state_dict(torch.load('E:/transfer_learning/pytorch/snapshot_1/net_params1.pkl'))
    if i>=1:
        D.load_state_dict(torch.load('./snapshot/D_net_params'+str(i)+'.pkl'))
        G.load_state_dict(torch.load('./snapshot/G_net_params'+str(i)+'.pkl'))
        
        for p in D.parameters():
            p.requires_grad = True
    #
    #        
        for p in G.parameters():
            p.requires_grad = True
        
        
        
        
        optimizer_d = torch.optim.Adam(D.parameters(), lr=0.001,betas=(0.0, 0.99))
        optimizer_g = torch.optim.Adam(G.parameters(), lr=0.001,betas=(0.0, 0.99))
        
        optimizer_d.load_state_dict(torch.load('./snapshot/D_opti'+str(i)+'.pkl'))
        optimizer_g.load_state_dict(torch.load('./snapshot/G_opti'+str(i)+'.pkl'))
    else:
        for p in D.parameters():
            p.requires_grad = True
    #
    #        
        for p in G.parameters():
            p.requires_grad = True
        
        optimizer_d = torch.optim.Adam(D.parameters(), lr=0.001,betas=(0.0, 0.99))
        optimizer_g = torch.optim.Adam(G.parameters(), lr=0.001,betas=(0.0, 0.99))
    
#    optimizer_d = torch.optim.Adam(D.parameters(), lr=0.001,betas=(0.0, 0.99))
#    optimizer_g = torch.optim.Adam(G.parameters(), lr=0.001,betas=(0.0, 0.99))
#    optimizer_d = torch.optim.Adam(net.parameters(), lr=0.001,betas=(0.0, 0.99))
#    params=net.state_dict() 
#    print(params['d_conv1_2_1.weight'])
#    path_2 = os.listdir('/home/usrp1/djj_cycle_GAN/img/man2woman/man//')
    path_2 = os.listdir('//home/zcd/djj/cele/img_align_celeba_face//')
    length_path_2  = len(path_2) 
    if i==0:
        max_iter = 20000
    else:
        max_iter = 40000
    for j in range(max_iter):
        if i==0:
            alpha = 1
        else:
            alpha = j/max_iter
        
        random_seed_y = random.sample(range(length_path_2),16)
        y_img_input_combine = np.zeros((16,3,np.power(2,i+2),np.power(2,i+2),))
        for k in range(16):
            y_img_input_combine[k,:,:,:] = (np.float32(cv2.resize(cv2.imread('/home/zcd/djj/cele/img_align_celeba_face//' +path_2[random_seed_y[k]]),(np.power(2,i+2),np.power(2,i+2)),interpolation=cv2.INTER_NEAREST))).transpose((2,0,1))
        y_img_input_lower = zoom(y_img_input_combine, zoom=[1, 1, 0.5, 0.5], mode='nearest')
        y_img_input_lower = zoom(y_img_input_lower, zoom=[1, 1, 2, 2], mode='nearest')
        
        y_img_input_combine =    (alpha*y_img_input_combine + (1-alpha)*y_img_input_lower)/127.5-1
        y_img_input_combine=Variable(torch.Tensor(y_img_input_combine)).cuda(1)
        
#        L_adv_loss(self,x_img,target_img,alpha,i):  
        latent = np.random.normal(size=(16,512,1,1))
        latent = np.tile(latent,[1,1,4,4])
        latent = Variable(torch.Tensor(latent)).cuda(1)
        
        
        
    
#        
#        d_loss = torch.mean(logit_fake)-torch.mean(logit_real) + net.L_adv_loss(fake_img,y_img_input_combine,alpha=alpha,i=i)
#        d_loss = torch.mean(logit_fake)-torch.mean(logit_real)
#        g_loss = -torch.mean(logit_fake)


        
        for l in range(1):
            D.zero_grad()
            G.zero_grad()
            
            
            fake_img = G.generator(latent,block_num=i+1,alpha=alpha)
            logit_fake = D.discriminator(fake_img,block_num=i+1,alpha=alpha)
            logit_real = D.discriminator(y_img_input_combine,block_num=i+1,alpha=alpha)
            d_loss = torch.mean(logit_fake)-torch.mean(logit_real)+0.001*torch.mean((logit_real**2))+ D.L_adv_loss(fake_img,y_img_input_combine,alpha=alpha,i=i)
            d_loss.backward( )
            optimizer_d.step()
        
#        g_loss = -torch.mean(logit_fake)
#        if j%3==0:
        D.zero_grad()
        G.zero_grad()
        

        
        fake_img = G.generator(latent,block_num=i+1,alpha=alpha)
        logit_fake = D.discriminator(fake_img,block_num=i+1,alpha=alpha)
#            logit_real = D.discriminator(y_img_input_combine,block_num=i+1,alpha=alpha)
        
        
        g_loss = -torch.mean(logit_fake)
        g_loss.backward()
        optimizer_g.step()
#            print(g_loss.data.cpu().numpy())
#        optimizer_d.step()
        if j%200==0:
            img = ((fake_img).data.cpu().numpy()).transpose((0,2,3,1))
            y_img_input_combine = ((y_img_input_combine).data.cpu().numpy()).transpose((0,2,3,1))
            
            cv2.imwrite('./train_out/img_'+str(j)+'.jpg',(img[0,:,:,:]+1)*127.5)
            cv2.imwrite('./train_out/real_img_'+str(j)+'.jpg',(y_img_input_combine[0,:,:,:]+1)*127.5)
        
        print(d_loss.data.cpu().numpy())
        print(g_loss.data.cpu().numpy())
        print(j)
        print('fade-in')
      ##################fade_in#####################################################################################
      ##################fade_in#####################################################################################
    
    for j in range(max_iter):
        alpha = 1
        
        random_seed_y = random.sample(range(length_path_2),16)
        y_img_input_combine = np.zeros((16,3,np.power(2,i+2),np.power(2,i+2),))
        for k in range(16):
            y_img_input_combine[k,:,:,:] = (np.float32(cv2.resize(cv2.imread('/home/zcd/djj/cele/img_align_celeba_face//' +path_2[random_seed_y[k]]),(np.power(2,i+2),np.power(2,i+2)),interpolation=cv2.INTER_NEAREST))).transpose((2,0,1))
        y_img_input_lower = zoom(y_img_input_combine, zoom=[1, 1, 0.5, 0.5], mode='nearest')
        y_img_input_lower = zoom(y_img_input_lower, zoom=[1, 1, 2, 2], mode='nearest')
        
        y_img_input_combine =    (alpha*y_img_input_combine + (1-alpha)*y_img_input_lower)/127.5-1
        y_img_input_combine=Variable(torch.Tensor(y_img_input_combine)).cuda(1)
        
#        L_adv_loss(self,x_img,target_img,alpha,i):  
        latent = np.random.normal(size=(16,512,1,1))
        latent = np.tile(latent,[1,1,4,4])
        latent = Variable(torch.Tensor(latent)).cuda(1)
        
        
        
    
#        
#        d_loss = torch.mean(logit_fake)-torch.mean(logit_real) + net.L_adv_loss(fake_img,y_img_input_combine,alpha=alpha,i=i)
#        d_loss = torch.mean(logit_fake)-torch.mean(logit_real)
#        g_loss = -torch.mean(logit_fake)
        
        
        for l in range(1):
            D.zero_grad()
            G.zero_grad()
            

            
            fake_img = G.generator(latent,block_num=i+1,alpha=alpha)
            logit_fake = D.discriminator(fake_img,block_num=i+1,alpha=alpha)
            logit_real = D.discriminator(y_img_input_combine,block_num=i+1,alpha=alpha)
            d_loss = torch.mean(logit_fake)-torch.mean(logit_real)+0.001*torch.mean((logit_real**2))+ D.L_adv_loss(fake_img,y_img_input_combine,alpha=alpha,i=i)
            d_loss.backward( )
            optimizer_d.step()
        
#        g_loss = -torch.mean(logit_fake)
#        if j%3==0:
        D.zero_grad()
        G.zero_grad()
        
        
        fake_img = G.generator(latent,block_num=i+1,alpha=alpha)
        logit_fake = D.discriminator(fake_img,block_num=i+1,alpha=alpha)
#            logit_real = D.discriminator(y_img_input_combine,block_num=i+1,alpha=alpha)
        
        
        g_loss = -torch.mean(logit_fake)
        g_loss.backward()
        optimizer_g.step()
#            print(g_loss.data.cpu().numpy())
#        optimizer_d.step()
        if j%200==0:
            img = ((fake_img).data.cpu().numpy()).transpose((0,2,3,1))
            y_img_input_combine = ((y_img_input_combine).data.cpu().numpy()).transpose((0,2,3,1))
            
            cv2.imwrite('./train_out/img_'+str(j)+'.jpg',(img[0,:,:,:]+1)*127.5)
            cv2.imwrite('./train_out/real_img_'+str(j)+'.jpg',(y_img_input_combine[0,:,:,:]+1)*127.5)
        
        print(d_loss.data.cpu().numpy())
        print(g_loss.data.cpu().numpy())
        print(j)
        

    torch.save(D.state_dict(), './snapshot/D_net_params'+str(i+1)+'.pkl')
    torch.save(G.state_dict(), './snapshot/G_net_params'+str(i+1)+'.pkl')
    torch.save(optimizer_d.state_dict(),'./snapshot/D_opti'+str(i+1)+'.pkl')
    torch.save(optimizer_g.state_dict(),'./snapshot/G_opti'+str(i+1)+'.pkl')
#    torch.cuda.empty_cache()
        
