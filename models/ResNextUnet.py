#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .unet_parts import *
from torchvision import models
import torch.nn as nn

model = models.resnext50_32x4d(pretrained=True)

class ResNextUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResNextUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.backbone = list(model.children())
        
        self.inc = DoubleConv(n_channels, 64)
        self.down0 = nn.Sequential(*self.backbone[:3])
        self.down1 = nn.Sequential(*self.backbone[3:5])
        self.down2 = self.backbone[5]
        self.down3 = self.backbone[6]
        factor = 2 if bilinear else 1
        self.down4 = self.backbone[7]
        
        self.up0 = Up(2048+1024, 1024 // factor, bilinear)
        self.up1 = Up(1024 // factor + 512, 512 // factor, bilinear)
        self.up2 = Up(512 // factor + 256, 256 // factor, bilinear)
        self.up3 = Up(256//factor + 64, 128 // factor, bilinear)
        self.up4 = Up(128 // factor + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = self.inc(x)
        
        x1 = self.down0(x) # 64
        x2 = self.down1(x1)# 256
        x3 = self.down2(x2)# 512
        x4 = self.down3(x3)# 1024
        x5 = self.down4(x4)# 2048
        
        x = self.up0(x5, x4) #(3072) -> 512
        
        x = self.up1(x, x3) #(512 + 512) - > 256
        x = self.up2(x, x2) #(256 + 256) -> 128
        x = self.up3(x, x1) #(128 + 64) -> 64
        x = self.up4(x, x0) #(64 + 64) -> 64
        out = self.outc(x)
        
#         out = self.activation(out)
        
        return out

