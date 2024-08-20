import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net(nn.Module):
    def __init__(self):
        super(enhance_net, self).__init__()
        
        self.relu = nn.ReLU(inplace = True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(4,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 
        
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices = False, ceil_mode = False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = 2)
        
    def forward(self, x, d):
        d = d.unsqueeze(0)
        input_ = torch.concat((x, d), dim = 1)
        
        x1 = self.relu(self.e_conv1(input_))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))
        
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


        o = x

        o = o + r1*(torch.pow(o,2)-o)
        o = o + r2*(torch.pow(o,2)-o)
        o = o + r3*(torch.pow(o,2)-o)
        enhance_image_1 = o + r4*(torch.pow(o,2)-o)
        o = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        o = o + r6*(torch.pow(o,2)-o)	
        o = o + r7*(torch.pow(o,2)-o)
        enhance_image = o + r8*(torch.pow(o,2)-o)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        
        
        return enhance_image_1,enhance_image,r
