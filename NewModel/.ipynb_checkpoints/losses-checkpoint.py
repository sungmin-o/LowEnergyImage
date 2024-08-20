import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

class L_color_frac(nn.Module):
    def __init__(self, window_size = 11):
        super(L_color_frac, self).__init__()
        self.window_size = window_size
        self.eps = 1e-8
        
    def forward(self, input_x, output_x):
        b, c, h, w = input_x.shape
        padding = self.window_size // 2
        
        input_mean = F.avg_pool2d(input_x, self.window_size, stride = 1, padding = padding)
        output_mean = F.avg_pool2d(output_x, self.window_size, stride = 1, padding = padding)
        
        input_sum = torch.sum(input_mean, dim = 1, keepdim = True) + self.eps
        output_sum = torch.sum(output_mean, dim = 1, keepdim = True) + self.eps
        
        input_ratio = input_mean / input_sum
        output_ratio = output_mean / output_sum
        
        loss = torch.sqrt(torch.mean((input_ratio - output_ratio) ** 2))
        
        return loss
    

class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
    
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = 0.2 * x[:, 0:1 , :, :] + 0.7 * x[:, 1:2, :, :] + 0.1 * x[:, 2:3, :, :]
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
    
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    
class L_power(nn.Module):
    def __init__(self, target_power = 0.8):
        super(L_power, self).__init__()
        self.target = target_power
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        intensity_x = 0.2 * x[:, 0, :, :] + 0.6 * x[:, 1, :, :] + 0.1 * x[:, 2, :, :] # b h w
        intensity_y = 0.2 * y[:, 0, :, :] + 0.6 * y[:, 1, :, :] + 0.1 * y[:, 2, :, :]
        
        intensity_x = intensity_x.reshape(b, -1)
        intensity_y = intensity_y.reshape(b, -1)
        
        power_x = torch.sum(intensity_x ** 2.2, dim = 1)
        power_y = torch.sum(intensity_y ** 2.2, dim = 1)
        
        return torch.abs(0.8 - power_y / (power_x + 1e-6))
    
    
class L_Depth(nn.Module):
    def __init__(self, target_power = 0.8):
        super(L_Depth, self).__init__()
        self.target = target_power
    def forward(self, output_image, depth):
        b, c, h, w = output_image.shape
        
        illumination = 0.2 * output_image[:, 0:1, :, :] + 0.7 * output_image[:, 1:2, :, :] + 0.1 * output_image[:, 2:3, :, :]
        #b, 1, h, w = depth
        
        losses = list()
        
        for i in range(b):
            illum_sample = illumination[i] #1 256 256
            depth_sample = depth[i] #1 h w
            
            for pixel in range(255): #   0 --- far, 256 --- close
                average_intensity_pixel = illum_sample[depth_sample == pixel / 255.0].mean() # 낮아야함.
                average_intensity_pixel_plus_one = illum_sample[depth_sample == (pixel + 1) / 255.0].mean() # 높아야함.
                
                losses.append(max(0, average_intensity_pixel - average_intensity_pixel_plus_one))
        
        loss = sum(losses) / (255 * b)
        
        return loss