import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import Mymodel
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import pandas as pd
import losses
 
def lowlight(image_path, depth_path, img_name, depth_name):
    L_power = losses.L_power()
    
    data_image = Image.open(image_path)
    data_depth = Image.open(depth_path).convert("L")
 

    data_image = (np.asarray(data_image) / 255.0)
    data_depth = (np.asarray(data_depth) / 255.0)
    
    
    data_image = torch.from_numpy(data_image).float()
    data_image = data_image.permute(2,0,1)
    data_image = data_image.cuda().unsqueeze(0)

    data_depth = torch.from_numpy(data_depth).float()
    data_depth = data_depth.cuda().unsqueeze(0)
    
    DCE_net = Mymodel.enhance_net().cuda()
    DCE_net.load_state_dict(torch.load('../model/Depth/DepthLoss_04100.pth'))
    start = time.time()
    _,enhanced_image,_ = DCE_net(data_image, data_depth)
    loss_power = torch.mean(L_power(data_image, enhanced_image))
    
    print("power loss: ", loss_power.item())
    
    
    
    end_time = (time.time() - start)
    print(end_time)

    torchvision.utils.save_image(enhanced_image, '../data/DepthLoss_04/' + img_name)

if __name__ == '__main__':
# test_images
    with torch.no_grad():
        filePath = '../data/DepthAndImage/'
        image_csv = pd.read_csv(filePath + 'image.csv')[2124: 2224]
        depth_csv = pd.read_csv(filePath + 'depth.csv')[2124: 2224]
        
        for i in range(100):
            image_path = filePath + 'Image/' + image_csv.iloc[i, 1]
            depth_path = filePath + 'Depth/' + depth_csv.iloc[i, 1]
            img_name = str(i) + '.jpg'
            depth_name = str(i) + '.png'
            lowlight(image_path, depth_path, img_name, depth_name)



