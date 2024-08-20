import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import model
import losses
import numpy as np
from torchvision import transforms
import Mydataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def train_loop(config):
    
    DCE_net = model.enhance_net().cuda()
    train_dataset = Mydataloader.ImageDepthLoader("/root/Zero-DCE/Zero-DCE_code/data/DepthAndImage/", "/root/Zero-DCE/Zero-DCE_code/data/DepthAndImage/Image/", "/root/Zero-DCE/Zero-DCE_code/data/DepthAndImage/Depth/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    c = 1e-5
    L_color = losses.L_color()
    L_spa = losses.L_spa()
    L_exp = losses.L_exp(16,0.5)
    L_TV = losses.L_TV()
    L_power = losses.L_power(0.8)
    L_Depth = losses.L_Depth(0.8)
    L_color_frac = losses.L_color_frac(11)
    
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    DCE_net.train()
    
    for epoch in range(config.num_epochs):
        print(epoch)
        for iteration, (image, depth) in enumerate(train_loader):
            input_img = input_img.cuda()
            depth = depth.unsqueeze(1).cuda()

            output_img_1, output_img, A  = DCE_net(input_img, depth)

            Loss_TV = 200 * L_TV(A)

            loss_spa = torch.mean(L_spa(output_img, input_img))

            loss_col = 15 * torch.mean(L_color(output_img))
            loss_col_frac = 15 * L_color_frac(input_img, output_img)

            loss_exp = 10*torch.mean(L_exp(output_img))
            loss_power = 5 * torch.mean(L_power(input_img, output_img))

            loss_depth = c * L_Depth(output_img, depth)

            # best_loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp + loss_power + loss_col_frac + loss_depth
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
            optimizer.step()
            
            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
                print("loss_power:", loss_power.item() / c)   
            if ((iteration+1) % config.snapshot_iter) == 0:
                if(epoch == 99):
                    torch.save(DCE_net.state_dict(), config.snapshots_folder + "test1" + str(epoch+1) + '.pth')
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--images_path', type=str, default="/root/Zero-DCE/Zero-DCE_code/data/DepthAndImage/Image/")
    parser.add_argument('--depths_path', type=str, default="/root/Zero-DCE/Zero-DCE_code/data/DepthAndImage/Depth/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "./snapshots/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train_loop(config)