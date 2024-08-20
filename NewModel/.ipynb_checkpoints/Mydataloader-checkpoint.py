import glob
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class ImageDepthLoader(Dataset):
    def __init__(self, csv_path, image_path, depth_path, num_samples=3000):
        self.image_path = image_path
        self.image_csv = pd.read_csv(csv_path + "image.csv")[:num_samples]
        self.depth_path = depth_path
        self.depth_csv = pd.read_csv(csv_path + "depth.csv")[:num_samples]
        self.num_samples = num_samples
        self.data_size = (self.num_samples)
        self.image_size = 256
        
        print("Total training examples:", self.data_size)
        
    def __getitem__(self, idx):
        image_path, depth_path = (self.image_path + self.image_csv.iloc[idx, 1]), (self.depth_path + self.depth_csv.iloc[idx, 1])
        
        raw_image = Image.open(image_path)
        raw_image = raw_image.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        raw_image = (np.asarray(raw_image) / 255.0)
        raw_image = torch.from_numpy(raw_image).float()
        
        raw_depth = Image.open(depth_path).convert("L")
        raw_depth = raw_depth.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        raw_depth = (np.asarray(raw_depth) / 255.0)
        raw_depth = torch.from_numpy(raw_depth).float()
        
        return raw_image.permute(2, 0, 1), raw_depth

    def __len__(self):
        return self.data_size