#from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
# from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F


class MyDataset(Dataset):
    def __init__(self, data_dir, max_len=-1, img_wh=None, downSample=1.0, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.build_metas()
        self.define_transforms()
#         self.imgs = []
#         self.sample = {}
        
    def build_metas(self):
        # self.imgs = []
        self.index_list = []    # {scan, v, l}
        different_views = range(0, 49)  # 1-49 different position per scene
        different_lights = range(7) # 0-6 different brightness
        with open(f'mvsnerf/configs/lists/dtu_{self.split}_all.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        for scan in self.scans:
            for v in different_views:
                for l in different_lights:
#                    img_filename = os.path.join(self.data_dir,f'Rectified/{scan}_train/rect_{v + 1:03d}_{l}_r5000.png')
                    self.index_list += [(scan, v, l)]
                    # img = Image.open(img_filename)
                    # img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
                    # img = img.resize(img_wh, Image.BILINEAR)
                    # img = self.transform(img)   # normalize
                    # self.imgs += [img]
#         self.imgs = torch.stack(self.imgs).float()
#         sample['images'] = imgs
            
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])

    def __len__(self):
        return len(self.index_list) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        scan, v, l = self.index_list[idx]
        img_filename = os.path.join(self.data_dir,f'Rectified/{scan}_train/rect_{v + 1:03d}_{l}_r5000.png')
        img = Image.open(img_filename)
        img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
        img = img.resize(img_wh, Image.BILINEAR)
        img = self.transform(img)   # normalize
        return img

def get_data_loaders(args):
    train_dataset = MyDataset(args.data_dir, max_len=-1, downSample=args.img_downscale, split='train')
    val_dataset   = MyDataset(args.data_dir, max_len=10, downSample=args.img_downscale, split='val')
    print("Complete")
    return train_dataset, val_dataset
            

