# -*- coding: utf-8 -*-
"""
file: dataset.py

@author: Suhail.Alnahari

@description: Pytorch Custom dataset that expects raccoon images and a label csv file

@created: 2021-04-06T18:27:51.082Z-05:00

@last-modified: 2021-04-13T10:55:09.697Z-05:00
"""

# standard library
import os

# 3rd party packages
import pandas as pd
from PIL import Image
import torch.utils.data as data
# local source

class RaccoonDataset(data.Dataset):
    labels =None
    def __init__(self,root="..",
                 img_folder = "Generate_Individual_IDs_dataset/croppedImages",
                 transforms = None, byol = False
                 ):
        self.img_folder = img_folder
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.refresh()
        self.transforms = transforms
        self.training_byol = byol

    def refresh(self):
        self.imgs = list(sorted([i for i in os.listdir(os.path.join(self.root, self.img_folder)) if (('.png' in i) or ('.jpg' in i) or ('.jpeg' in i))]))

    def __getitem__(self, idx):
        # load images ad masks
        if(isinstance(idx, str)):
            try:
                idx = int(idx)
            except:
                try:
                    if ((idx[-4:] == '.png') or (idx[-4:] == '.jpg')):
                        idx = self.imgs.index(idx[:-4]+'.jpg')
                    else:
                        idx = self.imgs.index(idx)
                except:
                    print("invalid index")
                    idx = np.random.randint(low=0,high=len(self.imgs))
        img_path = os.path.join(self.img_folder, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if not(self.transforms is None):
            img = self.transforms(img)
        if self.training_byol:
            return img
        return img, 1

    def __len__(self):
        return len(self.imgs)

    def __iter__(self,idx=0):
        while(idx < len(self)):
            idx+= 1
            yield self[idx-1]

