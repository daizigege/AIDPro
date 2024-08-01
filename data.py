

"""Custom datasets for CelebA and CelebA-HQ."""


import torchvision.transforms as transforms
import os
import glob
import random
from PIL import Image
from torch.utils import data
import numpy as np


class VGGFace(data.Dataset):
    def __init__(self, data_path, mode='a'):
        super(VGGFace, self).__init__()
        self.image_dir= data_path
        temp_path = os.path.join(self.image_dir, '*/')
        pathes = glob.glob(temp_path)
        self.images= []
        for dir_item in pathes:
            join_path = glob.glob(os.path.join(dir_item, '*.jpg'))
            for item in join_path:
                self.images.append(item)

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.image_dir, self.images[index])))
        return img

    def __len__(self):
        return self.length




