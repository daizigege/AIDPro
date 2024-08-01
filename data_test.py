

"""Custom datasets for CelebA and CelebA-HQ."""


import torchvision.transforms as transforms
import os
import glob
import random
from PIL import Image
from torch.utils import data
import numpy as np


class CelebA(data.Dataset):
    def __init__(self, data_path, mode='a'):
        super(CelebA, self).__init__()
        self.image_dir= data_path
        self.images= sorted(os.listdir(self.image_dir))
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.image_dir, self.images[index])))
        return img, self.images[index]

    def __len__(self):
        return self.length




