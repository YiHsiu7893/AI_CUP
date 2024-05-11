import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, data_folder, target_folder, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.data_folder = data_folder
        self.transform = transforms_
        self.data_files = sorted(os.listdir(data_folder))
        self.target_folder = target_folder
        self.target_files = sorted(os.listdir(target_folder))
        self.mode = mode

    def __getitem__(self, idx):
        data_name = self.data_files[idx]
        data_path = os.path.join(self.data_folder, data_name)
        target_name = self.target_files[idx]
        target_path = os.path.join(self.target_folder, target_name)

        # Load input data image
        data = Image.open(data_path).convert("RGB")

        # Assuming target images have the same filename but in a different folder
        target = Image.open(target_path).convert("RGB")

        if self.mode == "train" and self.transform:
            data = self.transform(data)
            target = self.transform(target)

        return data, target

    def __len__(self):
        return len(self.data_files)
