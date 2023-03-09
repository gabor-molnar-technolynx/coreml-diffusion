import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import numpy as np
from PIL import Image


class CelebDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_labels = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)
        image = image.detach().cpu().numpy()
        image = np.moveaxis(image, 0, -1)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.img_labels[idx]


        if torch.cuda.is_initialized():
            image = image.cuda()

        return image, label