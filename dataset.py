import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
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
        # image = image.detach().cpu().numpy()
        image = image.detach().cpu().numpy()
        image = np.moveaxis(image, 0, -1)
        image = Image.fromarray(image)
        # image = np.ones((5289, 38))

        if self.transform:
            image = self.transform(image)

        # image = image.float()
        # image = image / 255
        label = self.img_labels[idx]
        # label = torch.from_numpy(np.asarray(label))
        # label = torch.unsqueeze(label, dim=0)

        if torch.cuda.is_initialized():
            image = image.cuda()
            # label = label.cuda()

        return image, label