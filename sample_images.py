import torch
import matplotlib.pyplot as plt
from dataset import CelebDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from trainer import Trainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--T", default=300, type=int, help="Number of timesteps.")
args = parser.parse_args()

@torch.no_grad()
def sample_plot_image(trainer, device):
    # Sample noise
    img_size = args.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(args.T / num_images)

    for i in range(0, args.T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = trainer.sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()