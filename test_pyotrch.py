import os
import argparse
import torch
import coremltools as ct
from matplotlib import pyplot as plt
import numpy as np
from trainer import Trainer
from tqdm import tqdm

torch.set_grad_enabled(False)
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="model", help="Directory to save the checkpoints in.")
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--BATCH_SIZE", default=1, type=int, help="Batch count for the training process.")
parser.add_argument("--START_B", default=0.0001, type=float, help="Beta at the first timestep.")
parser.add_argument("--END_B", default=0.02, type=float, help="Beta at the last timestep.")
parser.add_argument("--T", default=300, type=int, help="Number of timesteps.")
args = parser.parse_args()


# def save_samples(trainer, out_path):
#     num_images = 5
#     fig, ax = plt.subplots((5,5))
#
#     for img_y_idx in range(num_images):
#         for img_x_idx in range(num_images):
#             img_size = trainer.IMG_SIZE
#             img = torch.randn((1, 3, img_size, img_size), device=trainer.map_device)
#             print(f"Working on image number: {img_x_idx}.")
#             for i in tqdm(range(0, trainer.T_STEPS)[::-1]):
#                 t = torch.full((1,), i, device=trainer.map_device, dtype=torch.long)
#                 img = trainer.sample_timestep(img, t)
#                 if i == 0:
#                     plt.subplot(1, num_images, img_x_idx + 1)
#                     ax[img_x_idx] = plt.imshow(trainer.to_tensor_image(img.detach().cpu()))
#
#     fig.savefig(os.path.join(out_path, "samples.png"))

def save_samples(trainer, out_path):
    num_images = 5
    fig, ax = plt.subplots(5, 5)
    for img_y_idx in range(num_images):
        for img_x_idx in range(num_images):
            img_size = trainer.IMG_SIZE
            img = torch.randn((1, 3, img_size, img_size), device=trainer.map_device)
            print(f"Working on image number_x: {img_x_idx}, number_y: {img_y_idx}.")
            for i in tqdm(range(0, trainer.T_STEPS)[::-1]):
                t = torch.full((1,), i, device=trainer.map_device, dtype=torch.long)
                img = trainer.sample_timestep(img, t)
                if i == 0:
                    ax[img_x_idx, img_y_idx].imshow(trainer.to_tensor_image(img.detach().cpu()))

    fig.savefig(os.path.join(out_path, "samples.png"))

# def save_samples(trainer, out_path):
#     num_images = 5
#     fig, ax = plt.subplots(5,5)
#
#     for img_x_idx in range(num_images):
#         for img_y_idx in range(num_images):
#             img_size = trainer.IMG_SIZE
#             img = torch.randn((1, 3, img_size, img_size), device=trainer.map_device)
#
#             # plt.subplot(1, num_images, img_x_idx + 1)
#             # ax[1, 1] = plt.imshow(trainer.to_tensor_image(img.detach().cpu()))
#             img_to_plot = trainer.to_tensor_image(img.detach().cpu())
#             ax[img_x_idx, img_y_idx].imshow(img_to_plot)
#
#     fig.savefig(os.path.join(out_path, "samples.png"))


trainer = Trainer(args.model_dir, args.T, args.START_B, args.END_B, args.IMG_SIZE, args.BATCH_SIZE)
trainer.load_checkpoint()
save_samples(trainer, "./output/")
print("done")