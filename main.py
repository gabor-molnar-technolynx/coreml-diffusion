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
parser.add_argument("--export_model", action="store_true", help="force training on CPU")
parser.add_argument("--restart_training", action="store_true", help="force training on CPU")
parser.add_argument("--train", action="store_true", help="force training on CPU")
args = parser.parse_args()


def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        img = img[0]
        img = torch.moveaxis(img, 0, -1)
        img = img.detach().cpu().numpy()

        plt.imshow(img)
    plt.show()
    print("done")

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

@torch.no_grad()
def sample_plot_image(trainer, device):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = trainer.sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()


data_path = "./data/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img"
T = 300
IMG_SIZE = 64
BATCH_SIZE = 2
START=0.0001
END=0.02
device = "cuda" if torch.cuda.is_available() else "cpu"

data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
data_transform = transforms.Compose(data_transforms)
CelebDataset(data_path, transform=data_transform)

data = CelebDataset(data_path, transform=data_transform)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# model = SimpleUnet()
trainer = Trainer("model", T, START, END, IMG_SIZE, BATCH_SIZE)



epochs = 100 # Try more!

if args.restart_training:
    trainer.save_checkpoint()
else:
    trainer.load_checkpoint()

for epoch in tqdm(range(epochs)):
    for step, batch in  enumerate(dataloader):
        loss = trainer.training_step(batch[0])
        print(f"epoch {epoch}, step {step}, loss: {loss}")
        if epoch % 5 == 0 and step == 0:
            trainer.save_checkpoint()
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")
            sample_plot_image(trainer, device)


