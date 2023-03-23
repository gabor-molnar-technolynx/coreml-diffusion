import torch
import matplotlib.pyplot as plt
from dataset import CelebDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from trainer import Trainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--restart_training", action="store_true", help="force training on CPU")
parser.add_argument("--train", action="store_true", help="force training on CPU")
parser.add_argument("--epochs", default=50, type=int, help="number of training epochs")
#TODO: save different checkpoint dirs and images sampled at that checkpoint!
parser.add_argument("--model_dir", default="model", help="Directory to save the checkpoints in.")
parser.add_argument("--T", default=300, type=int, help="Number of timesteps.")
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--BATCH_SIZE", default=2, type=int, help="Batch count for the training process.")
parser.add_argument("--START_B", default=0.0001, type=float, help="Beta at the first timestep.")
parser.add_argument("--END_B", default=0.02, type=float, help="Beta at the last timestep.")
parser.add_argument("--data_path", default="../../YoutubeProj/data/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img",
                    help="Path to the dataset folder.")
args = parser.parse_args()

def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        img = img[0]
        img = (img / 2.0) + 0.5
        img = torch.moveaxis(img, 0, -1)
        img = img.detach().cpu().numpy()

        plt.imshow(img)
    plt.show()
    print("done")

device = "cuda" if torch.cuda.is_available() else "cpu"

#TODO: move these inside the dataset class!
data_transforms = [
        transforms.Resize((args.IMG_SIZE, args.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
data_transform = transforms.Compose(data_transforms)
CelebDataset(args.data_path, transform=data_transform)

data = CelebDataset(args.data_path, transform=data_transform)
dataloader = DataLoader(data, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)

# Sample dataset:
# show_images(data, num_samples=20, cols=4)

# Simulate forward diffusion:
# image = next(iter(dataloader))[0]
# plt.figure(figsize=(15,15))
# plt.axis('off')
# num_images = 10
# stepsize = int(T/num_images)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     plt.subplot(1, num_images+1, int((idx/stepsize) + 1))
#     image, noise = trainer.forward_diffusion_sample(image, t)
#     show_tensor_image(image)
# plt.show()
# print("Done simulating forward diffusion.")

trainer = Trainer(args.model_dir, args.T, args.START_B, args.END_B, args.IMG_SIZE, args.BATCH_SIZE)
# last_checkpoint = trainer.find_last_checkpoint()

# trainer.clear_checkpoints()
# trainer.save_checkpoint()

if args.restart_training:
    trainer.clear_checkpoints()
else:
    trainer.load_checkpoint()

for epoch in tqdm(range(args.epochs)):
    for step, batch in  enumerate(dataloader):
        loss = trainer.training_step(batch[0])
        # trainer.log_history(loss)
        print(f"epoch {epoch}, step {step}, loss: {loss}")
        if epoch % 1 == 0 and step == 0:
            trainer.save_checkpoint()
            trainer.log_history(loss)
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")


