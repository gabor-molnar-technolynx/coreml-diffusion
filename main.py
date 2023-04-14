import torch
from dataset import CelebDataset
from torch.utils.data import DataLoader
import argparse
from trainer import Trainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--restart_training", action="store_true", help="force training on CPU")
parser.add_argument("--train", action="store_true", help="force training on CPU")
parser.add_argument("--epochs", default=50, type=int, help="number of training epochs")
parser.add_argument("--model_dir", default="model", help="Directory to save the checkpoints in.")
parser.add_argument("--T", default=300, type=int, help="Number of timesteps.")
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--BATCH_SIZE", default=2, type=int, help="Batch count for the training process.")
parser.add_argument("--START_B", default=0.0001, type=float, help="Beta at the first timestep.")
parser.add_argument("--END_B", default=0.02, type=float, help="Beta at the last timestep.")
parser.add_argument("--data_path", default="../../YoutubeProj/data/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img",
                    help="Path to the dataset folder.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

data = CelebDataset(args.data_path, args.IMG_SIZE)
dataloader = DataLoader(data, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)

trainer = Trainer(args.model_dir, args.T, args.START_B, args.END_B, args.IMG_SIZE, args.BATCH_SIZE)

trainer.save_checkpoint()

if args.restart_training:
    trainer.clear_checkpoints()
else:
    trainer.load_checkpoint()

for epoch in tqdm(range(args.epochs)):
    for step, batch in enumerate(dataloader):
        loss = trainer.training_step(batch[0])
        # trainer.log_history(loss)
        print(f"epoch {epoch}, step {step}, loss: {loss}")
        if epoch % 1 == 0 and step == 0:
            trainer.save_checkpoint()
            trainer.log_history(loss)
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")
