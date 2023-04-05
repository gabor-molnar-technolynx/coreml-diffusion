import torch
from trainer import Trainer
import coremltools as ct
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="model", help="Directory to save the checkpoints in.")
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--BATCH_SIZE", default=2, type=int, help="Batch count for the training process.")
parser.add_argument("--START_B", default=0.0001, type=float, help="Beta at the first timestep.")
parser.add_argument("--END_B", default=0.02, type=float, help="Beta at the last timestep.")
args = parser.parse_args()

trainer = Trainer(args.model_dir, 0, args.START_B, args.END_B, args.IMG_SIZE, args.BATCH_SIZE)
trainer.load_checkpoint()
model = trainer.model
model.eval()

# Create dummy inputs for tracing the model.
dummy_img = torch.rand((1, 3, args.IMG_SIZE, args.IMG_SIZE)).float()
dummy_timestep = torch.randint(0, 100, (1,)).long()

# Trace the model.
model_ts = torch.jit.trace(model, (dummy_img, dummy_timestep))
model_ct = ct.convert(model_ts,
                      inputs=[ct.TensorType(name="img_input", shape=dummy_img.shape),
                              ct.TensorType(name="timestep_input", shape=dummy_timestep.shape)],
                      outputs=[
                          ct.TensorType(name="noise_prediction")])

mlmodel_path = os.path.join(args.model_dir, "model.mlmodel")
model_ct.save(mlmodel_path)
model_ct = ct.models.MLModel(mlmodel_path)
