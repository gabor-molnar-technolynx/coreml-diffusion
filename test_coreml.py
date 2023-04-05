import os
import argparse
import torch
import coremltools as ct
from matplotlib import pyplot as plt
import numpy as np
from trainer import Trainer

torch.set_grad_enabled(False)
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="model", help="Directory of the mlmodel file.")
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--BATCH_SIZE", default=2, type=int, help="Batch count for the training process.")
parser.add_argument("--START_B", default=0.0001, type=float, help="Beta at the first timestep.")
parser.add_argument("--END_B", default=0.02, type=float, help="Beta at the last timestep.")
parser.add_argument("--T", default=300, type=int, help="Number of timesteps.")
args = parser.parse_args()


def save_samples():
    # Sample noise
    img_size = args.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size))
    num_images = 10
    fig, ax = plt.subplots(num_images)
    stepsize = int(args.T / num_images)

    for i in range(0, args.T)[::-1]:
        t = torch.full((1,), i, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            ax[int(i / stepsize)] = plt.imshow(trainer.to_tensor_image(img.detach().cpu()))
    fig.savefig( "test_samples.png")

def sample_timestep(x, t):
    betas_t = trainer.get_index_from_list(trainer.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = trainer.get_index_from_list(
        trainer.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = trainer.get_index_from_list(trainer.sqrt_recip_alphas, t, x.shape)

    # Convert inputs to be compatible with mlmodel.
    x_np = x.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy().astype(np.float32)
    model_output = unet_ct.predict({"img_input": x_np, "timestep_input": t_np})
    img_out = model_output["noise_prediction"]

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * img_out / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = trainer.get_index_from_list(trainer.posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

trainer = Trainer(args.model_dir, 0, args.START_B, args.END_B, args.IMG_SIZE, args.BATCH_SIZE)
print("loading model...")
unet_ct = ct.models.MLModel(os.path.join(args.model_dir, "model.mlmodel"), compute_units=ct.ComputeUnit.CPU_ONLY)
save_samples()
