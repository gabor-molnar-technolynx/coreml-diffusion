import torch
from unet import SimpleUnet
import torch.nn.functional as F
from torch.optim import Adam
from shutil import rmtree
import os
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np

class Trainer():
    def __init__(self, model_dir, t_steps, t_start, t_end, img_size, batch_size):
        self.model_dir = model_dir
        self.model = SimpleUnet()

        if not torch.cuda.is_initialized():
            self.map_device = torch.device("cpu")
        else:
            self.map_device = torch.device("cuda:0")

        self.model.to(self.map_device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        # Pre-calculate different terms for closed form
        self.betas = torch.linspace(t_start, t_end, t_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.T_START = t_start
        self.T_END = t_end
        self.T_STEPS = t_steps


    @torch.no_grad()
    def save_samples(self, checkpoint_path):
            # Sample noise
            img_size = self.IMG_SIZE
            img = torch.randn((1, 3, img_size, img_size), device=self.map_device)
            num_images = 10
            fig, ax = plt.subplots(num_images)
            stepsize = int(self.T_STEPS / num_images)

            for i in range(0, self.T_STEPS)[::-1]:
                t = torch.full((1,), i, device=self.map_device, dtype=torch.long)
                img = self.sample_timestep(img, t)
                if i % stepsize == 0:
                    plt.subplot(1, num_images, int(i / stepsize) + 1)
                    ax[ int(i / stepsize)] = plt.imshow(self.to_tensor_image(img.detach().cpu()))
            # plt.show()
            fig.savefig(os.path.join(checkpoint_path, "test_samples.png"))


    def to_tensor_image(self, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :]

        return reverse_transforms(image)

    def log_history(self, loss):
        fname = os.path.join(self.model_dir, "log.npy")
        logs = None
        if os.path.isfile(fname):
            logs = np.load(fname)

        if logs is None:
            logs = [loss]
        else:
            # logs = np.concatenate([logs, loss], axis=0)
            logs = np.append(logs, loss)
        np.save(fname, logs)

        plot_fname = os.path.join(self.model_dir, "plot.png")

        plt.clf()
        plt.plot(logs)
        # plt.show()

        plt.savefig(plot_fname)


    def save_checkpoint(self):
        idx = self.find_last_checkpoint() + 1
        checkpoint_path = os.path.join(self.model_dir, str(idx))
        os.mkdir(checkpoint_path)
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "opt.pt"))
        torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
        print("Model checkpoint saved successfully.")
        #TODO: dont forget to reenable for training
        self.save_samples(checkpoint_path)

    def load_checkpoint(self):
        idx = self.find_last_checkpoint()
        if idx > 0:
            checkpoint_path = os.path.join(self.model_dir, str(idx))
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), map_location=self.map_device))
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "opt.pt"), map_location=self.map_device))
            print(f"Model checkpoint loaded successfully. Continuing from checkpoint {idx}")
        else:
            print("There was no checkpoint to load, using newly initialized model weights.")

    def find_last_checkpoint(self):
        checkpoints = os.listdir(self.model_dir)
        checkpoints = [int(x) for x in checkpoints if os.path.isdir(os.path.join(self.model_dir, x))]
        if len(checkpoints) > 0:
            checkpoints.sort()
            last_checkpoint = checkpoints[-1]
            return last_checkpoint
        else:
            return 0

    def clear_checkpoints(self):
        if os.path.isfile(os.path.join(self.model_dir, "log.npy")):
            os.remove(os.path.join(self.model_dir, "log.npy"))

        checkpoints = os.listdir(self.model_dir)
        checkpoints = [int(x) for x in checkpoints if os.path.isdir(os.path.join(self.model_dir, x))]
        for idx in checkpoints:
            checkpoint_path = os.path.join(self.model_dir, str(idx))
            if os.path.isdir(checkpoint_path):
                rmtree(checkpoint_path, ignore_errors=True)

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.map_device) * x_0.to(self.map_device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.map_device) * noise.to(self.map_device),\
               noise.to(self.map_device)

    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def sample_timestep(self, x, t):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def training_step(self, images):
        self.optimizer.zero_grad()
        t = torch.randint(0, self.T_STEPS, (self.BATCH_SIZE,), device=self.map_device).long()
        loss = self.get_loss(self.model, images, t)
        loss.backward()
        self.optimizer.step()
        return loss.item()