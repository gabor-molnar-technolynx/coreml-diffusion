import torch
from unet import SimpleUnet
import torch.nn.functional as F
from torch.optim import Adam
import os

class Trainer():
    def __init__(self, model_dir, t_steps,t_start,t_end, img_size, batch_size):
        self.model_dir = model_dir
        self.model = SimpleUnet()

        if not torch.cuda.is_initialized():
            self.map_device = torch.device("cpu")
        else:
            self.map_device = torch.device("cuda:0")

        self.model.to(self.map_device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        # Pre-calculate different terms for closed form
        self.betas = torch.linspace(0.0001, 0.02, t_steps)
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

    def save_checkpoint(self):
        torch.save(self.optimizer.state_dict(), os.path.join(self.model_dir, "opt.pt"))
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model.pt"))

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, "model.pt"), map_location=self.map_device))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.model_dir, "opt.pt"), map_location=self.map_device))

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