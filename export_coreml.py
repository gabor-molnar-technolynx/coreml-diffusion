import torch
import os
from unet import SimpleUnet
from torch.optim import Adam
import coremltools as ct


def load_checkpoint():
    if os.path.isfile(os.path.join("model", "model.pt")) and os.path.isfile(os.path.join("model", "opt.pt")):
        if not torch.cuda.is_initialized():
            map_device = torch.device("cpu")
        else:
            map_device = torch.device("cuda:0")

        model.load_state_dict(torch.load(os.path.join("model", "model.pt"), map_location=map_device))
        optimizer.load_state_dict(torch.load(os.path.join("model", "opt.pt"), map_location=map_device))



model = SimpleUnet()
optimizer = Adam(model.parameters(), lr=0.001)
load_checkpoint()


dummy_img = torch.rand((2,3,64,64)).float()
dummy_timestep = torch.randint(0, 100, (2,)).long()
model_ts = torch.jit.trace(model, (dummy_img, dummy_timestep))
model_ct = ct.convert(model_ts,
                              inputs=[ct.TensorType(name="img_input", shape=dummy_img.shape),
                                      ct.TensorType(name="timestep_input", shape=dummy_timestep.shape)],
                              outputs=[
                                  ct.TensorType(name="noise_prediction")])

