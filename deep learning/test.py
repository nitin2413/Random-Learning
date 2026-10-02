import math

# UNet preserves global structure via downsampling and skip connections while embeddings guide what to denoise

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MaxPool2d
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
# ])
ds = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
)
dataloader = DataLoader(ds, shuffle=True, batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"


def linear_beta_schedular(Time_Step):
    start_betas = 1e-3
    end_betas = 0.02
    return torch.linspace(start_betas, end_betas, Time_Step)


T = 1000
betas = linear_beta_schedular(T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def noise_add(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bars = torch.sqrt(alpha_bars[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
    return x0 * sqrt_alpha_bars + noise * sqrt_one_minus


class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.GroupNorm(8, out_channel),  # Diffusion needs self-consistency, not batch consistency(BatchNorm).
            nn.SiLU(),  # x * sigmoid
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.GroupNorm(8, out_channel),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(  # Convert raw timestep info into a feature-wise instruction vector.
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)  # Produces a vector aligned with feature channels as dim differs
        )

    def forward(self, t):
        return self.mlp(t)  # (B, dim)


def sinusoidal_embedding(t, time_dim):
    device = t.device
    half = time_dim // 2
    freqs = torch.exp(
        - math.log(10000) * torch.arange(0, half, device=device) / half
    )
    emb = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class UNet_Diffusion(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_dim = time_dim

        # t tell us how much noise is present in image at particular time step
        # Time tells the network how much denoising to do.
        # Features tell the network what to denoise.
        # Interpret spatial features under the context of this noise level.
        self.time_emb = TimeEmbedding(time_dim)

        # Downsampling decreases image size and extracts features.
        # Upsampling enlarges the image back to original size.
        # nn.Upsample(scale_factor=2) resizes the image,
        # and details are added using convolution blocks (up1, up2) + skip connections

        self.down1 = Block(3, 64)  # skin texture , hair
        self.down2 = Block(64, 128)  # eyes , nose , mouth
        self.max_pool = MaxPool2d(2)

        self.mid = Block(128, 128)  # Bottleneck

        self.up1 = Block(256, 128)
        self.up2 = Block(192, 64)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="nearest")  # as pooling is done reduce by the factor of 2 scale_factor increase by 2
        # At each pixel, ask 64 “experts” for opinions,
        # then combine them into one final answer.
        # kernel 1 -> to focus on individual pixel , Mixes information across channels only
        # why not kernel = 3 :- All spatial reasoning already happened earlier, Final step is just:“Map features → noise value
        self.final = nn.Conv2d(64, 3, 1)  # out_channel -> 1 for gray scale 3 for rgb

    def forward(self, x, t):
        t = sinusoidal_embedding(t, self.time_dim)
        t = self.time_emb(t)[:, :, None, None]

        d1 = self.down1(x)  # (B, 64, 64, 64)
        d2 = self.down2(self.max_pool(d1))  # (B, 128, 32, 32)

        mid = self.mid(d2 + t[:, :128])  # (B, 128, 32, 32)

        # ❗ concatenate FIRST (same resolution)
        u2 = self.up1(torch.cat([mid, d2], dim=1))  # (B, 128, 32, 32)
        u2 = self.upsample(u2)  # (B, 128, 64, 64)

        u1 = self.up2(torch.cat([u2, d1], dim=1))  # (B, 64, 64, 64)

        return self.final(u1)


model = UNet_Diffusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)


def Diffusion_loss(model, x0):
    batch_size = x0.size(0)
    t = torch.randint(0, T, (batch_size,), device=device)  # For each image in the batch: Pick a random timestep
    noise = torch.randn_like(x0)

    xt = noise_add(x0, t, noise)  # see noise_add function once
    noise_pred = model(xt, t)
    return F.mse_loss(noise_pred, noise)


for epoch in range(5):
    for x0, _ in dataloader:
        x0 = x0.to(device)
        loss = Diffusion_loss(model, x0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"for {epoch} : loss is {loss.item():4f}")


@torch.no_grad()
def generate_img_ddpm(model, batch_size=8, img_size=64):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for t in reversed(range(T)):
        beta = betas[t].to(device)
        alpha = alphas[t].to(device)
        alpha_bar = alpha_bars[t].to(device)

        t_batch = torch.full((batch_size,), t,
                             device=device)  # It tells the model “all images in this batch are currently at the same noise level t.”
        noise_pred = model(x, t_batch)

        # remove noise predicted by model mean (reverse diffusion)
        x = (1 / torch.sqrt(alpha)) * (
                x - beta / torch.sqrt(1 - alpha_bar) * noise_pred
        )
        # Add noise except at last step
        if t > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(beta) * noise  # add noice for little randomness Keeps generation diverse Prevents collapse

    return x.clamp(-1, 1)  # forces every pixel value to stay within a fixed range. as process of (reverse diffusion)
    # can be unstable output Pixel values can explode beyond valid ranges act as safety check
    # Errors compound across 1000 steps
    # One bad step → snowballs into artifacts


samples = generate_img_ddpm(model, batch_size=4)

import matplotlib.pyplot as plt

for i in range(4):
    img = samples[i].clamp(0, 1).permute(1, 2, 0).cpu()
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# sinusoidal embedding
# currently our model have memories how much noice is present at each time step but does't know time flow like t = 17 is before 18
# as it is hard for model to understand time and also will take lots of resource to learned by neural network there for we use sinusoidal embedding
# Sinusoidal embedding turns one number t into a vector of waves:
# sin(t / 10000^(2i/d)) , cos(t / 10000^(2i/d))
# Low-frequency waves → capture slow changes (global stage of diffusion) ,
# High-frequency waves → capture fine changes (local step-to-step changes)
# No learning required as it use pure math
# Unlike learned embeddings:
# Sinusoidal embeddings generalize
# Work even for unseen timesteps
# formula and understanding (https://chatgpt.com/c/69722312-b310-8323-81e8-e21d6fe5401a)

# def sinusoidal_embedding(t , time_dim):
#     device = t.device
#     half = time_dim // 2
#     freqs = torch.exp(
#         - math.log(10000) * torch.arange(0 , half , device = device) / half
#     )
#     emb = t[:,None] * freqs[None,:]
#     emb = torch.cat([torch.sin(emb),torch.cos(emb)],dim=1)
#     return emb

# ddpm and ddim are two different sampling method not MODEL

def ddim_time_step(T, ddim_steps):
    return torch.linspace(0, T - 1, ddim_steps).long()  # skip time_step (1000 , 900 , 800 ..) (the momentum part).


@torch.no_grad()
# eta is deterministic factor 0.0 no random noise at all 0.1 some amount of random ness
def genrate_image_ddip(model, batch_size=8, img_size=64, ddim_step=50, eta=0.0):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    time_step = ddim_time_step(T, ddim_step)
    # ddim start from 1 as we calculate t_prev and for 0 will not have t_prev
    # need for t_prev as we skip time_step (900,800) t_prev might be far away ,We must explicitly know where to land
    for i in reversed(range(1, len(time_step))):
        t = time_step[i]
        t_prev = time_step[i - 1]
        # It tells the model “all images in this batch are currently at the same noise level t.”
        # at time t their can be multiple image
        t_batch = torch.full((batch_size,), t, device=device)
        noise_pred = model(x, t_batch)

        alpha_bar_t = alpha_bars[t].to(device=device)
        alpha_bar_t_prev = alpha_bars[t_prev].to(device=device)
        # Recover predicted clean image x0 by inverting the forward (math formula) diffusion equation
        x0_pred = (x - torch.sqrt(alpha_bar_t) * noise_pred) / (torch.sqrt(alpha_bar_t))
        # This rebuilds the image at the previous timestep by mixing the predicted clean image with the right amount of noise.
        x = (
                torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(alpha_bar_t_prev) * noise_pred
        )

    return x.clamp(-1, 1)


samples = genrate_image_ddip(model, batch_size=4, ddim_step=50)