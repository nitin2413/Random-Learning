import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

train_data = datasets.CIFAR10(root="./data",download=True,transform=transform,train=True)
train_load = DataLoader(train_data,shuffle=True,batch_size=128)

# ngf = 64  generator feature maps

class Generator(nn.Module):
    def __init__(self,latent_dim=100,ngf = 64):
        super(Generator , self).__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,ngf*8,4,2,1,bias=False),# bias false as batchnorm calculate them
            nn.BatchNorm2d(ngf*8),# act as stabilizer as gan is sensitive to exploding and vanishing gradiant
            nn.ReLU(True),# relu ture mean, no storage of input tensor, backpropogation get handel by pytorch,
                          # as in relu <= 0 is 0 and > 0 is 1 so all the 1 are mask when needed in backpro

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4), # also use for normalization ngf*4 tell how many feature are there to normalize
            nn.ReLU(True),# does not accept negative value cause dead neurons

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self,x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self,latent_dim = 100 , ndf = 64 ):
        super(Discriminator,self).__init__()

        self. discriminator = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),# prevent dead neuron by having negative value

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 2, 1, bias=False),# probability (0/1) 0-> fake 1-> real
            nn.Sigmoid()# out_feature 1 --> probability (0/1) 0-> fake 1-> real
        )

    def forward(self,x):
        return self.discriminator(x)

latent_dim = 100
natG = Generator()
natD = Discriminator()

loss_function = nn.BCELoss()

# betas=(β1, β2) are the exponential decay rates control momentum default value -> 0.9 but for gan it is to high
# lr is 0.0002 as G and D compete with each other -> training is unstable
optimizerG = optim.Adam(natG.parameters(), lr = 0.0002, betas=(0.5,0.999))
optimizerD = optim.Adam(natD.parameters(), lr=0.0002, betas=(0.5,0.999))

# use fixed noice to see improvement as the training progress,
# if we use random noice after every epoch then  generator’s output would always look different, even if it got better.
fixed_noise = torch.randn(64,latent_dim,1,1,device=device) # latent_dim = 100

epochs = 25
for epoch in range(epochs):
    for i , (real , _) in enumerate(train_load):
        b_size = real.size(0)# gives the size of the first dimension of the tensor.
        real = real.to(device)
        real_label = torch.ones(b_size,device=device)
        fake_label = torch.zeros(b_size,device=device)

        natD.zero_grad()
        output_real = natD(real)
        loss_D_real = loss_function(output_real,real_label)

        noise = torch.randn(b_size,latent_dim,1,1,device=device)
        fake = natG(noise)
        output_fake = natD(fake.detach())# .detach(): prevents gradients from flowing into generator while training discriminator.
        loss_D_fake = loss_function(output_fake,fake_label)

        lossD = loss_D_fake + loss_D_real
        lossD.backward()
        optimizerD.step()

        natG.zero_grad()
        output_fake = natD(fake)
        lossG = loss_function(output_fake,real_label ) # want fake to be real
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"... Loss_D: {lossD.item():.4f}, Loss_G: {lossG.item():.4f}")

     # Generate new fake images
    with torch.no_grad():
        # noise = torch.randn(64, latent_dim, 1, 1, device=device)  # fresh random noise not needed
        fake_images = natG(fixed_noise).detach().cpu()

     # Create a grid
    grid = vutils.make_grid(fake_images, nrow=8, normalize=True)

     # Plot
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()



'''       
import torch.nn as nn
import torch.optim as optim
from torchvision import pothole, transforms, utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------
# ✅ Setup
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
ngf = 64
ndf = 64
epochs = 70
batch_size = 128
lr = 0.0002  # slightly higher works well with stable training

# Create folder for results
os.makedirs("results", exist_ok=True)

# ------------------------------------------
# ✅ Data preprocessing
# ------------------------------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize between [-1, 1]
])

train_data = pothole.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

# ------------------------------------------
# 🧠 Generator
# ------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input Z -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Output (3) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# ------------------------------------------
# 🕵️‍♂️ Discriminator
# ------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input (3) x 32 x 32 -> (ndf) x 16 x 16
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 16 x 16 -> (ndf*2) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 8 x 8 -> (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


# ------------------------------------------
# ⚙️ Initialize Models + Optimizers
# ------------------------------------------
netG = Generator(latent_dim, ngf).to(device)
netD = Discriminator(ndf).to(device)

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# ------------------------------------------
# 🚀 Training Loop
# ------------------------------------------
for epoch in range(epochs):
    for i, (real, _) in enumerate(train_loader):
        b_size = real.size(0)
        real = real.to(device)

        # Labels
        real_label = torch.full((b_size,), 1., dtype=torch.float, device=device)
        fake_label = torch.full((b_size,), 0., dtype=torch.float, device=device)

        # --- Train Discriminator ---
        netD.zero_grad()
        output_real = netD(real)
        lossD_real = criterion(output_real, real_label)

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        lossD_fake = criterion(output_fake, fake_label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # --- Train Generator ---
        netG.zero_grad()
        output = netD(fake)
        lossG = criterion(output, real_label)
        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"[{epoch+1}/{epochs}] [{i}/{len(train_loader)}] "
                  f"Loss_D: {lossD.item():.4f} | Loss_G: {lossG.item():.4f}")

    # --- Save progress images after each epoch ---
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(f"results/epoch_{epoch+1:03d}.png")
    plt.close()

# Save final model
torch.save(netG.state_dict(), "generator_final.pth")
torch.save(netD.state_dict(), "discriminator_final.pth")

print("✅ Training complete! Images saved in ./results/")
        '''