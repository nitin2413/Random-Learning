'''
Autoencoders are self-supervised neural networks that learn to compress data into a smaller, meaningful representation (latent space) and then reconstruct it.
The encoder maps the input 𝑥 to a compressed latent vector 𝑧, and the decoder reconstruct 𝑥^ from that latent vector.
They are trained using a reconstruction loss such as MSE or BCE, which measures how close the reconstructed output is to the original input.
The latent space captures the important features and patterns in the data, which can be used for dimensionality reduction, denoising, anomaly detection, or as features for other models.
'''



import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


device = torch.device("mps" if torch.mps.is_available() else "cpu")

# added by gpt to solve size mismatch in BSELoss
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

ds = torchvision.datasets.MNIST(root = "./data", train = True,download=True, transform=transform)

train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size

torch.manual_seed(42)

train_ds , val_ds = random_split(ds , [train_size , val_size], )

train_loader = DataLoader(train_ds,batch_size=64,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=64,shuffle=False)
#
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
# #nn.Sequential ,It’s a container in PyTorch that lets you stack layers one after another in the order they’re written
#         self.encoder = nn.Sequential(
#             nn.Linear(784 , 128),
#             nn.ReLU(),
#             nn.Linear(128 , 64 ),
#             nn.ReLU(),
#             nn.Linear(64,32)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(32,64),
#             nn.ReLU(),
#             nn.Linear (64,128),
#             nn.ReLU(),
#             nn.Linear(128,784),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # stride shrink dimension also define how much pixel does kernel move,
            # padding ensure kernel does not shrink to much ,to fit in
            # in channels is 1 because mnist is grey scale for rgb it is 3
            nn.Conv2d(in_channels=1,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            #output_padding allows you to fine-tune the output size so it exactly matches the target dimensions.
            # ConvTranspose2d is opposite of conv2d
            nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=1,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)  # latent representation
        reconstructed = self.decoder(latent)  # reconstructed output
        return reconstructed

model = Autoencoder().to(device)
loss_function  = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for batch in train_loader:
        images,_ = batch
        images = images.to(device)
        reconstructed = model(images) # model is returning reconstructed from forward_pass
        loss = loss_function(reconstructed , images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() # item convert tensor to python int / float
    print(f"epoch:[{epoch}/{epochs}] || loss : {running_loss/len(train_loader):.4f}")

import matplotlib.pyplot as plt

sample = next(iter(val_loader))[0].to(device)
recon = model(sample)
plt.subplot(1,2,1)
plt.imshow(sample[0][0].cpu(), cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(recon[0][0].detach().cpu(), cmap='gray')
plt.title("Reconstructed")
plt.show()
