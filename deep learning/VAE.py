# Main idea of building VAEs is using probability distributions.
# A distribution represents a range of possible values for z.
# Instead of giving an input to a fixed latent vector z, we encode it as a distribution.
# This allows us to sample different z’s for the same input, creating a continuous latent space.
# For example, the latent representations of digits 1 and 7 form a smooth path through intermediate points like 4.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import  torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
transform1 = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root="./data",train= True,transform=transform ,download=True)
train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)

train_dataset_cvae = datasets.CIFAR10(root="./data",train= True,transform=transform1 ,download=True)
train_loader_cvae = DataLoader(train_dataset_cvae,batch_size=128,shuffle=True)

class VAE(nn.Module):
    def __init__(self,latent_dim=128):
        super(VAE,self).__init__()

        self.latent_dim = latent_dim # vector example :- z = [0.1,0.5]
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),# 3 -> rgb
            nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),
            nn.Flatten() # currently 4D convert 2D as linear expect input (B,feature)
        )

        self.fc_mu = nn.Linear(128*4*4,latent_dim)# mean , 128*4*4 after flatten
        self.fc_logvar = nn.Linear(128*4*4,latent_dim)# log variance should not be -ve

        self.decoder_input  = nn.Linear(latent_dim,128*4*4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d (32,3,4,2,1),
            nn.Tanh()# output -1,1
        )

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5 * logvar) # using exp we ensure logvar is not -ve
        eps = torch.randn_like(std) # sample random noise from N(0, 1)
        # z = mu + std * eps
        return mu + (std * eps)

    def forward(self,x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu,logvar)

        out = self.decoder_input(z) # give dimension (B,128*4*4) or (B,2048)
        out = out.view(-1, 128 , 4 , 4 ) # .view reshape number into 3D tensor -1 torch automatically provide batch_size
        x_hat = self.decoder(out)
        return x_hat , mu , logvar

    # loss in VAE's are of two type
    # if we go in deep math of VAE's their are 2 distribution q and p
    # q is posterior and p is prior , posterior was made as calculating prior is very hard
    # 1- Reconstruction Loss (MSE) loss face while rebuilding the input
    # 2- KL Divergence how close our approximate posterior is to the prior.

def vae_loss(x,x_hat,mu,logvar):
    # Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # loss = recon_loss + kl_loss
    return recon_loss + kl_loss

def generate_img(model,num_sample=8):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_sample,model.latent_dim).to(device)
        out = model.decoder_input(z)
        out = out.view(-1,128,4,4)
        sample = model.decoder(out).cpu()

        fig, axes = plt.subplots(1, num_sample, figsize=(15, 2))
        for i in range(num_sample):
            img = (sample[i].permute(1, 2, 0).numpy() + 1) / 2  # denormalize to [0,1]
            axes[i].imshow(img)
            axes[i].axis("off")
        plt.show()


class CVAE(nn.Module):
    def __init__(self,latent_dim=128,num_classes=10):
        super(CVAE,self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.enc_conv = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Label embedding (one-hot → feature vector) eg cat =[0,1,0,0,0,0,0,0,0,0]
        # 50 is to add more vector to learn relationship between class
        # dog and cat near in the cloud and cat and car far
        self.label_emb = nn.Linear(num_classes,50)

        self.fc_mu = nn.Linear(128*4*4+50,latent_dim) # + 50 to add label in it
        self.fc_logvar = nn.Linear(128*4*4+50,latent_dim)

        self.decoder_input = nn.Linear(latent_dim+50,128*4*4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),
            nn.Sigmoid()
        )

        # this function is to pass label with the encoder (selective learning)
    def encoder(self,x,y):
        h = self.enc_conv(x) # input x is pass in conv2d
        y_emb = self.label_emb(y) # perform one hot encoding
        h = torch.cat([h,y_emb],dim=1)# concatenating h->value after x pass in conv2d and y-> which class we are traning in on
        mu , logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu , logvar

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + (std*eps)

    def decoder(self,z,y):
        y_emb = self.label_emb(y)
        z = torch.cat([z,y_emb],dim =1)
        h = self.decoder_input(z) # here the input is 2D ((latent_dim,128*4*4+50))
        h = h.view(h.size(0),128,4,4)# decoder expect 4d tensor input ((batch_size, 128, 4, 4)) so we convert
        return self.dec_conv(h)

    def forward(self,x,y):
        mu,logvar = self.encoder(x,y)
        z = self.reparameterize(mu,logvar)
        x_hat = self.decoder(z,y)
        return x_hat, mu ,logvar

def cvae_loss(x, x_hat, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)

def generate_img_cvae(model,num_sample=8):
    model1.eval()
    with torch.no_grad():
        y = torch.tensor([5] * 8).to(device)  # Class 5 = "dog" Generate 16 images of class = dog.Without this, the decoder won’t know what category to generate
        y = F.one_hot(y, num_classes=10).float() # convertion of y generated above(y = [5, 5, 5, 5, 5, 5, 5, 5]) to onehot encoding
        z = torch.randn(num_sample,model1.latent_dim).to(device)# select random z vector value for variation
        sample = model1.decoder(z,y).cpu() # combine random z and label to generate image

        fig, axes = plt.subplots(1, num_sample, figsize=(15, 2))
        for i in range(num_sample):
            img = (sample[i].permute(1, 2, 0).numpy() + 1) / 2  # denormalize to [0,1]
            axes[i].imshow(img)
            axes[i].axis("off")
        plt.show()


device = "mps" if torch.mps.is_available() else "cpu"
model = VAE(latent_dim=128).to(device)
model1 = CVAE(latent_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

epochs = 10
for epoch in range(epochs):
    model1.train()
    train_loss = 0
    for x, y in train_loader_cvae:
        x, y = x.to(device), F.one_hot(y, num_classes=10).float().to(device)

        x_hat,mu,logvar = model1(x,y)
        loss = cvae_loss(x,x_hat,mu,logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f"Epoch {epoch+1}, Train_Loss: {train_loss/len(train_loader_cvae)}")

generate_img_cvae(model1,num_sample=8)


for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch,(x,_) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()

        x_hat,mu,logvar = model(x)
        loss = vae_loss(x,x_hat,mu,logvar)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss/len(train_loader.dataset):.4f}")

generate_img(model,num_sample=8)



# import torch
# import torch.nn as nn
# from torchvision import pothole, transforms
# from torch.utils.data import DataLoader
# import  torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# transform1 = transforms.Compose([transforms.ToTensor()])
# train_dataset_cvae = pothole.CIFAR10(root="./data",train= True,transform=transform1 ,download=True)
# train_loader_cvae = DataLoader(train_dataset_cvae,batch_size=128,shuffle=True)
#
# class CVAE(nn.Module):
#     def __init__(self,latent_dim=256,num_classes=10):
#         super(CVAE,self).__init__()
#
#         self.latent_dim = latent_dim
#         self.num_classes = num_classes
#
#         self.enc_conv = nn.Sequential(
#             nn.Conv2d(3,32,4,2,1),
#             nn.ReLU(),
#             nn.Conv2d(32,64,4,2,1),
#             nn.ReLU(),
#             nn.Conv2d(64,128,4,2,1),
#             nn.ReLU(),
#             nn.Conv2d(128,256,4,2,1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         # Label embedding (one-hot → feature vector) eg cat =[0,1,0,0,0,0,0,0,0,0]
#         # 50 is to add more vector to learn relationship between class
#         # dog and cat near in the cloud and cat and car far
#         self.label_emb = nn.Linear(num_classes,50)
#         ## Encoder output size = 256 * 2 * 2 = 1024
#         self.fc_mu = nn.Linear(1024+50,latent_dim) # + 50 to add label in it
#         self.fc_logvar = nn.Linear(1024+50,latent_dim)
#
#         self.decoder_input = nn.Linear(latent_dim+50,1024)
#         self.dec_conv = nn.Sequential(
#             nn.ConvTranspose2d(256,128,4,2,1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128,64,4,2,1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64,32,4,2,1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32,3,4,2,1),
#             nn.Sigmoid()
#         )
#
#         # this function is to pass label with the encode (selective learning)
#     def encoder(self,x,y):
#         y = F.one_hot(y, num_classes=self.num_classes).float()
#         h = self.enc_conv(x) # input x is pass in conv2d
#         y_emb = self.label_emb(y) # perform one hot encoding
#         h = torch.cat([h,y_emb],dim=1)# concatenating h->value after x pass in conv2d and y-> which class we are traning in on
#         mu , logvar = self.fc_mu(h), self.fc_logvar(h)
#         return mu , logvar
#
#     def reparameterize(self,mu,logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + (std*eps)
#
#     def decoder(self,z,y):
#         y = F.one_hot(y, num_classes=self.num_classes).float()
#         y_emb = self.label_emb(y)
#         z = torch.cat([z,y_emb],dim =1)
#         h = self.decoder_input(z) # here the input is 2D ((latent_dim,128*4*4+50))
#         h = h.view(h.size(0),256,2,2)# decoder expect 4d tensor input ((batch_size, 128, 4, 4)) so we convert
#         return self.dec_conv(h)
#
#     def forward(self,x,y):
#         mu,logvar = self.encoder(x,y)
#         z = self.reparameterize(mu,logvar)
#         x_hat = self.decoder(z,y)
#         return x_hat, mu ,logvar
#
# def cvae_loss(x, x_hat, mu, logvar, beta=0.1):
#     recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return (recon_loss + beta * kl_loss) / x.size(0)
#
# def generate_img_cvae(model,num_sample=8):
#     model1.eval()
#     with torch.no_grad():
#         y = torch.tensor([5] * num_sample,dtype=torch.long,device = device)  # Class 5 = "dog" Generate 16 images of class = dog.Without this, the decoder won’t know what category to generate
#         # convertion of y generated above(y = [5, 5, 5, 5, 5, 5, 5, 5]) to onehot encoding
#         z = torch.randn(num_sample,model1.latent_dim).to(device)# select random z vector value for variation
#         sample = model1.decoder(z,y).cpu() # combine random z and label to generate image
#
#         fig, axes = plt.subplots(1, num_sample, figsize=(15, 2))
#         for i in range(num_sample):
#             img = (sample[i].permute(1, 2, 0).numpy()) # denormalize to [0,1]
#             axes[i].imshow(img)
#             axes[i].axis("off")
#         plt.show()
#
#
# device = "mps" if torch.mps.is_available() else "cpu"
# model1 = CVAE(latent_dim=256).to(device)
# optimizer = optim.Adam(model1.parameters(), lr = 1e-3)
#
# epochs = 50
# for epoch in range(epochs):
#     model1.train()
#     train_loss = 0
#     for x, y in train_loader_cvae:
#         x, y = x.to(device), y.to(device)
#
#         x_hat,mu,logvar = model1(x,y)
#         loss = cvae_loss(x,x_hat,mu,logvar)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#     print(f"Epoch {epoch+1}, Train_Loss: {train_loss/len(train_loader_cvae)}")
#
# generate_img_cvae(model1,num_sample=8)