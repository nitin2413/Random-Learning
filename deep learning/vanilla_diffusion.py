# We add Gaussian noise to an image,
# and the model learns the direction in which noise was added (the score / noise vector).
# During generation, we follow this direction in reverse, step by step, to remove noise.
# We add noise to an image,and the model predicts the flow at which noise was added, and then reduces it every iteration.
# https://chatgpt.com/c/697216df-dc48-8320-bbf3-91ab1df51d65 -> questioning
# https://chatgpt.com/c/696f7dcb-73f4-8322-91c1-bad62549155e -> code and theory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
ds = torchvision.datasets.MNIST(root = "./data", train = True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True)

def linear_beta_schedular(time_step):
    beta_start = 1e-4
    beta_end =  0.02
    return torch.linspace(beta_start,beta_end,time_step)

T = 1000
# β (beta) How much new noise you add at step t
betas = linear_beta_schedular(T)
# α (alpha)How much of the image you keep at step t
alphas = 1. - betas
#ᾱ tells you how much is left after t steps.
alpha_bars = torch.cumprod(alphas,dim=0)

def q_sample(x0,t,noise=None):
    if noise is None:
        noise = torch.randn_like(x0) # hold the value of noise added at time t

    sqrt_alpha_bars = torch.sqrt(alpha_bars[t])[:,None,None,None] # adjusting dimension as x0(batch_size,x,y,z)
    sqrt_one_minus = torch.sqrt(1-alpha_bars[t])[:,None,None,None] # sqrt one dimensional for multi same dim
    return x0 * sqrt_alpha_bars + sqrt_one_minus * noise # xt
    # x0 is still partially present therefor,structure of x0 is still present in xt
    # (Edges, strokes, digit shapes are still partially visible.)

# Same image looks different at different noise levels, time embedding tells the model which level it is
# Without t, the model cannot know how aggressive it should be.
class Time_Embedding(nn.Module):
    def __init__(self,T,dim):
        super().__init__()
        self.embed = nn.Embedding(T,dim)

    def forward(self,t):
        return self.embed(t)
    
class Conv_Diffusion(nn.Module):
    def __init__(self,T):
        super().__init__()

        #Time Embedding Layers
        self.time_embed = Time_Embedding(T,128)
        self.time_prob = nn.Linear(128,64) # maps 128 → 64 to match CNN channels

        # conv layers
        self.conv1 = nn.Conv2d(1 , 64 , kernel_size=3 , padding=1) # gray scale in_channele = 1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1) # same here

    def forward(self,x,t):
        """
                x: (batch, 1, H, W)   noisy image
                t: (batch,)          timestep
        """
        # image features (h)
        # time information (t_emb)

        # time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_prob(t_emb)#[:,:,None,None] or
        t_emb = t_emb[:,:,None,None] # t: (batch, 64 ,1 ,1)

        # forward pass
        # Each channel(conv) answers:“ How much does this pattern exist at each pixel?”
        h = F.relu(self.conv1(x))# extract important feature store them , relu suppress weak / negative responses
        h = h + t_emb # are combined to decide:at Given time step how noisy the image is, what noise should be removed?
        h = F.relu(self.conv2(h))# h is a learned representation of local structures (edges, blobs, textures) present in the noisy image at timestep t.

        return self.conv3(h) # predict output

model = Conv_Diffusion(T)
optimizer = optim.Adam(model.parameters(),lr=1e-3)


def diffusion_loss(model , x0):
    batch_size = x0.size(0)
    t = torch.randint(0,T,(batch_size,))
    noise = torch.randn_like(x0)

    xt = q_sample(x0,t,noise)
    noise_pred = model(xt , t)

    return F.mse_loss(noise_pred,noise)

for epoch in range(10):
    for x0 , _ in dataloader:
        loss = diffusion_loss(model , x0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# up until know we were doing the forward precess, form here we will generate img
@torch.no_grad()
# reverse process mean formula
def p_sample(model, x ,t):
    beta = betas[t]
    alpha = alphas[t]
    alpha_bar = alpha_bars[t]

    noise_pred = model(x , torch.trunc([t]))

    mean = (1 / torch.sqrt(alpha)) * (
        x - beta / torch.sqrt(1 - alpha_bar) * noise_pred
    )

    if t > 0:
        noise = torch.randn_like(x)
    else:
        noise = torch.zeros_like(x)

    return mean + torch.sqrt(beta) * noise

@torch.no_grad()
def sample(model, img_size=(1, 28, 28)):
    x = torch.randn(1, *img_size)

    for t in reversed(range(T)):
        x = p_sample(model, x, t)

    return x