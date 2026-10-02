import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,8,3,1,1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8,16,3,1,1)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(16*7*7,10)


    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dense(x)
        return x

model = CNNModel()
model.load_state_dict(torch.load("/Users/nitinsingh/PycharmProjects/ML_Project/API/mnist_cnn.pth",map_location="cpu"))
model.eval()

from fastapi import FastAPI , UploadFile, File
from fastapi.responses import  JSONResponse
from torchvision.transforms import transforms
from PIL import Image

app = FastAPI()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to 1 channel (MNIST is grayscale)
    transforms.Resize((28,28)),                   # resize to 28x28 pixels (MNIST format)
    transforms.ToTensor(),                        # convert image to tensor (C,H,W) format
    transforms.Normalize((0.5,), (0.5,))          # normalize pixel values (mean=0.5, std=0.5)
])


@app.post("/predict/")
# FastAPI expects the request body to be multipart/form-data (the standard format for uploading files in HTTP).
async def predict(file :UploadFile = File(...)):
    # async def → allows FastAPI to handle file I/O efficiently
    # UploadFile = File(...) → tells FastAPI this endpoint accepts a file upload
    image = Image.open(file.file).convert("L")
    # file.file → raw file object
    # convert("L") → ensures image is grayscale
    image = transform(image).unsqueeze(0)
    # transform → preprocess the image
    # unsqueeze(0) → add batch dimension (model expects [batch_size, channels, H, W])

    with torch.no_grad():  # inference mode (no gradient computation, faster + saves memory)
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        # argmax(dim=1) → get class index with highest probability
        # .item() → convert tensor scalar → Python int

    return JSONResponse({"digit": prediction})
    # return prediction as JSON response
