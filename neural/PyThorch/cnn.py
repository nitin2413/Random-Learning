# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
#
# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Transform to tensor and normalize
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # Download and load MNIST data
# train_dataset = torchvision.pothole.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = torchvision.pothole.MNIST(root='./data', train=False, transform=transform)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
#
# # Define the CNN model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Flattened from 32x7x7 to 128
#         self.fc2 = nn.Linear(128, 10)          # Output layer for 10 classes
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # -> [16, 14, 14]
#         x = self.pool(F.relu(self.conv2(x)))  # -> [32, 7, 7]
#         x = x.view(-1, 32 * 7 * 7)            # Flatten
#         x = F.relu(self.fc1(x))              # Dense layer
#         x = self.fc2(x)                      # Output layer (raw scores)
#         return x
#
# # Initialize model, loss function, and optimizer
# model = SimpleCNN().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # Training loop
# num_epochs = 3
# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
#
# # Test accuracy
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f"Test Accuracy: {100 * correct / total:.2f}%")

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device("mps" if torch.mps.is_available() else "cpu")

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root="./MNIST" , train= True , download=True , transform = transform)
test_dataset = torchvision .datasets.MNIST(root='./MNIST', train= False, download= True , transform= transform)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle= True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        #stride refers to how many pixels the filter (or pooling window) moves at each step when it scans over the input.
        #in_channels ,Input has 1 channel (MNIST is grayscale). mnist can also have RGB which have 3 channel
        #kernel is use to dicied size of filter
        #out_channels tells us how many different filters (also called kernels) the layer applies to the input. overr here we have applied 8 different filters
        #each filter represent one feature maps
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=8,kernel_size=3,padding=1,stride= 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)#over here we reducing width and height by half.
        #pooling is use to decreasing the dimensions also called downsampling
        self.conv2 = nn.Conv2d(8,16,3,1,1)#second convolution its input is output of 1st convolution
        self.flatten = nn.Flatten()#Converts 3D feature maps [16, 7, 7] into a flat 1D vector [16 * 7 * 7] = [784]
        self.dense = nn.Linear(16*7*7,10)#flatten vector and 10 for output vecter [0-9]

    def forward(self,x):
        # Each line applies a transformation. Order is conv → relu → pool → conv → relu → pool → flatten → dense
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dense(x)
        return x

model = CNNModel().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)#model.parameters(), Gives all the learnable parameters of the model.

epochs = 7

for epoch in range(epochs):
    running_loss = 0.0
    for images ,labels in train_loader:
        images , labels = images.to(device), labels.to(device)
        # 2. Forward pass
        outputs = model(images)
        # 3. Compute loss
        loss = loss_function(outputs, labels)
        # 4. Backward pass
        #Optimizer Steps
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()#loss.item() gives the scalar (Python number) from the tensor loss.
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Save trained model weights
torch.save(model.state_dict(), "mnist_cnn.pth")

model.eval()
correct = 0
total  = 0

with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs.data , 1)#predicted will contain the index of the class with the highest score
        # when you specify 1, you're asking PyTorch to find the maximum value along the columns (the second dimension) of that tenso
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

import matplotlib.pyplot as plt
import numpy as np


# Function to show images
def imshow(img):
    img = img.numpy().squeeze()  # remove batch dimension and convert tensor to numpy
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# Get some test images and labels
dataiter = iter(test_loader)
images, labels = next(dataiter)


# Move to device
images, labels = images.to(device), labels.to(device)

# Get model predictions
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Move tensors back to CPU for visualization
images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()

# Show 6 images with predicted and true labels
for idx in range(2):
    print(f"True Label: {labels[idx].item()}, Predicted: {predicted[idx].item()}")
    imshow(images[idx])

# def imgshow(img):
#     img = img.numpy().squeeze()
#     plt.imshow(img,cmap='gray')
#     plt.axis('off')
#     plt.show()
#
# dataiter = iter(test_loader)
# images,labels = next(dataiter)
#
# images,labels = images.to(device),labels.to(device)
#
# with torch.no_grad():
#     outputs = model(images)
#     _,pridicted = torch.max(outputs,1)
#
# images = images.cpu()
# labels = labels.cpu()
# pridicted = pridicted.cpu()
#
# for idx in range(3):
#     print(f"true label: {labels[idx].item()}, pridected:{pridicted[idx].item()}")
#     imgshow(images[idx])
