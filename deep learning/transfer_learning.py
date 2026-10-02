# why do we need transfer_learning?
# Transfer learning is necessary because it allows models to leverage pre-existing knowledge from related tasks.
# This technique significantly reduces the time, computational resources, and data required to train new models from scratch.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models , datasets , transforms
from torch.utils.data import  DataLoader , random_split


BATCH_SIZE = 42
lr = 1e-3
Epochs = 5
device = "mps" if torch.mps.is_available() else "cpu"


# As we are using resnet18 model need must make sure dataset fit the dimensionality requirement
transform = transforms.Compose([
    transforms.Resize((224,224)), #  ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
train_data = int(0.8 * len(dataset))
val_data = len(dataset) - train_data
train_ds , val_ds = random_split(dataset, [train_data, val_data])

train_loader = DataLoader(train_ds , shuffle= True , batch_size=BATCH_SIZE)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=BATCH_SIZE)

# resnet18 has 18 layer trained on IMAGENET data set, weights= use the weights collect by the dataset
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 )


# freeze all the layer (weights does not update )
for param in resnet.parameters():
    param.requires_grad = False

########  FINE Tuning Encoder  #########
# # in fine tuning we add some encoder layer so that model is suited for our dataset
# # Unfreeze only layer4 (last residual block)
# for name , param in resnet.parameters():
#     if "layer4" in name:
#         param.requires_grad = True
#
# optimizer = optim.Adam(filter(lambda  p: p.requires_grad, model.parameter()), lr=1e-4)
# # we could write this optimizer = optim.Adam(model.fc.parameters(), lr = 1e-4) but it would include
# # all the layer even the frozen one which would not raise any problem except consuming space
########################################

# resnet.fc.in_features, this collect input feature of last layer fc(last layer )
num_feature = resnet.fc.in_features
# out feature 10 as are data set is for 10 classes
resnet.fc = nn.Linear(num_feature,10) # we remove last layer from pre-trained model and add are own layer

model = resnet.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = lr)

for epoch in range(Epochs):
    running_loss = 0.0
    correct = 0
    for image, label in train_loader:
        image , label = image.to(device)  , label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output,label)
        loss.backward()
        optimizer.step()
        # image.size(0) == number of samples in the batch,
        # Multiplying ensures each image contributes equally to the total epoch loss, regardless of batch size.
        running_loss += loss.item() * image.size(0)
        correct += (output.argmax(1) == label).sum().item()

    epoch_loss = running_loss / len(train_ds)
    epoch_acc = correct /len(train_ds)

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for image , label in val_loader:
            image , label = image.to(device) , label.to(device)
            output = model(image)
            val_correct += (output.argmax(1) == label).sum().item()

    val_acc = val_correct/len(val_ds)
    print(f'Epoch: [{epoch+1}/{Epochs}] || loss:{epoch_loss:.4f} || acc:{epoch_acc} || Val Acc: {val_acc:.4f}')
