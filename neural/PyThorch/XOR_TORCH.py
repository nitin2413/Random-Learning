import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

class XORNet (nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2,2) # input to hidden layer
        self.output = nn.Linear(2,1) # hidden to output layer
        self.sigmoid = nn.Sigmoid() # we can use any activation function
        # self.relu = nn.ReLU()

    def forward(self,x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) #update model parameter using Stochastic Gradient Descent,(SGD)
losses = []

# train model
epochs = 10000
for epoch in range(epochs):
    # forward pass (This section computes the output of the model using the current parameters.)
    output = model(X)
    loss = criterion(output, y)
    losses.append(loss.item())

    # backward pass (This section updates the model's parameters to minimize the loss.)
    optimizer.zero_grad()#eleminates grafient of privious iteration
    loss.backward() #backward propagation (Computes the gradients of the loss with respect to the model's parameters using backpropagation.)
    optimizer.step() #update parameter

    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

with torch.no_grad():
    predictions = model(X).round()
    print("\nPredictions:")
    print(predictions)

plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# class XORNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden= nn.Linear(2,4)
#         self.output = nn.Linear(4,1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self,x):
#         x = self.sigmoid(self.hidden(x))
#         x = self.sigmoid(self.output(x))
#         return x
#
# model =XORNet()
# criterion =nn.BCELoss()
# optimizer = optim.Adam(model.parameters(),lr=0.2 )
# losses = []
#
# epochs = 1000
# for epoch in range(epochs):
#     output = model(X)
#     loss = criterion(output,y)
#     losses.append(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # if epoch % 1000 == 0:
#     #     print(f"Epoch {epoch}, Loss: {loss}")
#     if (epoch + 1) % 200 == 0 or epoch == 0:
#         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#
# with torch.no_grad():
#     prediction = model(X).round()
#     print("\nPredictions:")
#     print(prediction)

