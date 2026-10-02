import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader , TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv("/Users/nitinsingh/PycharmProjects/ML_Project/project/earthquake.csv")

print(df.head())
print(df.columns.tolist())
print(df.describe())

X = df.drop(columns=["tsunami"]).values
y = df["tsunami"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size= 0.2, random_state=42)

X_train_tensor = torch.tensor(X_train , dtype=torch.float32)
y_train_tensor = torch.tensor(y_train , dtype=torch.float32).view(-1,1)

X_test_tensor = torch.tensor(X_test , dtype=torch.float32)
y_test_tensor = torch.tensor(y_test , dtype=torch.float32).view(-1,1)

train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
train_loader  = DataLoader(train_dataset,batch_size=32,shuffle=True)

test_dataset = TensorDataset(X_test_tensor,y_test_tensor)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

input_feature = X.shape[1]
hidden_layer = 64
output_feature = 1

class Tsunami(nn.Module):
    def __init__(self,input_feature,hidden_layer,output_feature):
        super().__init__()

        self.tsunami = nn.Sequential(
            nn.Linear(input_feature,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,output_feature),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.tsunami(x)


model = Tsunami(input_feature,hidden_layer,output_feature)
optimizer = optim.Adam(model.parameters(),lr=0.01)
loss_function = nn.BCELoss()

epochs = 20

for epoch in range(epochs):
    for x , y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred,y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()
# Convert to numpy for sklearn
y_true_np = y_test_tensor.numpy()
y_pred_np = y_pred_class.numpy()

cm = confusion_matrix(y_true_np, y_pred_np)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true_np, y_pred_np))

# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Matrix of Numerical Features")
# plt.show()