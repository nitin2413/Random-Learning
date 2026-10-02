import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
import numpy as np
import seaborn as sns


df = pd.read_csv("house_price_pridictor.csv")
df = df.dropna()
df = df.drop(columns="country")


########################################################################
# numerical_data = df.select_dtypes(include=["number"])
# # print(numerical_data)
# categorical_data = df.select_dtypes(include=["object",'category'])
# # print(categorical_data)
# encoding is not useful here as it is creating way to much columns
# one-hot encoding created 4,525 extra columns for 4,600 houses
# encoder = OneHotEncoder(drop=None)
# encoded_df = encoder.fit_transform(categorical_data)
#
#########################################################################


df["city_id"] = pd.factorize(df['city'])[0]
# print(df[['city','city_id']])

df["zip_id"] = pd.factorize(df['statezip'])[0]
# print(df[['statezip','zip_id']])

df = df.drop(columns=['city','statezip','street','date','sqft_above'])

# Separate target variable and features
# Log-transform price for better modeling
df["price"] = np.log1p(df["price"])
df["room_density"] = (df["bathrooms"]+df["bedrooms"]) / df["sqft_living"]
df["age"] = 2014 - df["yr_built"]



final_df = df.columns.tolist()
final_df.remove("price")

X = df[final_df]
y = df["price"]


corr_matrix = df[final_df].corr()
# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_data = scaler.fit_transform(X_train)
X_test_data = scaler.transform(X_test)

# storing city_id and zip_id , as we need them for embedding
X_train_city = X_train["city_id"].values
X_train_zip = X_train["zip_id"].values
X_test_city = X_test["city_id"].values
X_test_zip = X_test["zip_id"].values

# converting numpy array to tensor
X_train_tensor = torch.tensor(X_train_data, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_data, dtype=torch.float32)

X_train_city = torch.tensor(X_train_city, dtype=torch.long)
X_test_city = torch.tensor(X_test_city, dtype=torch.long)

X_train_zip = torch.tensor(X_train_zip, dtype=torch.long)
X_test_zip = torch.tensor(X_test_zip, dtype=torch.long)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# concatinating all data in single space
train_data = TensorDataset(X_train_tensor,X_train_city,X_train_zip,y_train_tensor)
test_data = TensorDataset(X_test_tensor,X_test_city ,X_test_zip ,y_test_tensor )

# loading final cleaned and tensor data
train_loader = DataLoader(train_data ,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

num_cities = df["city_id"].nunique()
num_zips = df["zip_id"].nunique()

print(num_zips)
print(num_cities)

# num_numeric_feature is X_train_tensor and rest all like wise
# embedding city and zip though they are numeric but represent string value of orignal data
class HousePriceModel(nn.Module):
    def __init__(self,num_numeric_feature,num_city,num_zip,emb_dim_city=50,emb_dim_zip=50,hidden_dim=64):
        super().__init__()

        self.emb_city = nn.Embedding(num_city,emb_dim_city)
        self.emb_zip = nn.Embedding(num_zip,emb_dim_zip)

        input_dim= num_numeric_feature  + emb_dim_city + emb_dim_zip

        self.fc = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),# input dimension /64
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim//2),# 64 /32
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features=1) # 32 / 1 , 1 because predicting continuous output or can say single class
        )

    def forward(self,x_num,x_city,x_zip):
        city_vec = self.emb_city(x_city)
        zip_vec = self.emb_zip(x_zip)
        x = torch.cat([x_num,city_vec,zip_vec], dim=1)
        return self.fc(x)

model = HousePriceModel(
    num_numeric_feature= len(final_df),
    num_city= num_cities,
    num_zip= num_zips
)
optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
loss_function = nn.MSELoss()

epochs = 500

for epoch in range(epochs+1):
    model.train()
    total_loss = 0

    for x_num , x_city , x_zip, y in train_loader: # train_data = TensorDataset(X_train_tensor,X_train_city,X_train_zip,y_train_tensor)
        optimizer.zero_grad()
        pred = model(x_num,x_city,x_zip)
        loss = loss_function(pred,y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_num.size(0)
    avg_loss = total_loss / len(train_loader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_num, x_city, x_zip, y in test_loader:
            output = model(x_num, x_city, x_zip)
            loss = loss_function(output, y)
            test_loss += loss.item() * x_num.size(0)
        avg_test_loss = test_loss / len(test_loader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, test_Loss: {avg_test_loss} || train_Loss: {avg_loss}")

