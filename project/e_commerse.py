import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv", encoding='ISO-8859-1')#encoding ensures that special characters in the file are correctly interpreted.

# Drop missing CustomerIDs
df = df.dropna(subset=['CustomerID'])

# Remove returns (negative quantity)
df = df[df["Quantity"] > 0]

# Create a TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(df.shape)
print(df.head())


reference_date = df['InvoiceDate'].max() + pd.Timedelta(days= 1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,# Recency (How recently did they buy?)
    'InvoiceNo': 'nunique',                                  # fecuancy
    'TotalPrice': 'sum'                                      # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print(rfm.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# this part is to find most optimal cluster for k_mean using elbow curve
#Optimal clusters (Elbow method){
inertia = []

# Try cluster sizes from 1 to 10
for k in range(1,11):
    k_mean = KMeans(n_clusters=k, random_state=42)
    k_mean.fit(rfm_scaled)
    inertia.append(k_mean.inertia_)
#   }

# Plot the results
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.grid(True)
# plt.show()

k_mean = KMeans(n_clusters= 4 , random_state= 42)
rfm['Cluster'] = k_mean.fit_predict(rfm_scaled) # create a new colomn name cluster pridicted values from rfm_scaled are added in cluster

# Variance = how spread out the data is.
# If a feature has high variance, it means the values vary a lot (they’re spread out).
# If it has low variance, the values are all close to the mean.

from sklearn.decomposition import PCA # pca is use to reduce dimension which eases graphical representation of feature
pca = PCA(n_components=2)
components = pca.fit_transform(rfm_scaled)

# plt.figure(figsize=(8,6))
# plt.scatter(components[:, 0], components[:, 1], c=rfm['Cluster'], cmap='Set1', s=50)
# plt.title('Customer Segments (K-Means Clustering)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.grid(True)
# plt.colorbar(label='Cluster')
# plt.show()
#import plotly.express as px
#
# df_pca = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
# df_pca['Cluster'] = rfm['Cluster']
#
# fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Cluster')
# fig.show()

# Group by cluster and look at mean RFM values
rfm_summary = rfm.groupby('Cluster').mean(numeric_only=True)
rfm_count = rfm['Cluster'].value_counts().sort_index()
rfm_summary['count']= rfm_count
print(rfm_summary)
# Cluster 0, 1, and 2 are just labels — you need to translate them:
# for example :- we take 0 th cluster if Recency is low, Frequency high,Monetary high
# then it means Loyal customer, high-value customers (VIPs)
# Add cluster column to the original dataframe
original_data_with_clusters = df.copy()
original_data_with_clusters['Cluster'] = rfm['Cluster']

# Optional: sort by cluster
original_data_with_clusters = original_data_with_clusters.sort_values('Cluster')

original_data_with_clusters.to_csv('customers_segmented.csv', index=False)
print("CSV file saved successfully.")
print(inertia)
