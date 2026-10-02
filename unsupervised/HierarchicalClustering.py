import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage

X,_ = make_blobs(n_samples=50, centers= 3, random_state= 42)
X = StandardScaler().fit_transform(X)

# It calculates how clusters should be merged — step-by-step — and provides the raw data to plot a dendrogram.
# use for Visualization only
linked = linkage(X , method= 'ward') # 'ward' minimizes variance within clusters

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# It is the actual clustering algorithm that assigns a cluster label to each data point.
#It merges clusters based on a linkage method (same as above: 'ward', 'complete', 'average', etc.)
model = AgglomerativeClustering(n_clusters=3, linkage= 'ward', metric ='euclidean')
label = model.fit_predict(X)

plt.scatter(X[:,0] , X[:,1], c=label , cmap='rainbow')
plt.title("Agglomerative Clustering")
plt.show()