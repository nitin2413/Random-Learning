#the point have random value and based on the the value they find nearest centroid (1centroid = 1cluster)
import numpy as np
import matplotlib.pyplot as plt

# creating sample set
np.random.seed(42)
from sklearn.datasets import make_blobs
X ,y = make_blobs(n_samples=600,centers=3,cluster_std= [2.4 , 0.5 , 0.9])#cluster_std controls the spread (standard deviation) of each cluster.

# select centroid
def initialize_centroids(X,k):
    indices = np.random.choice(X.shape[0],k,replace = False) # replace is false because we want uniqe selection
    return X[indices]

# determine which centroid is nearest to each data point
def closest_centroid (X , centroids):
    distance = np.linalg.norm(X[:, np.newaxis] - centroids, axis = 2) # np.newaxis is use to re shape x so that it can match centroids dimenssion
    return np.argmin(distance , axis= 1) # argmin find the nearest centroids for data point
    #axis=2 means: compute the Euclidean norm (L2 distance) along the feature dimension
    #axis=1 means: for each row, find the index (i.e., centroid) with the minimum distance
# re-compute the centroid
def compute_centroid(X, lable , k):
    return np.array([X[lable == i].mean(axis=0) for i in range(k)])
#After running K-Means, every data point is assigned to one of the k clusters.
#These assignments are stored in an array called labels.
#labels = [0, 1, 1, 0, 2, 2]
#The first point belongs to cluster 0
#The second and third points belong to cluster 1



# full kmean loop
def kmean (X,k , max_iter = 100):
    centroids = initialize_centroids(X,k)
    for _ in range (max_iter):
        lable = closest_centroid(X,centroids)
        new_centroid = compute_centroid(X, lable , k)
        if np.all(centroids == new_centroid):
            break
        centroids = new_centroid
    return centroids, lable

k = 3
centroids , lable = kmean(X, k)

plt.scatter(X[:, 0], X[:, 1], c=lable, s=30, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200)
plt.title("K-Means Clustering (from scratch)")
plt.show()


# from sklearn.cluster import KMeans
#
# X = np.random.rand (100, 2)
# k_mean = KMeans(n_clusters=3 , random_state= 42 )
# k_mean.fit(X)
#
# center = k_mean.cluster_centers_
# labels = k_mean.labels_
#
# plt.scatter(X[:,0], X[:,1] , c = labels, cmap= 'viridis')
# plt.scatter(center[:, 0], center[:, 1], c='red', marker='x', s=100)
# plt.title('K-Means Clustering (scikit-learn)')
# plt.show()