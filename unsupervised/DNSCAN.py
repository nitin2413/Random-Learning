# using sklearn
# eps radius hota hai points ka , min_sample minimum no of point , jab kese ek point ke eps(radis) main min_sam ke equal ya
# usse jada ponit ho to vo core point ban jata hai , boundary point must contain 1 core point but did not contain  min_sample
# number of points

# from sklearn.pothole import make_moons
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN

# X,_ = make_moons(n_samples=300 , noise= 0.1 , random_state=42)
# db_model = DBSCAN(eps = 0.4, min_samples= 10)
# labels = db_model.fit_predict(X)
#
# plt.scatter(X[:,0] , X[:,1], c=labels , cmap='rainbow' ,  s=40, edgecolors='k')
# plt.title("DBSCAN Clustering")
# plt.show()
# Unlike K-Means, which assumes clusters are roughly spherical (or convex),
# DBSCAN can find clusters of any shape — like moons, rings, or elongated blobs.
# It groups together points that are densely packed and separates outliers or noise.

# using numpy
import numpy as np
from collections import deque
def euclidean_distance(p1,p2):
    return np.linalg.norm(p1 - p2)
# point_idx refers to the index of the current point in the dataset X we can also see them as (core point)
# eps max distance consider to be neighbour
def region_quary(X,point_idx,eps):
    neighbour = []
    for i in range(len(X)):
        if euclidean_distance(X[point_idx],X[i]) < eps:
            neighbour.append(i)
    return neighbour

def expand_cluster(X,point_idx,label,eps,min_sample,cluster_id):
    neighours = region_quary(X,point_idx,eps)
    if len(neighours) < min_sample:
        label[point_idx] = -1 # make as noise
        return False
    else:
        label[point_idx] = cluster_id
        queue = deque(neighours)
        while(queue):
            neighours_id = queue.popleft()
            if label[neighours_id] == -1:# noice are the points which have < point than min_sample and also are not neighbours of core point
                label[neighours_id] = cluster_id #Change noise to border point,min_sample nahi hai per vo noice bhe nahi hai kyu ki uske pass core point hai
            if label[neighours_id] == 0:
                label[neighours_id] = cluster_id
                neighour_neighbours = region_quary(X,point_idx,eps)
                if len(neighour_neighbours) >= min_sample:
                    queue.extend(neighour_neighbours)
        return True

def dbscan(X,eps= 0.5 ,min_sample=10):
    label = np.zeros(len(X), dtype=int )
    cluster_id = 0
    for i in range(len(X)):
        if label[i] == 0:# not visited
            if expand_cluster(X,i,label,eps,min_sample,cluster_id + 1):# We call expand_cluster() to try to grow a new cluster starting from point i.
                cluster_id += 1
                #If it's successful (i.e. point i is a core point, and its neighborhood is large enough), then:
                #The function returns True then We increment cluster_id by 1
        return label

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Create dataset
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Run DBSCAN
label = dbscan(X, eps=0.2, min_sample=10)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=label, cmap='rainbow')
plt.title("DBSCAN Clustering with NumPy")
plt.show()
