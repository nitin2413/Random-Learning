# # link(https://chatgpt.com/share/6857ce1d-0c4c-800f-8bf6-4bd224033777)
# Gaussian Mixture Models(gmm) E-step explain in simple term

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N=300
x1 = np.random.multivariate_normal([0,0],[[1,0.1],[0.1,1]],N//3)
x2 = np.random.multivariate_normal([5,5],[[1,-0.3],[-0.3,1]],N//3)
x3 = np.random.multivariate_normal([0,5],[[1,0.5],[0.5,1]],N//3)

X= np.vstack ((x1,x2,x3))
plt.scatter(X[:,0],X[:,1],s=10)
plt.title("data")
plt.show()

# Step 2: Initialize parameters(At this point, GMM thinks there are K Gaussian blobs, centered somewhere at random.)
def initialize_parameters(X, K):
    n_samples, n_features = X.shape
    indices = np.random.choice(n_samples, K, replace=False)# if k = 3 then it will choose 3 random point from data set and make them mean
    #mean
    mu = X[indices]# above line happen over here as it use X dataset
    #covariance is a measure of how two variables change together.
    #Variance measures how spread out a single feature is.
    cov = np.array([np.eye(n_features)] * K)# this part {np.eye(n_features)} create identity matrix * k mean k number of matrix, creating circular or spherical shape
    # same weight of 1/k , indicating that each component is assumed to contribute equally to the overall mixture.
    pi = np.ones(K) / K
    return mu, cov, pi

# Step 3: Multivariate Gaussian PDF
#This function computes the probability density of a point x under a multivariate Gaussian, using the mean and covariance.
# this is a mathematical formula which is use to computes probability density
# use mean , covariance(determinant and inverse) and prior pi
def multivariate_gaussian(x,mu,cov):
    n = x.shape[0]                             #number of features (e.g. 2D → n = 2)
    diff = x - mu                              # difference from mean
    det = np.linalg.det(cov)                   # determinant of covariance matrix
    inv = np.linalg.inv(cov)                   # inverse of covariance matrix

    norm_const = 1/(np.sqrt((2*np.pi) ** n *det))
    exp = np.exp(-0.5* diff.T @ inv @ diff) # @ is used for matrix multiplication
    return norm_const * exp

# Step 4:
#E-step: Compute soft assignments — the probability each point belongs to each cluster.”
#E-step computes the responsibilities gamma
# def e_step(X, mu, cov, pi, K):
#     n_samples = X.shape[0]
#     gamma = np.zeros((n_samples, K))
#     for k in range(K):
#         for i in range(n_samples):
#             gamma[i, k] = pi[k] * multivariate_gaussian(X[i], mu[k], cov[k])
#     gamma /= gamma.sum(axis=1, keepdims=True) #type of softmax in neural net
#     return gamma
def e_step(X,mu,cov,pi,K):
    n_samples = X.shape[0]
    gamma = np.zeros((n_samples,K))
    for k in range(K):
        for i in range(n_samples):
            gamma[i,k] = pi[k] * multivariate_gaussian(X[i],mu[k],cov[k]) # compute probability density
    gamma /= gamma.sum(axis=1, keepdims=True) # normalize point all over k cluster
    return gamma

# Step 5: M-step
# def m_step(X, gamma, K):
#     n_samples, n_features = X.shape
#     N_k = np.sum(gamma, axis=0)# weighted sum of responsibility(gamma)
#     mu = np.zeros((K, n_features))
#     cov = np.zeros((K, n_features, n_features))
#     pi = N_k / n_samples # fraction of data assign to each cluster
#     for k in range(K):
#         #weighted average of all data points, where points with higher responsibility for cluster k contribute more
#         #[:, np.newaxis]Converts the 1D array (n_samples,) into a column vector (n_samples, 1).
#         mu[k] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / N_k[k]# X ---> (n_sample,n_feature)
#         diff = X - mu[k]
#         #weighted covariance matrix, showing how the features vary together for points in cluster k.
#         cov[k] = (gamma[:, k][:, np.newaxis] * diff).T @ diff / N_k[k]
#     return mu, cov, pi
def m_step(X,gamma ,K):
    n_samples , n_feature = X.shape
    N_k = np.sum(gamma , axis= 0 )
    mu = np.zeros((K,n_feature))
    cov = np.zeros((K,n_feature,n_feature))# two time n_feature as covariance is squared matrix
    pi = N_k / n_samples
    for k in range(K):
        mu[k] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / N_k[k]# N_k[k] point belong to cluster k
        diff = X-mu[k]
        cov[k] = (gamma[:, k][:, np.newaxis] * diff).T @ diff / N_k[k]
    return mu ,cov, pi

# Step 6: Run EM algorithm
# def run_em(X, K, max_iters=100, tol=1e-4):
#     mu, cov, pi = initialize_parameters(X, K)
#     for iteration in range(max_iters):
#         old_mu = mu.copy()
#         gamma = e_step(X, mu, cov, pi, K)
#         mu, cov, pi = m_step(X, gamma, K)
#         if np.linalg.norm(mu - old_mu) < tol:
#             print(f"Converged in {iteration+1} iterations.")
#             break
#     return mu, cov, pi, gamma
def run_em(X,K,max_eter=100,tol=1e-4):
    mu,cov,pi = initialize_parameters(X,K)
    for iteration in range(max_eter):
        old_mu = mu.copy()
        gamma = e_step(X,mu,cov,pi,K)
        mu,cov,pi = m_step(X,gamma,K)
        if np.linalg.norm(mu-old_mu) < tol:
            print(f'Converged in {iteration+1} iterations.')
            break
    return mu,cov,pi,gamma


# Step 7: Predict labels from soft probabilities
def predict_cluster(gamma):
    return np.argmax(gamma, axis=1)

# Run the algorithm
K = 3
mu, cov, pi, gamma = run_em(X, K)
labels = predict_cluster(gamma)

# Step 8: Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.scatter(mu[:, 0], mu[:, 1], c='red', marker='x')  # cluster centers
plt.title("GMM Clustering (NumPy from Scratch)")
plt.show()