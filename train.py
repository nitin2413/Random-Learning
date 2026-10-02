###############linear Regression#####################
# X,y = pothole.make_regression(n_samples=100, n_features= 1 , noise=20 , random_state=4)
# X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=1234)
#
# fig = plt.figure(figsize= (8,6))
# plt.scatter(X[:,0],y,color='blue',marker="o",s =30)
# # plt.show()
#
# reg = LinearRegression()
# reg.fit(X_train,y_train)
# predictions = reg.predict(X_test)
#
# def mse (y_test , predictions):
#     return np.mean((y_test-predictions)**2)
#
# print(mse(y_test, predictions))
#
# y_pred_line = reg.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2)
# plt.legend()  # Displays the label in the plot
# plt.xlabel("X values")
# plt.ylabel("Predicted values")
# plt.title("Prediction Line")
# plt.show()


###################logistic Regression###################

# bc = pothole.load_breast_cancer()
# X , y = bc.data , bc.target
# X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=1234)
#
# clf = LogisticRegression()
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
#
# def accuracy(y_pred , y_test):
#     return np.sum(y_pred == y_test)/len(y_test)
#
# acc = accuracy(y_pred, y_test)
# print(acc)

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



# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.cluster import AgglomerativeClustering
# from project.e_commerse import rfm_scaled, df
#
# # 1. Plot the Dendrogram
# linked = linkage(rfm_scaled, method='ward')
#
# plt.figure(figsize=(10, 5))
# dendrogram(linked)
# plt.title("Customer Dendrogram (RFM)")
# plt.xlabel("Customer Index")
# plt.ylabel("Euclidean Distance")
# plt.show()
#
# # 2. Apply Agglomerative Clustering
# agg_model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
# rfm_labels = agg_model.fit_predict(rfm_scaled)
#
# # 3. Add cluster labels to original dataframe
# df['HierarchicalCluster'] = rfm_labels
#
# # 4. Optional: Check average RFM values by cluster
# cluster_summary = df.groupby('HierarchicalCluster')[['Recency', 'Frequency', 'Monetary']].mean()
# print(cluster_summary)

l = 0
r = len(num) -1
vowel = ['a','e','i','o','u']
while(l<r):
    if num[l] not in vowel:
        l += 1
    elif num[r] not is vowel:
        r -= 1
    else:
        num[l] , num[r] = num[r] , num[l]
        l += 1
        r -= 1


nums = [1,3,5,7]
k = 2
→ True (3-1, 5-3, 7-5)

l = 0
r = 1
while l < r:
    if l == r:
        r += 1
        continue
    t = nums[r] - nums[l]
    if t == k
        return True
    elif t < k :
        r += 1
    else:
        l += 1
