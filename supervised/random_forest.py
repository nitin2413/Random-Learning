import numpy as np
from collections import Counter
# import pothole and label
X = np.array([
    [2.7, 2.5],
    [1.3, 1.5],
    [3.0, 3.5],
    [0.8, 0.6],
    [3.1, 2.9],
    [1.0, 1.0]
])
y = np.array([1,0,1,0,1,0])

#gini
def gini(y):
    classes, counts = np.unique(y,return_counts= True)
    pri_sq = sum((count/ len(y))**2 for count in counts)
    return 1-pri_sq

#spilt data
def spilt_data(X,y,feature_idx,threshold):
    left_idx = X[:,feature_idx] <= threshold
    right_idx = X[:,feature_idx] > threshold
    return X[left_idx],X[right_idx],y[left_idx],y[right_idx]

# best spilt
def best_split(X,y,num_feature=None):
    best_gini = 1
    best_feature = None
    best_threshold = None
    features = np.random.choice(X.shape[1], num_feature or X.shape[1], replace=False)# this line introduce randomness in selecting feature
    for feature in features:
        thresholds = np.unique(X[:,feature])
        for threshold in thresholds:
            X_left,y_left,X_right,y_right = spilt_data(X,y,feature,threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            left_gini = gini(y_left)
            right_gini = gini(y_right)
            weighted_gini = (len(y_left)/len(y)) * left_gini + (len(y_right)/len(y)) * right_gini

            if weighted_gini < best_gini :
                best_gini  = weighted_gini
                best_feature = feature
                best_threshold = threshold
    return best_feature,best_threshold


#node
class Node:
    def __init__(self,left=None,right=None,feature=None,threshold=None,value=None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold= threshold
        self. value = value


#buildtree
def build_tree(X,y,depth=0,max_depth=3,num_feature= None):
    if len(np.unique(y)) == 1 or depth > max_depth:
        leaf_node = np.bincount(y).argmax()
        return Node(value=leaf_node)
    feature, threshold = best_split(X,y,num_feature)
    if feature is None:
        return Node(value=np.bincount(y).argmax())
    X_left, X_right, y_left, y_right = spilt_data(X, y, feature, threshold)####
    # without this line we were passing all the data on both left and right side of tree with was causing error
    left = build_tree(X_left,y_left,depth+1,max_depth,num_feature)
    right = build_tree(X_right,y_right,depth+1,max_depth,num_feature)
    return Node(left= left, right= right , feature = feature , threshold = threshold)

# pridict
def pridict_tree(x,node):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return pridict_tree(x,node.left)
    else:
        return pridict_tree(x,node.right)

# bootstrap generate random sample and also take random feature for very tree

# The first argument n_samples specifies the range of indices to choose from (0 to n_samples - 1).
# The second argument n_samples defines how many indices to draw from that range.
# replace=True means that the same index can be sampled more than once.
def bootstrap(X,y):
    n_smaples = len(y)
    idx = np.random.choice(n_smaples,n_smaples,replace=True)
    return X[idx], y[idx]

class random_forest:
    def __init__(self,n_trees= 5,max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []# to store every tree in a single array

    def fit(self,X,y):
        self.trees = []# make sure here the tree is empty
        for _ in range(self.n_trees):
            X_samp , y_samp = bootstrap(X,y)
            tree = build_tree(X_samp,y_samp,depth=0,max_depth=self.max_depth,num_feature= int(np.sqrt(X.shape[1])))
            self.trees.append(tree)# we did not use return, as append will return none

    def pridict(self,X):
        tree_preds = np.array([[pridict_tree(x,tree) for tree in self.trees] for x in X])# sabke alag input sample (x)hai, pridiction of every tree is stored in it
        final_pred = [Counter(row).most_common(1)[0][0] for row in tree_preds]
        return np.array(final_pred)
rf = random_forest(n_trees=10, max_depth=3)
rf.fit(X, y)

pred = rf.pridict(X)
print("Predictions:", pred)
print("True Labels:", y)
accuracy = np.sum(pred == y) / len(y)
print("Accuracy:", accuracy)
