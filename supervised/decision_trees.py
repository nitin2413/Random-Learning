import numpy as np
from sklearn.tree import plot_tree,DecisionTreeClassifier
import matplotlib.pyplot as plt

# Each row: [feature1, feature2]
X = np.array([
    [2.7, 2.5],
    [1.3, 1.5],
    [3.0, 3.5],
    [0.8, 0.6],
    [3.1, 2.9],
    [1.0, 1.0]
])

# Labels: 0 or 1
y = np.array([1, 0, 1, 0, 1, 0])

def gini(y):# Measures how often a randomly chosen element would be incorrectly labeled.
    classes , counts = np.unique(y,return_counts= True)# classes mai unique identity and count mai no of time it have been repeated
    pro_sq = sum((count/ len(y))**2 for count in counts)
    return 1-pro_sq

def split_data(X,y,feature_idx,threshold):# feature colomn hai and : row hai
    left_idx = X[:,feature_idx] <= threshold
    right_idx = X[:, feature_idx] > threshold
    return X[left_idx],y[left_idx],X[right_idx],y[right_idx]

def best_split(X,y):
    best_gini = 1
    best_feature = None
    best_threshold = None
    for feature in range(X.shape[1]):#loops over each feature (column) in the dataset X.
        thresholds = np.unique(X[:,feature])#Finds all unique values in the current feature column. loop is run for every unique feature
        for threshold in thresholds:
            X_left,y_left ,X_right,y_right = split_data(X,y,feature,threshold)
            if len(y_left ) == 0 or len(y_right ) == 0:
                continue
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            weighted_gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature,best_threshold
class Node:
    def __init__(self,left=None,right=None,feature=None,threshold=None,value=None):
        self.left= left
        self.right= right
        self.feature = feature
        self.threshold = threshold
        self.value = value# for leaf node

def build_tree(X,y,depth=0,max_depth=2):
    if len(np.unique(y)) == 1 or depth > max_depth:
        leaf_node = np.bincount(y).argmax()# select most repeated value in y as leaf node
        return Node(value = leaf_node )
    feature, threshold = best_split(X,y)
    if feature is None:
        return Node(value=np.bincount(y).argmax())

    X_left, y_left,X_right,y_right = split_data(X,y,feature,threshold)
    left = build_tree(X_left,y_left,depth+1,max_depth)
    right = build_tree(X_right ,y_right,depth+1,max_depth)

    return Node(left= left,right = right,feature= feature,threshold = threshold)
    # we are not including value Because decision nodes only split — they do not predict a value. Only leaf nodes do, so only they need a value

def predict_single(node,x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold :
        return predict_single(node.left,x)
    else:
        return predict_single(node.right,x)

#For each data point x in X, call the predict_single function with the root of the tree and the data point.
def predict(tree,X):
    return np.array([predict_single(tree,x) for x in X])

tree = build_tree(X,y,max_depth=2)
Predict = predict(tree,X)
print("prediction:", Predict)
print("true label:", y)


########################################################################################################################

# clf = DecisionTreeClassifier(max_depth = 2,criterion='gini')
# clf.fit(X,y)
#
# plt.figure(figsize=(10,7))
# plot_tree(clf,feature_names=["f1","f2"],class_names = ["0","1"], filled = True)
# plt.title("sklearn decision tree ")
# plt.show()
