## reggresion tree

import numpy as np

class Regression_Tree:
    def __init__(self,max_depth=3, min_no_split=2):
        self.max_depth = max_depth
        self.min_no_split = min_no_split
        self.tree = None

    def fit(self,X,y):
        self.tree = self.build_tree(X,y,depth=0)

    def build_tree(self,X,y,depth):
            num_samples, num_features = X.shape# store all the valied row in sample and cloumn in feature
            if depth >= self.max_depth or num_samples < self.min_no_split:
                leaf_node = np.mean(y)
                return leaf_node

            best_mse = float("inf") # inf say infinite
            best_feature, best_threshold = None,None

            for feature in range(num_features):
                thresholds =  np.unique(X[:, feature])
                for threshold in thresholds:
                    left_side = X[:,feature] <= threshold
                    right_side = X[:,feature] > threshold

                    if np.sum(left_side) == 0 or np.sum(right_side) == 0:
                        continue

                    y_left , y_right = y[left_side] , y[right_side]
                    mse = self.calculate_mse(y_left,y_right)

                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold

            if best_feature is None:
                return np.mean(y)

            left_side = X[:,best_feature] <= best_threshold
            right_side = X[:,best_feature] > best_threshold

            left_tree = self.build_tree(X[left_side],y[left_side],depth+1)
            right_tree = self.build_tree(X[right_side],y[right_side],depth+1)

            return {
                'feature':best_feature,
                'threshold':best_threshold,
                'left':left_tree,
                'right':right_tree
            }
    def calculate_mse (self , y_left, y_right):
        left_mse = np.var(y_left) * len(y_left)
        right_mse = np.var(y_right ) * len(y_right)
        return (left_mse + right_mse) / (len(y_left) + len(y_right))

    def predict_one(self,x,node):
        if not isinstance(node,dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x,node['left'])
        else:
            return self.predict_one(x,node['right'])

    def predict(self,X):
        return np.array([self.predict_one(x,self.tree) for x in X])

## simple gradiant boosting ##

class SimpleGradiantBoosting:
    def __init__(self,no_of_trees = 10,learning_rate = 0.5, max_depth_forall= 3):
        self.no_of_trees = no_of_trees
        self.learning_rate = learning_rate
        self.max_depth_forall = max_depth_forall
        self.trees = []#list of stored train trees
        self.initial_prediction = None#Starting guess before any trees are added

    def fit (self,X,y):
        self.initial_prediction = np.mean(y)
        prediction = np.full(y.shape , self.initial_prediction)

        for _ in range(self.no_of_trees):

            residual = y - prediction
            tree = Regression_Tree(max_depth=self.max_depth_forall)
            tree.fit(X,residual)

            update = tree.predict(X)
            prediction += self.learning_rate*update

            self.trees.append(tree)

    def predict(self,X):
        prediction = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return prediction


# Dummy dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.1, 1.9, 3.0, 4.1, 5.0])

# Fit Gradient Boosting
gb = SimpleGradiantBoosting(no_of_trees = 10,learning_rate = 0.5, max_depth_forall= 3)
gb.fit(X, y)

# Predictions
preds = gb.predict(X)
print("Predictions:", preds)





