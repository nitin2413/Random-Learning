import numpy as np

def sigmoid(x):
  x = np.clip(x, -500, 500)#value less than -500 becomes -500 and greater 500 become 500
  return (1/(1+np.exp(-x)))


class LogisticRegression():

    def __init__(self,lr=0.01,n_iters=1000):
        self.weights =None
        self.bias = None
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples,n_features =X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X,self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T , (predictions - y))#1/n * submission from 0 to n X transpose y(pred) - y
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self,X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)

        # classification
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred


# import numpy as np
#
# def sigmoid(x):
#     x = np.clip(x, -500, 500)
#     return 1 / (1 + np.exp(-x))
#
# def binary_cross_entropy(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Avoid log(0) errors
#     return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, n_iters=1000):
#         self.weights = None
#         self.bias = None
#         self.lr = lr
#         self.n_iters = n_iters
#
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#
#         for i in range(self.n_iters):
#             linear_pred = np.dot(X, self.weights) + self.bias
#             predictions = sigmoid(linear_pred)
#
#             dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
#             db = (1 / n_samples) * np.sum(predictions - y)
#
#             self.weights -= self.lr * dw
#             self.bias -= self.lr * db
#
#             # Compute and print loss every 100 iterations
#             if i % 100 == 0:
#                 loss = binary_cross_entropy(y, predictions)
#                 print(f"Iteration {i}, Loss: {loss:.4f}")
#
#     def predict(self, X):
#         linear_pred = np.dot(X, self.weights) + self.bias
#         y_pred = sigmoid(linear_pred)
#         return [1 if y > 0.5 else 0 for y in y_pred]
