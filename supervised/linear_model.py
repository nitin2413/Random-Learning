from sklearn.model_selection import train_test_split
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
import opendatasets as od
import pandas as pd
import numpy as np

class LinearRegression:

      def __init__(self, lr = 0.1 , n_iters = 1000):
          self.lr = lr
          self.n_iters = n_iters
          self.weights = None
          self.bias = None

      def fit(self,X,y):
          n_samples , n_features = X.shape
          self.weights = np.zeros(n_features)#It creates a NumPy array filled with zero also initialise weights with zero
          self.bias = 0

          for _ in range(self.n_iters):
              y_pred = np.dot(X,self.weights) + self.bias
              # np.dot product include submission
              # X.T signals to (x tranpose) without T it will show dimensional error
              dw = (1/n_samples) * np.dot(X.T,(y_pred - y))
              db = (1/n_samples) * np.sum(y_pred - y)

              #calculation of new weights and bias
              self.weights = self.weights - self.lr * dw
              self.bias = self.bias - self.lr * db


      def predict(self,X):
          y_pred = np.dot(X, self.weights) + self.bias
          return y_pred


