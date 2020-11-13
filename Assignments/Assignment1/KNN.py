#!/usr/bin/python

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import Utils

class KNN:
    def __init__(self, K, normalize=True):
        self.K = K
        self.normalize = normalize
        self.X = None
        self.Y = None
        self.mean = None
        self.std = None
        self.classes = None

    def initialize_model(self, X, Y):
        self.X = X
        self.Y = Y
        self.classes = list(set(Y))

    def normalize_data(self):
        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = (self.X - self.X.mean()) / self.X.std()

    def fit(self, X, Y):
        self.initialize_model(X, Y)
        if self.normalize:
            self.normalize_data()

    def get_KNN_indices(self, point):
        if self.normalize:
            point = (point - self.mean) / self.std
        distances = self.X.apply(lambda _: np.linalg.norm(_ - point), axis=1)
        distances = distances.sort_values(ascending=True)
        # print(distances)
        return list(distances.index[:self.K])

    def predict_point(self, point):
        near_neighbours = self.get_KNN_indices(point)
        votes = [list(self.Y[near_neighbours]).count(class_type) for class_type in self.classes]
        # print(votes)
        return list(self.classes)[np.argmax(votes)]

    def predict(self, x_test):
        y_pred = x_test.apply(self.predict_point, axis=1)
        return y_pred

# x, y = Utils.read_data()
# x, y = Utils.shuffle_data(x, y)
# x_train, x_test, y_train, y_test = Utils.split_data(x, y, 0.8)
# knn = KNN(10)
# knn.fit(x_train, y_train)
# # print(x_test.loc[0, :])
# # print(knn.get_KNN_indices(x_test.loc[1, :]))
# print(Utils.get_accuracy(y_test, knn.predict(x_test)))
# print(knn.predict(x_test))


