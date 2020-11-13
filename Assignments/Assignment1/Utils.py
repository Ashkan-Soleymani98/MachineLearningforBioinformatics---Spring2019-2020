#!/usr/bin/python

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def read_data():
    dataframe = pd.read_csv('data/heart.csv')
    X, Y = dataframe.drop('target', axis=1), dataframe['target']
    # print(dataframe)
    return X, Y

def shuffle_data(X, Y):
    # print(len(X), len(Y))
    perm = np.random.permutation(len(X))
    X, Y = X.loc[perm, :], Y[perm]
    X, Y = X.reset_index(drop=True), Y.reset_index(drop=True)
    return X, Y

def split_data(X, Y, train_ratio):
    split_border = int(np.floor(train_ratio * len(X)))
    X_train, X_test = X.loc[0:split_border, :], X.loc[split_border + 1:, :]
    Y_train, Y_test = Y[0:split_border + 1], Y[split_border + 1:]
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    Y_train, Y_test = Y_train.reset_index(drop=True), Y_test.reset_index(drop=True)
    return X_train, X_test, Y_train, Y_test

def get_confusion_matrix(Y_test, Y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i, j in zip(Y_test, Y_pred):
        if i == 1 and j == 1:
            TP += 1
        elif i == 1 and j == 0:
            FN += 1
        elif i == 0 and j == 1:
            FP += 1
        elif i == 0 and j == 0:
            TN += 1
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

def get_accuracy(Y_test, Y_pred):
    conf_matrix = get_confusion_matrix(Y_test, Y_pred)
    return (conf_matrix["TP"] + conf_matrix["TN"]) \
           / (conf_matrix["TP"] + conf_matrix["TN"] + conf_matrix["FP"] + conf_matrix["FN"])

def get_evaluation_metrics(Y_test, Y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    P, N = 0, 0
    for i, j in zip(Y_test, Y_pred):
        if i == 1 and j == 1:
            TP += 1
            P += 1
        elif i == 1 and j == 0:
            FN += 1
            P += 1
        elif i == 0 and j == 1:
            FP += 1
            N += 1
        elif i == 0 and j == 0:
            TN += 1
            N += 1
    metrics = {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "P":P, "N":N}
    return metrics

def get_classification_report(Y_test, Y_pred):
    metrics = get_evaluation_metrics(Y_test, Y_pred)
    report = {}
    report["Recall"] = metrics["TP"] / metrics["P"]
    report["Specificity"] = metrics["TN"] / metrics["N"]
    report["Precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
    report["F1 Score"] = 2 * metrics["TP"] / (2 * metrics["TP"] + metrics["FP"] + metrics["FN"])
    report["Accuracy"] = get_accuracy(Y_test, Y_pred)
    return report

def calculate_tValue(distribution1, distribution2):
    s = np.sqrt((distribution1.var(ddof=1) + distribution2.var(ddof=1)) / 2) * np.sqrt(2 / len(distribution1))
    t = (distribution1.mean() - distribution2.mean()) / s
    return t, 2 * len(distribution1) - 2

def ttest(distribution1, distribution2):
    t, df = calculate_tValue(distribution1, distribution2)
    return 1 - stats.t.cdf(t, df=df)


# x, y = read_data()

# print(x)
# print(y)


# x, y = shuffle_data(x, y)

# print(x.iloc[:, 0])
# print(y[180])

# x_train, x_test, y_train, y_test = split_data(x, y, 0.8)
# print(x.loc[x["ca"] == 4, :])
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
# print(list(x.index[x.iloc[:, 12] == 3]))
#
# print(set(y_test))
# print(x.shape)
# print(x.columns)

# a = np.array([1, 2])
# print(np.append(a, 3))


