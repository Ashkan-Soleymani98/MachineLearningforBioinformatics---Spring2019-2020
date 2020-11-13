#!/usr/bin/python

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Utils


class DecisionTree:
    def __init__(self, max_depth, threshold, prune_confidence_level=0.95):
        self.max_depth = max_depth
        self.threshold = threshold
        self.X = None
        self.Y = None
        self.classes = None
        self.variable_types = []
        self.model = None
        self.prune_confidence_level = prune_confidence_level


    def get_IG(self, splits):
        # print([len(split) for split in splits])
        total_size = sum([len(split) for split in splits])
        score = 0
        for split in splits:
            split_entropy = 0
            split_size = len(split)
            for class_type in self.classes:
                # print(list(self.Y[split]).count(class_type))
                if split_size > 0 and (list(self.Y[split]).count(class_type) / split_size) != 0:
                    split_entropy += (list(self.Y[split]).count(class_type) / split_size) \
                                     * np.log2((list(self.Y[split]).count(class_type) / split_size))
            score += (split_size / total_size) * split_entropy
        return score

    def split_continuous_variable(self, variable_index, value, subset):
        splits = [[], []]
        for i, x in enumerate(self.X.iloc[subset, variable_index]):
            if x <= value:
                splits[0].append(subset[i])
            else:
                splits[1].append(subset[i])
        return splits

    # def split_discrete_variable(self, variable_index):
    #     splits = list()
    #     for category in np.unique(self.X.iloc[:, variable_index]):
    #         splits.append(list(self.X.index[self.X.iloc[:, variable_index] == category]))
    #     return splits

    def get_possible_continuous_thresholds(self, values):
        return np.append(np.unique(values), np.min(values) - 0.1)

    def find_best_split(self, subset):
        best_score, best_var_index, best_value, best_splits = -1, -1, -1, None
        for var_index in range(self.X.shape[1]):
            # print(self.X.iloc[subset, var_index])
            for possible_value in self.get_possible_continuous_thresholds(self.X.iloc[subset, var_index]):
                splits = self.split_continuous_variable(var_index, possible_value, subset)
                # print("------------")
                # print(self.get_IG(splits), var_index, possible_value)
                if self.get_IG(splits) > best_score:
                    best_score = self.get_IG(splits)
                    best_value = possible_value
                    best_var_index = var_index
                    best_splits = splits
            # print("------------")
            # print("------------")
            # print(best_score, best_var_index, best_value)
        return {"score": best_score, "variable index": best_var_index, "value": best_value, "splits": best_splits,
                "state": "non-terminal"}

    def find_outcome(self, subset):
        target_values = list(self.Y[subset])
        return {"state": "terminal", "label": max(set(target_values), key=target_values.count)}

    def check_termination_subset(self, subset):
        for class_type in self.classes:
            if list(self.Y[subset]).count(class_type) >= self.threshold * len(self.Y[subset]):
                return True
        return False

    def recursive_find_tree(self, node, depth):
        splits = node["splits"]
        if depth >= self.max_depth:
            node['left'] = self.find_outcome(splits[0])
            node['right'] = self.find_outcome(splits[1])
            return
        if self.check_termination_subset(splits[0]):
            node['left'] = self.find_outcome(splits[0])
        else:
            node['left'] = self.find_best_split(splits[0])
            self.recursive_find_tree(node['left'], depth + 1)
        if self.check_termination_subset(splits[1]):
            node['right'] = self.find_outcome(splits[1])
        else:
            node['right'] = self.find_best_split(splits[1])
            self.recursive_find_tree(node['right'], depth + 1)

    def build_tree(self, subset):
        root = self.find_best_split(subset)
        self.recursive_find_tree(root, 1)
        return root

    def debug_tree(self, node, depth=0):
        if node["state"] == "non-terminal":
            print("%s[%s <= %.3f, score=%.6f]" % (depth * ' ', (self.X.columns[node['variable index']]), node['value'], node['score']))
            self.debug_tree(node['left'], depth + 1)
            self.debug_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', node))

    def initialize_model(self, X, Y):
        self.X = X
        self.Y = Y
        self.classes = list(set(Y))
        self.variable_types = []
        for var_index in range(X.shape[1]):
            # print(X.columns[var_index])
            # print(np.unique(X.iloc[:, var_index]))
            # print(len(np.unique(X.iloc[:, var_index])))
            if len(np.unique(X.iloc[:, var_index])) <= 5:
                self.variable_types.append("discrete")
            else:
                self.variable_types.append("continuous")

    def fit(self, X, Y):
        self.initialize_model(X, Y)
        self.model = self.build_tree(range(len(X)))
        return self.model

    def recursive_predict(self, node, x_test_point):
        if x_test_point[node['variable index']] <= node['value']:
            node = node['left']
        else:
            node = node['right']
        if node['state'] == "terminal":
            return node['label']
        else:
            return self.recursive_predict(node, x_test_point)

    def predict_point(self, x_test_point):
        return self.recursive_predict(self.model, x_test_point)

    def predict(self, x_test):
        y_pred = x_test.apply(self.predict_point, axis=1)
        return y_pred

    def calculate_chi_square(self, node):
        total_size = sum([len(split) for split in node["splits"]])
        pivotal = 0
        for class_type in self.classes:
            class_size = sum([list(self.Y[split]).count(class_type) for split in node["splits"]])
            for split in node["splits"]:
                split_size = len(split)
                E = class_size * split_size / total_size
                O = list(self.Y[split]).count(class_type)
                pivotal += (E - O) ** 2 / E
        return pivotal, len(node["splits"])

    def chi_test(self, pivotal, m):
        # print((1 - stats.chi2.cdf(pivotal, m - 1)), (1 - self.prune_confidence_level))
        return (1 - stats.chi2.cdf(pivotal, m - 1)) < (1 - self.prune_confidence_level)

    def recursive_prune(self, node):
        if node["state"] == "terminal":
            return
        else:
            pivotal, m = self.calculate_chi_square(node)
            if self.chi_test(pivotal, m):
                self.recursive_prune(node["left"])
                self.recursive_prune(node["right"])
            else:
                node["state"] = "terminal"
                flatten_splits = [_ for sublist in node['splits'] for _ in sublist]
                node["label"] = self.find_outcome(flatten_splits)["label"]
                del node["splits"]
                del node["score"]
                del node["variable index"]
                del node["value"]
                del node["left"]
                del node["right"]
                return

    def prune(self):
        self.recursive_prune(self.model)





# Y = np.array([0, 0, 0, 1])
# splits = [[0, 1], [2, 3]]
#
# tree = DecisionTree(2, 0.8, [123], Y)
# print(tree.get_IG(splits))

# x, y = Utils.read_data()
# x, y = Utils.shuffle_data(x, y)
# x_train, x_test, y_train, y_test = Utils.split_data(x, y, 0.8)
# tree = DecisionTree(15, 0.95)
# tree.debug_tree(tree.fit(x_train, y_train))
# print(Utils.get_accuracy(y_test, tree.predict(x_test)))
# tree.prune()
# tree.debug_tree(tree.model)
#
# # print(y_test)
# # print(tree.predict(x_test))
# print(Utils.get_accuracy(y_test, tree.predict(x_test)))
# print(tree.variable_types)

# print(tree.find_best_split(range(len(x))))

# X1			X2			Y
# 2.771244718		1.784783929		0
# 1.728571309		1.169761413		0
# 3.678319846		2.81281357		0
# 3.961043357		2.61995032		0
# 2.999208922		2.209014212		0
# 7.497545867		3.162953546		1
# 9.00220326		3.339047188		1
# 7.444542326		0.476683375		1
# 10.12493903		3.234550982		1
# 6.642287351		3.319983761		1

# X = pd.DataFrame({"X1": [2.771244718, 1.728571309, 3.678319846, 3.961043357, 2.999208922, 7.497545867, 9.00220326
#     , 7.444542326, 10.12493903, 6.642287351], "X2": [1.784783929, 1.169761413, 2.81281357, 2.61995032, 2.209014212
#     , 3.162953546, 3.339047188, 0.476683375, 3.234550982, 3.319983761]})
#
# Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
#
# tree = DecisionTree(10, 0.8, X, Y)
# root = tree.build_tree(range(len(X)))
# # print(tree.find_best_split(range(len(X))))
# print("ads")
