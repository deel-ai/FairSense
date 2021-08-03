import unittest

import numpy as np
import pandas as pd
from libfairness.data_management.factory import create_fairness_problem
from libfairness.indices.cvm import compute_cvm
from sklearn import tree
from sklearn.datasets import load_iris


class TestCVM(unittest.TestCase):

    def test_cvm(self):
        # data
        iris = load_iris()
        X, y = iris.data, iris.target

        # labels
        y_train_0 = y[:40]
        y_test_0 = y[40:50]
        y_train_1 = y[50:90]
        y_test_1 = y[90:100]
        y_train_2 = y[100:140]
        y_test_2 = y[140:150]

        y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)
        y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)

        # inputs
        X_train_0 = X[:40, :]
        X_test_0 = X[40:50, :]
        X_train_1 = X[50:90, :]
        X_test_1 = X[90:100, :]
        X_train_2 = X[100:140, :]
        X_test_2 = X[140:150, :]

        X_train = np.concatenate((X_train_0, X_train_1, X_train_2), axis=0)
        X_test = np.concatenate((X_test_0, X_test_1, X_test_2), axis=0)

        # model
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        res = clf.predict(X_test)

        # fairness problem
        my_problem = create_fairness_problem(inputs=X_test, outputs=res)
        compute_cvm(my_problem)

        result = my_problem.get_result().to_numpy()
        result_hard = np.array([[0.09232480533926586, 0.0], [0.0, 0.0], [
            0.5461624026696329, 0.07341490545050056], [0.3592880978865406, 0.0]])
        self.assertTrue(np.array_equal(result, result_hard))


if __name__ == '__main__':
    unittest.main()
