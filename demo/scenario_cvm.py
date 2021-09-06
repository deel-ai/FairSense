import numpy as np
import pandas as pd
from libfairness.data_management.factory import create_fairness_problem
from libfairness.indices.cvm import compute_cvm
from libfairness.visualization.cvm_visu import visu_cvm
from sklearn import tree
from sklearn.datasets import load_iris


def cvm_scenario_one():
    # Setup + Data
    samples = 10000
    cols = ["X_{}".format(i) for i in range(5)]
    x = pd.DataFrame(np.random.normal(size=(samples, 5)), columns=cols)
    y = pd.DataFrame()
    y["Y"] = x["X_0"] + 2 * x["X_1"] - x["X_2"] + np.random.normal(0, 0.005, len(x))

    # Use Case
    my_problem = create_fairness_problem(inputs=x, outputs=y.to_numpy())
    compute_cvm(my_problem)
    visu_cvm(my_problem)


def cvm_scenario_two():
    iris = load_iris()
    X, y = iris.data, iris.target

    y_train_0 = y[:40]
    y_test_0 = y[40:50]
    y_train_1 = y[50:90]
    y_test_1 = y[90:100]
    y_train_2 = y[100:140]
    y_test_2 = y[140:150]

    y_train = np.concatenate((y_train_0, y_train_1, y_train_2), axis=0)
    y_test = np.concatenate((y_test_0, y_test_1, y_test_2), axis=0)

    X_train_0 = X[:40, :]
    X_test_0 = X[40:50, :]
    X_train_1 = X[50:90, :]
    X_test_1 = X[90:100, :]
    X_train_2 = X[100:140, :]
    X_test_2 = X[140:150, :]

    X_train = np.concatenate((X_train_0, X_train_1, X_train_2), axis=0)
    X_test = np.concatenate((X_test_0, X_test_1, X_test_2), axis=0)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    res = clf.predict(X_test)

    my_problem = create_fairness_problem(inputs=X_test, outputs=res)
    compute_cvm(my_problem)
    visu_cvm(my_problem)


if __name__ == "__main__":
    cvm_scenario_one()
    cvm_scenario_two()
