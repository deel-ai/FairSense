from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

def get_data():
    # data
    iris = load_iris()
    X, y = iris.data, iris.target

    # labels
    y_train_0 = y[:40]
    y_test_0 = y[40:50]
    y_train_1 = y[50:90]
    y_test_1 = y[90:100]

    y_train = np.concatenate((y_train_0,y_train_1),axis=0)
    y_test = np.concatenate((y_test_0,y_test_1),axis=0)

    # inputs
    X_train_0 = X[:40,:]
    X_test_0 = X[40:50,:]
    X_train_1 = X[50:90,:]
    X_test_1 = X[90:100,:]

    X_train = np.concatenate((X_train_0,X_train_1),axis=0)
    X_test = np.concatenate((X_test_0,X_test_1),axis=0)

    # model
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    res = clf.predict(X_test)

    return X_test, y_test, res

def f_for_test(x):
    # data
    iris = load_iris()
    X, y = iris.data, iris.target

    # labels
    y_train_0 = y[:40]
    y_test_0 = y[40:50]
    y_train_1 = y[50:90]
    y_test_1 = y[90:100]

    y_train = np.concatenate((y_train_0,y_train_1),axis=0)
    y_test = np.concatenate((y_test_0,y_test_1),axis=0)

    # inputs
    X_train_0 = X[:40,:]
    X_test_0 = X[40:50,:]
    X_train_1 = X[50:90,:]
    X_test_1 = X[90:100,:]

    X_train = np.concatenate((X_train_0,X_train_1),axis=0)
    X_test = np.concatenate((X_test_0,X_test_1),axis=0)

    # model
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    return clf.predict(x)

