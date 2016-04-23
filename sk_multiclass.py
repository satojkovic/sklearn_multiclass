#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


def main():
    # We only take the first two features.
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    pred = clf.fit(X_train, y_train).predict(X_test)
    y_uniq = map(np.str, np.unique(y_test))
    print classification_report(y_test, pred,
                                target_names=y_uniq)

    # plot boundary
    h = .02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

    # plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

if __name__ == '__main__':
    main()
