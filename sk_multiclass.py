#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier, \
    OneVsRestClassifier, OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


def main():
    # We only take the first two features.
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target

    # multiclass and multilabel algorithms
    ovr = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
    ovo = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y)
    oc = OutputCodeClassifier(LinearSVC(random_state=0)).fit(X, y)

    # create a mesh to plot in
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # titles
    titles = ['OneVsRest', 'OneVsOne', 'OutputCode']

    # plot boundary
    for idx, clf in enumerate((ovr, ovo, oc)):
        # Prepare a canvas
        plt.subplot(2, 2, idx + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # plot boundary
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

        # plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(titles[idx])

    plt.show()

if __name__ == '__main__':
    main()
