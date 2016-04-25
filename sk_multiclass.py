#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage:
    sk_multiclass.py [--lr|--lsvc|--svc]
    sk_multiclass.py -h | --help
Options:
    --lr    LogisticRegression
    --lsvc  LinearSVC
    --svc   SVC
"""

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier, \
    OneVsRestClassifier, OutputCodeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from docopt import docopt


def main():
    # Parse args
    args = docopt(__doc__)
    lr = args['--lr']
    lsvc = args['--lsvc']
    svc = args['--svc']
    if lr:
        bc = LogisticRegression()
    elif lsvc:
        bc = LinearSVC(C=1.0, random_state=0)
    elif svc:
        bc = SVC(C=1.0, random_state=0)
    else:
        bc = LogisticRegression()

    # Apply PCA
    iris = load_iris()
    X, y = iris.data, iris.target
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.fit_transform(X)

    # multiclass and multilabel algorithms
    ovr = OneVsRestClassifier(bc).fit(X, y)
    ovo = OneVsOneClassifier(bc).fit(X, y)
    oc = OutputCodeClassifier(bc).fit(X, y)

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
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(titles[idx])

    plt.show()

if __name__ == '__main__':
    main()
