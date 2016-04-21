#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)
    y_uniq = map(np.str, np.unique(y_test))
    print classification_report(y_test, pred,
                                target_names=y_uniq)

if __name__ == '__main__':
    main()
