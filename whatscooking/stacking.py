
from __future__ import print_function
import numpy
from sklearn.linear_model import LogisticRegression


class Stacking(object):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.stacker = LogisticRegression()

    def fit(self, X, y, X_stacker, y_stacker):
        # Train classifier
        for classifier in self.classifiers:
            classifier.fit(X)
        self._fit_stacker(X_stacker, y_stacker)

    def _fit_stacker(self, X, y):
        predictions = numpy.transpose(numpy.array([
            classifier.predict(X)
            for classifier in self.classifiers
        ]))
        self.stacker.fit(predictions, y)

    def predict(self, X):
        predictions = numpy.transpose(numpy.array([
            classifier.predict(X)
            for classifier in self.classifiers
        ]))
        return self.stacker.predict(predictions)
