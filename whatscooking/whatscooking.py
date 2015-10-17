#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Solve "What's Cooking?" Kaggle.

Usage:
    whatscooking [options] <train> <test>
    whatscooking -h | --help

Options:
    -d, --dimension D   dimension [default: 1000]
    --nmf               nmf.
    --lsa               lsa.
    --chi               chi2.
    -h, --help          Show help.
"""

from __future__ import print_function
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from time import time
import docopt
import json


def load_train_data(filename):
    X, Y = [], []
    with open(filename, "rb") as input_data:
        for sample in json.load(input_data):
            X.append(sample['ingredients'])
            Y.append(sample['cuisine'])
    return X, Y


def load_test_data(filename):
    X = []
    uids = []
    with open(filename, "rb") as input_data:
        for sample in json.load(input_data):
            X.append(sample['ingredients'])
            uids.append(sample['id'])
    return X, uids


def analyze(ingredients):
    terms = set()
    for ingredient in ingredients:
        terms.add(ingredient)
        # for sub_ingredient in ingredient.split():
        #     terms.add(sub_ingredient)
    return list(terms)


def train_classifier(clf, X_train, y_train, X_test, y_test):
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy_score: %0.3f" % score)

    clf_descr = str(clf).split('(')[0]
    return clf, score


def main():
    args = docopt.docopt(__doc__)

    print("Load data")
    X, y = load_train_data(args['<train>'])

    print("Vectorize data (tfidf)")
    vectorizer = TfidfVectorizer(analyzer=analyze)
    X = vectorizer.fit_transform(X)

    print("Encode labels")
    labels_encoder = LabelEncoder()
    y = labels_encoder.fit_transform(y)

    dimension = int(args['--dimension'])
    print("Reduce dimension to %i" % dimension)

    # Chi2 test
    reducer = None
    if args['--chi']:
        reducer = SelectKBest(chi2, k=dimension)
        X = reducer.fit_transform(X, y)

    print("Split train data")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.3
    )

    print("Train classifier")
    clf, score = train_classifier(
        LogisticRegression(),
        X_train,
        Y_train,
        X_test,
        Y_test)

    print("Run classifier on test data")
    X_test, uids = load_test_data(args['<test>'])
    X_test_vectorized = vectorizer.transform(X_test)

    if reducer is not None:
        X_test_vectorized = reducer.transform(X_test_vectorized)

    clf.fit(X, y)
    test_pred = clf.predict(X_test_vectorized)
    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        labels = labels_encoder.inverse_transform(test_pred)
        for uid, label in zip(uids, labels):
            print("%s,%s" % (uid, label), file=output)


if __name__ == "__main__":
    main()
