#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Solve "What's Cooking?" Kaggle.

Usage:
    whatscooking <train> <test>
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import ProjectedGradientNMF
from sklearn.decomposition import DictionaryLearning

from sklearn import ensemble

import docopt
import json
import numpy
import nltk


def load_train_data(filename):
    X, Y, uids = [], [], []
    with open(filename, "rb") as input_data:
        for sample in json.load(input_data):
            X.append(sample['ingredients'])
            Y.append(sample['cuisine'])
            uids.append(sample['id'])
    return X, Y, uids


def load_test_data(filename):
    X = []
    uids = []
    with open(filename, "rb") as input_data:
        for sample in json.load(input_data):
            X.append(sample['ingredients'])
            uids.append(sample['id'])
    return X, uids


LETTER = frozenset("abcdefghijklmnopqrstuvwxyz")
def analyze(ingredients):
    features = []
    for ingredient in ingredients:
        # Remove non-ascii chars
        ingredient = ''.join([l if l in LETTER else ' ' for l in ingredient.lower()])
        splitted = ingredient.split()
        features.append(ingredient)
        features.extend(splitted)
    return features


def compute_score(labels, prediction):
    return metrics.accuracy_score(labels, prediction, normalize=False)


def main():
    args = docopt.docopt(__doc__)

    train_path = args["<train>"]
    test_path = args["<test>"]

    print("Load data")
    X_raw, y, uids = load_train_data(train_path)
    vectorizer = TfidfVectorizer(analyzer=analyze)
    X = vectorizer.fit_transform(X_raw).toarray()

    print("Encode labels")
    labels_encoder = LabelEncoder()
    y = labels_encoder.fit_transform(y)

    # Classification
    print(X.shape)
    clf_base = ensemble.RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_features=None,
        random_state=42,
        n_jobs=-1)
    clf = ensemble.AdaBoostClassifier(
        clf_base,
        n_estimators=4,
        random_state=10)

    print("Split train data")
    score = 0
    samples = 0
    skf = StratifiedKFold(y, 5)
    for i, (train, test) in enumerate(skf):
        print("%ith fold" % (i + 1))
        X_train = X[train]
        Y_train = [y[j] for j in train]
        X_test = X[test]
        Y_test = [y[j] for j in test]

        print("Train")
        print(X_train.shape)
        clf.fit(X_train, Y_train)
        print("Predict")
        print(X_test.shape)
        predictions = clf.predict(X_test)

        score += compute_score(Y_test, predictions)
        samples += len(Y_test)

    print("Score:", score / float(samples))
    return

    X_unlabeled, uids_test = load_test_data(test_path)
    X_unlabeled = map(analyze, X_unlabeled)

    # Predict unlabeled data
    clf = vw.VW(arguments)
    clf.fit(full_data)
    prediction = clf.predict(to_predict)
    print()
    print("BEST MODEL =>", score)
    print(arguments)

    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        for uid, label in zip(uids, prediction):
            print("%s,%s" % (uid, labels_encoder[str(label - 1)]), file=output)


if __name__ == "__main__":
    main()
