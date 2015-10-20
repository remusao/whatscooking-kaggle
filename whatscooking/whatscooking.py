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
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
import docopt
import json
import xgboost as xgb
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
COUNTER = 0
def analyze(ingredients):
    global COUNTER
    print("Analyze", COUNTER)
    COUNTER += 1
    features = Counter()
    total_length = 0
    for ingredient in ingredients:
        # Remove non-ascii chars
        ingredient = ''.join([l if l in LETTER else ' ' for l in ingredient.lower()])
        splitted = ingredient.split()
        total_length += len(ingredient)
        features[ingredient] = 2.0
        features[splitted[-1]] += 1.0
        for sub_ingredient in splitted[:-1]:
            features[sub_ingredient] += 0.5
    # features["n_features"] = len(features)
    # features["avg_length"] = total_length / float(len(ingredients))
    features["n_ingredients"] = len(ingredients)
    return features


def compute_score(labels, prediction):
    return metrics.accuracy_score(labels, prediction, normalize=False)


def main():
    args = docopt.docopt(__doc__)

    train_path = args["<train>"]
    test_path = args["<test>"]

    print("Load data")
    X_raw, y, uids = load_train_data(train_path)

    print("Encode labels")
    labels_encoder = LabelEncoder()
    y = labels_encoder.fit_transform(y)

    # Classification
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=analyze)),
        ('model', LogisticRegression())
    ])

    print("Split train data")
    score = 0
    samples = 0
    skf = StratifiedKFold(y, 5)
    for i, (train, test) in enumerate(skf):
        X_train = (X_raw[j] for j in train)
        Y_train = [y[j] for j in train]
        X_test = (X_raw[j] for j in test)
        Y_test = [y[j] for j in test]

        clf.fit_transform(X_train, Y_train)
        predictions = clf.predict(X_test)

        score += compute_score(Y_test, predictions)
        samples += len(X)

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
