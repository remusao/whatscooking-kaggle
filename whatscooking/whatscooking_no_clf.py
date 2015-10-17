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
from collections import Counter
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
import numpy


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


def analyze(ingredients, split=True):
    terms = set()
    for ingredient in ingredients:
        terms.add(ingredient)
        if split:
            for sub_ingredient in ingredient.split():
                terms.add(sub_ingredient)
    return list(terms)


def add_features(X, X_vec, ingredient2country, countries):
    return X_vec
    X_res = []
    for (features, sample) in zip(X, X_vec):
        extra_features = []
        count = {c: 0 for c in countries}
        for feature in features:
            for country in ingredient2country.get(feature, []):
                count[country] += 1
        for c in countries:
            extra_features.append(count[c])
        X_res.append(numpy.concatenate(sample, np.array(extra_features)))
    return numpy.array(X_res)


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
    X, y, uids = load_train_data(args['<train>'])

    all_countries = list(set(y))
    with open("singletons.json", "rb") as input_singletons:
        singletons = json.load(input_singletons)
    with open("bigrams_singletons.json", "rb") as input_singletons:
        bigrams_singletons = {
            tuple(key): label
            for key, label in json.load(input_singletons)
        }
    with open("ingredient2country.json", "rb") as input_ingredient2country:
        ingredient2country = json.load(input_ingredient2country)

    print("Run manual classifier on test data")
    X_test, test_uids = load_test_data(args['<test>'])
    labels = {}
    X_remaining, uids_remaining = [], []
    for sample, uid in zip(X_test, test_uids):
        countries = set()
        features = analyze(sample, split=False)
        for f1 in features:
            if f1 in singletons:
                countries.add(singletons[f1])
            else:
                for f2 in features:
                    key = tuple(sorted((f1, f2)))
                    if f2 != f1 and key in bigrams_singletons:
                        countries.add(bigrams_singletons[key])
        if len(countries) == 1:
            labels[uid] = list(countries)[0]
        else:
            X_remaining.append(sample)
            uids_remaining.append(uid)

    print("Manually classified:", len(labels))
    print("Remaining samples:", len(X_remaining))

    print("Classify remaining data")
    print("Vectorize data (tfidf)")
    vectorizer = TfidfVectorizer(analyzer=analyze)
    X_vec = vectorizer.fit_transform(X)
    X_vec = add_features(X, X_vec, ingredient2country, all_countries)

    print("Encode labels")
    labels_encoder = LabelEncoder()
    y = labels_encoder.fit_transform(y)

    dimension = int(args['--dimension'])
    print("Reduce dimension to %i" % dimension)

    # Chi2 test
    reducer = None
    if args['--chi']:
        reducer = SelectKBest(chi2, k=dimension)
        X_vec = reducer.fit_transform(X_vec, y)

    print("Split train data")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vec, y, test_size=0.3
    )

    print("Train classifier")
    clf, score = train_classifier(
        LogisticRegression(),
        X_train,
        Y_train,
        X_test,
        Y_test)

    # Classify remaining samples
    X_test_vec = vectorizer.transform(X_remaining)
    X_test_vec = add_features(X_remaining, X_test_vec, ingredient2country, all_countries)
    if reducer is not None:
        X_test_vec = reducer.transform(X_test_vec)

    clf.fit(X_vec, y)
    test_pred = labels_encoder.inverse_transform(clf.predict(X_test_vec))

    for prediction, uid in zip(test_pred, uids_remaining):
        labels[uid] = prediction

    ordered_labels = [(uid, labels[uid]) for uid in test_uids]

    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        for uid, label in ordered_labels:
            print("%s,%s" % (uid, label), file=output)


if __name__ == "__main__":
    main()
