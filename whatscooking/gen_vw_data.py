#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Solve "What's Cooking?" Kaggle.

Usage:
    whatscooking [options] <train> <test>
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from collections import Counter
import docopt
import json
import itertools


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


def make_line(label, features):
    def normalize_feature((feature, score)):
        feature = feature.replace(' ', '_')
        return "%s:%s" % (feature, score)
    if label is None:
        label = 1
    return "%s | %s" % (
        label,
        " ".join(map(normalize_feature, features.items()))
    )


def output_data(filename, X, y=[]):
    print(filename)
    with open(filename, "wb") as output:
        for namespaces, response in itertools.izip_longest(X, y, fillvalue=None):
            # for namespace in namespaces:
            #     clf.add_namespace(namespace)
            print(make_line(response, features=namespaces).encode("utf-8"), file=output)


def main():
    args = docopt.docopt(__doc__)

    print("Load data")
    X, y, uids = load_train_data(args['<train>'])
    X = map(analyze, X)

    X_unlabeled, uids_test = load_test_data(args['<test>'])
    X_unlabeled = map(analyze, X_unlabeled)

    print("Encode labels")
    labels_encoder = LabelEncoder()
    y = labels_encoder.fit_transform(y)
    y = map(lambda l: l + 1, y)

    print("Split train data")
    skf = StratifiedKFold(y, 5)
    for i, (train, test) in enumerate(skf):
        X_train = (X[j] for j in train)
        Y_train = [y[j] for j in train]
        X_test = (X[j] for j in test)
        Y_test = [y[j] for j in test]

        # Dump splitted dataset
        output_data("train_%i.txt" % i, X_train, Y_train)
        output_data("test_x_%i.txt" % i, X_test)
        with open("test_y_%i.txt" % i, "wb") as output:
            output.write("\n".join(map(str, Y_test)))

    # Full dataset and unlabeled data
    output_data("full_train.txt", X, y)
    output_data("to_predict.txt", X_unlabeled)
    with open("uids.txt", "wb") as output:
        output.write("\n".join(map(str, uids_test)))

    # Dump labels mapping
    with open("labels.json", "wb") as output:
        json.dump(dict(zip(
                labels_encoder.transform(labels_encoder.classes_),
                labels_encoder.classes_)),
            output)


if __name__ == "__main__":
    main()
