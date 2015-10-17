#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Solve "What's Cooking?" Kaggle.

Usage:
    whatscooking [--generate_data <train> <test>]
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from collections import Counter
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import docopt
import itertools
import json
import subprocess

import whatscooking.data


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


def generate_data(train_path, test_path):
    print("Load data")
    X, y, uids = load_train_data(train_path)
    X = map(analyze, X)

    X_unlabeled, uids_test = load_test_data(test_path)
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


class VW(object):
    def __init__(self, options):
        self.options = options
        self.model = None

    def fit(self, X, y=None):
        # Dump data to vowpal wabbit format
        if not isinstance(X, (str, unicode)):
            assert y is not None
            filename = "train_%s.vw" % len(X)
            output_data(filename, X, y)
        else:
            filename = X
        arguments = [
            "/usr/bin/vw",
            filename,
            "-f", "model",
            "--cache_file", "cache"
        ]
        arguments.extend(self.options)
        print("$", " ".join(arguments))
        try:
            subprocess.check_call(arguments)
        except subprocess.CalledProcessError:
            return None
        self.model = "model"

    def predict(self, X):
        # Dump data to vowpal wabbit format
        if not isinstance(X, (str, unicode)):
            filename = "train_%s.vw" % len(X)
            output_data(filename, X)
        else:
            filename = X
        arguments = [
            "/usr/bin/vw",
            "-t",
            "-i", self.model,
            "-p", "output",
            filename
        ]
        print("$", " ".join(arguments))
        subprocess.check_call(arguments)
        prediction = []
        with open("output", "rb") as input_prediction:
            for line in input_prediction.read().split():
                prediction.append(int(float(line)))
        return prediction


def compute_score(labels, prediction):
    return metrics.accuracy_score(labels, prediction, normalize=False)


OPTIONS = [
    ['--l1 0.1', '--l2 0.1', ''],
    ['--loss_function squared', '--loss_function logistic', '--loss_function hinge', '--loss_function quantile'],
    ['--sgd', '--adaptative', '--invariant', '--normalized'],
    ['-q', '--cubic'],
    ['--ngram 3', '--skip 3'],
    ['--spelling', ''],
    ['--passes 1', '--passes 2', '--passes 10', '--passes 100'],
    ['--log_multi 20', '--oaa 20', '--ect 20', '--nn 20']
]
# --bfgs: batch lbfgs instead of stochastic gradient descent
# --conjugate_gradient
# --ksvm
# --l1 L   vs --l2 L (try with 0 ?)
# --loss_function <loss>  (squared, logistic, hinge, quantile)
# --sgd --adaptative --invariant --normalized
# -q    quadratic features
# --ngram N
# --skip N
# --feature_limit
# --spelling
# --cubic or -q (argument _ ?)
# --passes N (number of training passes ?)

# Debug:
# --invert_hash arg

# Multiclass
# --log_multi k   vs.  --oaa k   vs. --ect k   vs.  --nn k


def main():
    args = docopt.docopt(__doc__)

    if args['--generate_data']:
        generate_data(args['<train>'], args['<test>'])

    # Files
    full_data = "full_train.txt"
    to_predict = "to_predict.txt"
    with open("labels.json", "rb") as input_labels:
        labels_encoder = json.load(input_labels)
    with open("uids.txt", "rb") as input_uids:
        uids = map(int, input_uids)

    # Select best options
    arguments = [
        "--oaa", "20",
        "--passes", "1000"
    ]
    # Eval model with kfold
    total = 0
    score = 0
    for i in range(5):
        clf = VW(arguments)
        clf.fit("train_%i.txt" % i)
        prediction = clf.predict("test_x_%i.txt" % i)
        with open("test_y_%i.txt" % i, "rb") as input_labels:
            labels = map(int, input_labels)
        score += compute_score(labels, prediction)
        total += len(labels)
    score = float(score) / float(total)

    # Predict unlabeled data
    clf = VW(arguments)
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
