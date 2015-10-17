#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Solve "What's Cooking?" Kaggle.

Usage:
    whatscooking
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from sklearn import metrics
import docopt
import json
import subprocess
import os
import sys


def train_classifier(data, options):
    arguments = [
        "/usr/bin/vw",
        data,
        "-f", "model",
        "--cache_file", "cache"
    ]
    arguments.extend(options)
    print("$", " ".join(arguments))
    try:
        subprocess.check_call(arguments)
    except subprocess.CalledProcessError:
        return None
    return "model"


def predict(model, data):
    arguments = [
        "/usr/bin/vw",
        "-t",
        "-i", model,
        "-p", "output",
        data
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

    # Files
    full_data = "full_train.txt"
    train = "train.txt"
    test_x = "test_x.txt"
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
    for i in range(3):
        model = train_classifier("train_%i.txt" % i, arguments)
        if model is None:
            continue
        prediction = predict(model, "test_x_%i.txt" % i)
        with open("test_y_%i.txt" % i, "rb") as input_labels:
            labels = map(int, input_labels)
        score += compute_score(labels, prediction)
        total += len(labels)
    score = float(score) / float(total)

    # Predict unlabeled data
    model = train_classifier(full_data, arguments)
    prediction = predict(model, to_predict)
    print()
    print("BEST MODEL =>", score)
    print(arguments)

    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        for uid, label in zip(uids, prediction):
            print("%s,%s" % (uid, labels_encoder[str(label - 1)]), file=output)


if __name__ == "__main__":
    main()
