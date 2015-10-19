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
from sklearn import metrics
import docopt
import json

import whatscooking.data as data
import whatscooking.vw as vw
import whatscooking.stacking as stacking


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


def create_classifier():
    return stacking.Stacking([
        vw.VW(["--oaa", "20", "-b", "24", "--passes", "1", "--sort_features", "-q", "iq"]),
        vw.VW(["--log_multi", "20", "-b", "24", "--passes", "1", "--sort_features", "-q", "iq"]),
        vw.VW(["--ect", "20", "-b", "24", "--passes", "1", "--sort_features", "-q", "iq"]),
        vw.VW(["--oaa", "20", "--passes", "100", "--sort_features"]),
        vw.VW(["--log_multi", "20", "--passes", "100", "--sort_features"]),
        vw.VW(["--ect", "20", "--passes", "100", "--sort_features"])
    ])


def main():
    args = docopt.docopt(__doc__)

    if args['--generate_data']:
        data.generate_data(args['<train>'], args['<test>'])

    # Files
    full_data = "full_train.txt"
    to_predict = "to_predict.txt"
    with open("labels.json", "rb") as input_labels:
        labels_encoder = json.load(input_labels)
    with open("uids.txt", "rb") as input_uids:
        uids = map(int, input_uids)

    # Select best options
    # Eval model with kfold
    total = 0
    score = 0
    for i in range(5):
        clf = create_classifier()

        with open("test_y_%i.txt" % i, "rb") as input_labels:
            labels = map(int, input_labels)
        with open("test_y_%i.txt" % ((i + 1) % 5), "rb") as input_labels:
            stacking_labels = map(int, input_labels)
        clf.fit(
            X="train_%i.txt" % i,
            X_stacker="train_%i.txt" % ((i + 1) % 5),
            y=labels,
            y_stacker=stacking_labels
        )
        prediction = clf.predict("test_x_%i.txt" % i)
        score += compute_score(labels, prediction)
        total += len(labels)
    score = float(score) / float(total)
    print(score)

    # Predict unlabeled data
    clf = create_classifier()
    clf.fit(full_data, "train_0.txt")
    prediction = clf.predict(to_predict)
    print("BEST MODEL =>", score)

    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        for uid, label in zip(uids, prediction):
            print("%s,%s" % (uid, labels_encoder[str(label - 1)]), file=output)


if __name__ == "__main__":
    main()
