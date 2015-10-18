#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
import json

from whatscooking.vw import VW
from whatscooking.features import analyze


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
        VW.output_data("train_%i.txt" % i, X_train, Y_train)
        VW.output_data("test_x_%i.txt" % i, X_test, Y_test)
        with open("test_y_%i.txt" % i, "wb") as output:
            output.write("\n".join(map(str, Y_test)))

    # Full dataset and unlabeled data
    VW.output_data("full_train.txt", X, y)
    VW.output_data("to_predict.txt", X_unlabeled)
    with open("uids.txt", "wb") as output:
        output.write("\n".join(map(str, uids_test)))

    # Dump labels mapping
    with open("labels.json", "wb") as output:
        json.dump(dict(zip(
                labels_encoder.transform(labels_encoder.classes_),
                labels_encoder.classes_)),
            output)
