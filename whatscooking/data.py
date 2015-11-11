#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import json


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
