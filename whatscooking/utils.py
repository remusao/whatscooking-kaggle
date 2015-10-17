#! /usr/bin/env python
# -*- coding: utf-8 -*-


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
    with open(filename, "rb") as input_data:
        for sample in json.load(input_data):
            X.append(sample['ingredients'])
    return X
