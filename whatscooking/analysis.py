#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Analysis of "What's Cooking?" Kaggle.

Usage:
    whatscooking <train>
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from collections import defaultdict, Counter
import docopt
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
    uids = []
    with open(filename, "rb") as input_data:
        for sample in json.load(input_data):
            X.append(sample['ingredients'])
            uids.append(sample['id'])
    return X, uids


def analyze(ingredients):
    terms = set()
    for ingredient in ingredients:
        terms.add(ingredient)
        # for sub_ingredient in ingredient.split():
        #     terms.add(sub_ingredient)
    return list(terms)


def main():
    args = docopt.docopt(__doc__)

    print("Load data")
    X, y = load_train_data(args['<train>'])

    bigram = defaultdict(set)
    ingredients_count = Counter()
    for sample, label in zip(X, y):
        features = analyze(sample)
        for f1 in features:
            ingredients_count[f1] += 1
            for f2 in features:
                if f2 != f1:
                    bigram[tuple(sorted((f1, f2)))].add(label)

    print(ingredients_count.most_common(100))

    # Find bigram of ingredients found in only one country
    countries = set()
    singleton = {}
    multiples = {}
    for ingredient, countries in bigram.iteritems():
        if len(countries) == 1:
            country = list(countries)[0]
            singleton[ingredient] = country
            countries.add(country)
        else:
            multiples[ingredient] = list(countries)
    print(len(singleton), "singletons")
    print(len(countries), "countries with singletons")
    print(len(multiples), "multiples")
    print(countries)

    with open("counts.txt", "wb") as output:
        for ingredient, count in sorted(ingredients_count.items(), reverse=True, key=lambda c: c[1]):
            print(ingredient.encode("utf-8"), "\t", count, file=output)


if __name__ == "__main__":
    main()
