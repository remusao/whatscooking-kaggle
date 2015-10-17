#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
from collections import Counter


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
