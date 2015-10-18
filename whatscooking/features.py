#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
from collections import defaultdict


LETTER = frozenset("abcdefghijklmnopqrstuvwxyz")
def analyze(ingredients):
    features = defaultdict(dict)
    total_length = 0
    for ingredient in ingredients:
        # Remove non-ascii chars
        ingredient = ''.join([l if l in LETTER else ' ' for l in ingredient.lower()])
        splitted = ingredient.split()
        total_length += len(ingredient)
        features["f"][ingredient] = 1.0
        features["i"][splitted[-1]] = 1.0
        for sub_ingredient in splitted[:-1]:
            features["q"][sub_ingredient] = 1.0
    features["e"]["n_qualif"] = len(features["q"])
    features["e"]["n_base"] = len(features["i"])
    features["e"]["avg_length"] = total_length / float(len(ingredients))
    features["e"]["n_ingredients"] = len(ingredients)
    return features
