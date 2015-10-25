#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
import regex as re


CORRECTION = {
    "mayonaise": "mayonnaise",
    "cheese,": "cheese",
    "chees": "cheese",
    "corn,": "corn",
    "crabmeat": "crab"
}


with open("features.txt", "rb") as input_features:
    FEATURES = frozenset(input_features.read().split("\n"))
LEMMATIZER = WordNetLemmatizer()
def analyze(ingredients):
    features = defaultdict(dict)
    for ingredient in ingredients:
        # Remove non-ascii chars
        # normalized = unidecode.unidecode(ingredient)
        normalized = re.sub(ur"\([^)]*\)", "", ingredient).strip().lower()
        normalized = re.sub(ur"\s+", " ", normalized)
        splitted = map(LEMMATIZER.lemmatize, re.split(ur"\s+", normalized))
        corrected = filter(lambda f: f in FEATURES, map(lambda f: CORRECTION.get(f, f), splitted))
        if not corrected:
            continue
        features[""][" ".join(corrected)] = 1.0
        for sub_ingredient in corrected:
            features[""][sub_ingredient] = 1.0
    return features
