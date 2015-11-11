#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import regex as re


CORRECTION = {
    "mayonaise": "mayonnaise",
    "cheese,": "cheese",
    "chees": "cheese",
    "corn,": "corn",
    "crabmeat": "crab"
}


STOPWORDS = frozenset(nltk.corpus.stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
def analyze(ingredients):
    features = []
    for ingredient in ingredients:
        # Lemmatize words
        splitted = map(
            LEMMATIZER.lemmatize,
            filter(None, re.split(ur"\W+", ingredient.lower())))
        corrected = map(lambda f: CORRECTION.get(f, f), splitted)
        filtered = filter(lambda t: (t not in STOPWORDS) and len(t) > 2, corrected)
        features.append('_'.join(filtered))
        features.extend(filtered)
    return " ".join(features)
