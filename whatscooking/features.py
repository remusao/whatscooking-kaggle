#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
from collections import defaultdict, Counter
import os
from nltk.stem import WordNetLemmatizer
import spacy.parts_of_speech as pos


def get_wordnet_pos(tag):
    if tag == pos.ADJ:
        return 'a'
    elif tag == pos.VERB:
        return 'v'
    elif tag == pos.NOUN:
        return 'n'
    elif tag == pos.ADV:
        return 'r'


from spacy.en import English, LOCAL_DATA_DIR
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
NLP = English(data_dir=data_dir)


EXTRA_TAGS = {
    u"arbol": "n",
    u"bertolli": "n",
    u"bok": "n",
    u"casa": "n",
    u"chai": "a",
    u"chiles": "n",
    u"choy": "n",
    u"cilantro": "n",
    u"dulce": "n",
    u"eau": "n",
    u"fleur": "n",
    u"Foods®": "n",
    u"framboise": "n",
    u"Frank's®": "n",
    u"gallo": "n",
    u"hanout": "n",
    u"hellmannâ€™": "n",
    u"Hellmann's®": "n",
    u"Johnsonville®": "n",
    u"JOHNSONVILLE®": "n",
    u"knorr": "n",
    u"knox": "n",
    u"leche": "n",
    u"maida": "n",
    u"menthe": "n",
    u"minicub": "n",
    u"mora": "n",
    u"ox": "n",
    u"oz": "n",
    u"paprika": "n",
    u"pepe": "n",
    u"Pillsbury™": "n",
    u"pimenton": "n",
    u"pimento": "n",
    u"pinto": "n",
    u"pro.activ": "a",
    u"ras": "n",
    u"sel": "n",
    u"smith": "n",
    u"tamari": "n",
    u"tangzhong": "n",
    u"Than": "n",
    u"Truvía®": "n",
    u"vegan": "n",
    u"verde": "n",
    u"viande": "n",
    u"vie": "n",
    u"yardlong": "n",
    u"Yoplait®": "n",
    u"yu": "n",
    u"za'atar": "n",
    u"zero": "n"
}


LETTER = frozenset("abcdefghijklmnopqrstuvwxyz")
LEMMATIZER = WordNetLemmatizer()
COUNTER = 1
def analyze(ingredients):
    # Show progress
    global COUNTER
    print(COUNTER)
    COUNTER += 1

    # Store namespaces of features (VW format)
    features = defaultdict(Counter)

    total_length = 0
    for ingredient in ingredients:
        # Remove non-ascii chars
        total_length += len(ingredient)
        # Analyze with SpaCy
        analyzed = NLP(ingredient)
        for token in analyzed:
            # Store each part of speech in its own namespace
            # Use lemmatized form.
            wordnet_pos = get_wordnet_pos(token.pos)
            if not wordnet_pos:
                wordnet_pos = EXTRA_TAGS.get(token.text)
            if wordnet_pos:
                features[wordnet_pos][LEMMATIZER.lemmatize(token.text, wordnet_pos).lower()] += 1

    # Add some extra features
    features["e"]["num_noun"] = len(features["n"])
    features["e"]["num_verb"] = len(features["v"])
    features["e"]["num_adj"] = len(features["a"])
    features["e"]["num_adv"] = len(features["r"])
    features["e"]["avg_length"] = total_length / float(len(ingredients))
    features["e"]["n_ingredients"] = len(ingredients)
    return features
