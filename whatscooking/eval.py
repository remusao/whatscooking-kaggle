#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Eval prediction

Usage:
    whatscooking [options] <labels> <prediction>
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from sklearn import metrics
import docopt


def main():
    args = docopt.docopt(__doc__)

    with open(args['<labels>'], "r") as input_labels:
        labels = map(int, input_labels.read().split())
    with open(args['<prediction>'], "r") as input_prediction:
        prediction = []
        for line in input_prediction.read().split():
            prediction.append(int(float(line)))

    score = metrics.accuracy_score(labels, prediction)
    print(score)


if __name__ == "__main__":
    main()
