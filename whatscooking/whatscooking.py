#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Solve "What's Cooking?" Kaggle.

Usage:
    whatscooking <train> <test>
    whatscooking -h | --help

Options:
    -h, --help          Show help.
"""

from __future__ import print_function
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from collections import Counter, defaultdict
import docopt
import json
import numpy
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(labels, cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(20)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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


LETTER = frozenset("abcdefghijklmnopqrstuvwxyz")
LEMMATIZER = WordNetLemmatizer()
def analyze(ingredients):
    features = []
    for ingredient in ingredients:
        # Remove non-ascii chars
        ingredient = ''.join([l if l in LETTER else ' ' for l in ingredient.lower()])
        splitted = ingredient.split()
        # features.append(ingredient)
        for sub in splitted:
            features.extend(LEMMATIZER.lemmatize(sub))
    return features


def compute_score(labels, prediction):
    return metrics.accuracy_score(labels, prediction, normalize=False)


class BayesienClassifier(object):

    def _build_index(self, X, y):
        # Count terms
        self.index = defaultdict(Counter)
        for sample, label in zip(X, y):
            self.index[label].update(sample)

        # Merge all indexes in a global one
        self.global_index = {}
        for terms in self.index.itervalues():
            self.global_index.update(terms)

        # Create ids for each term
        self.term2id = dict(zip(self.global_index, range(len(self.global_index))))

    def fit(self, X, y):
        self._build_index(X, y)

        # Compute probability P(term | label)
        topics = numpy.zeros((len(self.global_index), len(self.index)))
        total_terms = sum(self.global_index.itervalues())
        term2id = self.term2id
        for label, terms in self.index.iteritems():
            for term, count in terms.iteritems():
                term_id = term2id[term]
                term_proba = float(count) / float(total_terms)
                # Update topic array
                topics[term_id, label] = term_proba
            # Normalize label's column so that it sums to 1
            topics[:, label] /= numpy.sum(topics[:, label])
        print(topics.shape)
        self.topics = topics

    def predict(self, X):
        predictions = []
        for sample in X:
            vectorized = numpy.zeros((1, len(self.global_index)))
            for term in sample:
                if term in self.term2id:
                    term_id = self.term2id[term]
                    vectorized[0, term_id] = 1
                else:
                    # print(term)
                    pass
            predictions.append(numpy.argmax(numpy.dot(vectorized, self.topics)))

        return predictions


def main():
    args = docopt.docopt(__doc__)

    train_path = args["<train>"]
    test_path = args["<test>"]

    print("Load data")
    X_raw, y, uids = load_train_data(train_path)
    X = map(analyze, X_raw)

    print("Encode labels")
    id2label = sorted(set(y))
    label2id = dict(zip(id2label, range(len(id2label))))
    y = [label2id[label] for label in y]

    # Classification
    clf = BayesienClassifier()

    print("Split train data")
    score = 0
    samples = 0
    skf = StratifiedKFold(y, 5)
    all_predictions = []
    all_labels = []
    for i, (train, test) in enumerate(skf):
        print("%ith fold" % (i + 1))
        X_train = [X[j] for j in train]
        Y_train = [y[j] for j in train]
        X_test = [X[j] for j in test]
        Y_test = [y[j] for j in test]

        print("Train")
        clf.fit(X_train, Y_train)
        print("Predict")
        predictions = clf.predict(X_test)

        score += compute_score(Y_test, predictions)
        all_predictions.extend(predictions)
        all_labels.extend(Y_test)
        samples += len(Y_test)

    print("Score:", score / float(samples))

    predictions = [id2label[label] for label in all_predictions]
    y = [id2label[label] for label in all_labels]
    cm = confusion_matrix(y, predictions)
    print('Confusion matrix, without normalization')
    numpy.set_printoptions(precision=2)
    print(cm)
    plt.figure()
    plot_confusion_matrix(sorted(label2id), cm)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(sorted(label2id), cm_normalized, title='Normalized confusion matrix')

    plt.show()
    return


    X_unlabeled, uids_test = load_test_data(test_path)
    X_unlabeled = map(analyze, X_unlabeled)

    # Predict unlabeled data
    clf = BayesienClassifier()
    clf.fit(X, y)
    prediction = clf.predict(X_unlabeled)
    print()
    print("BEST MODEL =>", score)

    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        for uid, label in zip(uids, prediction):
            print("%s,%s" % (uid, id2label[label]), file=output)


if __name__ == "__main__":
    main()
