#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import itertools
import subprocess


class VW(object):
    def __init__(self, options):
        self.options = options
        self.model = None

    def fit(self, X, y=None):
        # Dump data to vowpal wabbit format
        if not isinstance(X, (str, unicode)):
            assert y is not None
            filename = "train_%s.vw" % len(X)
            VW.output_data(filename, X, y)
        else:
            filename = X
        arguments = [
            "vw",
            filename,
            "-f", "model",
            "--cache_file", "cache"
        ]
        arguments.extend(self.options)
        print("$", " ".join(arguments))
        try:
            subprocess.check_call(arguments)
        except subprocess.CalledProcessError:
            return None
        self.model = "model"

    def predict(self, X):
        # Dump data to vowpal wabbit format
        if not isinstance(X, (str, unicode)):
            filename = "train_%s.vw" % len(X)
            VW.output_data(filename, X)
        else:
            filename = X
        arguments = [
            "vw",
            "-t",
            "-i", self.model,
            "-p", "output",
            "-r", "raw_output",
            filename
        ]
        print("$", " ".join(arguments))
        subprocess.check_call(arguments)
        prediction = []
        with open("output", "rb") as input_prediction:
            for line in input_prediction.read().split():
                prediction.append(int(float(line)))
        return prediction

    @staticmethod
    def make_line(label, features=None, namespaces=None):
        def normalize_feature((feature, score)):
            feature = feature.replace(' ', '_')
            return "%s:%s" % (feature, score)
        assert (features and not namespaces) or (namespaces and not features)
        if features is not None:
            namespaces = {"default": features}
        if label is None:
            label = 1
        namespaces_features = []
        for name, namespace in namespaces.items():
            features = " ".join(map(normalize_feature, namespace.items()))
            namespaces_features.append("|%s %s" % (name, features))
        return "%s %s" % (label, " ".join(namespaces_features))

    @staticmethod
    def output_data(filename, X, y=[]):
        print(filename)
        with open(filename, "wb") as output:
            for namespaces, response in itertools.izip_longest(X, y, fillvalue=None):
                print(
                    VW.make_line(response, namespaces=namespaces).encode("utf-8"),
                    file=output)
