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
from sklearn import metrics
import docopt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from joblib import Parallel, delayed
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


from whatscooking.features import analyze
from whatscooking.xgboost_wrapper import XGBOOST
from whatscooking.kmeans_embedding import KMeansEmbedding
import whatscooking.data as data


def compute_score(labels, prediction):
    return metrics.accuracy_score(labels, prediction, normalize=False)


def grid_search(model, X, y):
    # Grid search
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__penalty': ('l1', 'l2'),
        'clf__solver': ('newton-cg', 'lbfgs', 'liblinear')
    }

    print("Grid search")
    gs_clf = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        n_jobs=-1)
    classifier = gs_clf.fit(X, y)

    # Display best parameters
    print("Best parameters")
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print("Score:", score)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    return classifier, score


def kcross_validation(model, X, y):
    # Predict unlabeled data
    skf = StratifiedKFold(y, n_folds=3)
    score = 0
    total = 0
    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        score += compute_score(y_test, prediction)
        total += len(y_test)
    score = float(score) / float(total)
    print(score)
    model.fit(X, y)
    return model, score


def id_analyze(tokens):
    return tokens


def main():
    args = docopt.docopt(__doc__)

    print("> Load data")
    X_train, y, uids = data.load_train_data(args["<train>"])
    X_test, uids = data.load_test_data(args["<test>"])

    print("> Encode labels")
    labels_encoder = LabelEncoder()
    y = labels_encoder.fit_transform(y)
    print(labels_encoder)

    print("> Analyze data")
    X_train = Parallel(n_jobs=-1)(delayed(analyze)(sample) for sample in X_train)
    X_test = Parallel(n_jobs=-1)(delayed(analyze)(sample) for sample in X_test)

        # Feature extraction
    features = FeatureUnion([
        # Features based on TfidfVectorizer with 'word' analyzer
        ("tfidf_words", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer='word', strip_accents="unicode", lowercase=False, norm='l2')),
            ("union", FeatureUnion([
                # Select best feature according to chi2 test
                ("chi2", SelectKBest(chi2, k=2000)),
                # Create dictionary of kmeans features
                ("kmeans_embedding", KMeansEmbedding(n_clusters=100)),
                # Create tree embedding features
                ("tree_embedding", RandomTreesEmbedding(n_estimators=30, random_state=0, max_depth=3)),
                # Use principal components from SVD
                ("svd", TruncatedSVD(50))
            ], n_jobs=-1))
        ])),
        # Features based on TfidfVectorizer with 'char' analyzer
        ("tfidf_char", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer='char', ngram_range=(2, 8), strip_accents="unicode", lowercase=False, norm='l2')),
            ("union", FeatureUnion([
                # Select best feature according to chi2 test
                ("chi2", SelectKBest(chi2, k=50)),
                # Create dictionary of kmeans features
                ("kmeans_embedding", KMeansEmbedding(n_clusters=20)),
                # Create tree embedding features
                ("tree_embedding", RandomTreesEmbedding(n_estimators=20, random_state=0, max_depth=3))
            ], n_jobs=-1))
        ]))
    ])

    X_train = features.fit_transform(X_train, y)
    X_test = features.transform(X_test)

    print(X_train.shape)

    print("> Train classifier")
    classifier, score = kcross_validation(
        LogisticRegression(),
        X_train,
        y
    )
    print(score, classifier)

    print("Make prediction on test data")
    prediction = map(int, classifier.predict(X_test))
    labels = labels_encoder.inverse_transform(prediction)
    with open(str(score).replace(".", '_'), "wb") as output:
        print("id,cuisine", file=output)
        for uid, label in zip(uids, labels):
            print("%s,%s" % (uid, label), file=output)


if __name__ == "__main__":
    main()
