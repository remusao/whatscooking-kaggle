#! /usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans


class KMeansEmbedding(object):
    def __init__(self, n_clusters=50):
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)

    def fit(self, X, y=None):
        self.kmeans.fit(X)

    def fit_transform(self, X, y=None):
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        # Project to new space
        # Transform X_train (n_samples x n_features) into a dictionary
        # of features (centroids centers) of dim (n_samples x n_clusters)
        return pairwise_distances(
            X,
            self.kmeans.cluster_centers_,
            metric="cosine")
