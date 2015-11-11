#! /usr/bin/env python
# -*- coding: utf-8 -*-

import xgboost as xgb


class XGBOOST(object):

    def __init__(self):
        self.param = {
            'objective': 'multi:softmax'
        }
        self.param['eta'] = 0.1
        self.param['max_depth'] = 6
        self.param['silent'] = 1
        self.param['nthread'] = 4

        self.bst = None

    def fit(self, X, y=None):
        self.param['num_class'] = len(set(y))
        data = xgb.DMatrix(X, label=y)
        self.bst = xgb.train(self.param.items(), data, 10)

    def predict(self, X, y=None):
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)
    transform = predict

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)
    fit_transform = fit_predict
