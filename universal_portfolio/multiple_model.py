# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: jenny hung
"""

import sys
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM
from collections import defaultdict
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from util import process_data




def clustering(n, X, y):
    mlp = MLPClassifier(solver='sgd', learning_rate='invscaling', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    parameters = {
        'NN__hidden_layer_sizes': [(n,), (n, n, n), (n, n, n, n, n),(n,n, n, n, n, n, n),
                                   (n,), (n, int(0.9*n),int(0.9*n)), (n, int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n)),
                                   (n, int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n)),
                                   (n,), (n, int(0.8 *n), int(0.8 * n)), (n, int(0.8 *n), int(0.8 * n), int(0.8 * n), int(0.8 * n)),
                                   (n, int(0.8 *n), int(0.8 * n), int(0.8 * n), int(0.8 * n), int(0.8 * n), int(0.8 * n)),
                                   (n,), (n, int(0.7 *n), int(0.7 * n)), (n, int(0.7 *n), int(0.7 * n), int(0.7 * n), int(0.7 * n)),
                                   (n, int(0.7 *n), int(0.7 * n), int(0.7 * n), int(0.7 * n), int(0.7 * n), int(0.7 * n)),]}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    #km = KMeans(random_state=5)
    #pipe = Pipeline([('KM', km), ('NN', mlp)])
    pipe = Pipeline([('NN', mlp)])
    gs = GridSearchCV(pipe, parameters, scoring=make_scorer(accuracy_score), verbose=10)
    gs.fit(X, y)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)
    mat = clf.predict_proba(X)
    print(mat)

    return clf, gs.best_score_, gs



if __name__ == '__main__':
    datapath = 'util/stock_dfs/'
    all = process_data.merge_all_data(datapath)
    inputdf, targetdf = process_data.embed(all)
    labeled = process_data.process_target(targetdf)

    clf, score, gs = clustering(len(inputdf.columns), inputdf, labeled['multi_class'])

