import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, learning_curve, validation_curve
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)
import get_all_data
import mimic
import networkx as nx
import random
from scipy import stats
from sklearn.metrics import mutual_info_score

np.set_printoptions(precision=4)


models = {
    #'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
    'NeuralNetwork': MLPClassifier(verbose=5, hidden_layer_sizes=(109, 76, 76, 76, 76)),
    #'GradientBoosting': GradientBoostingClassifier(max_depth=1, n_estimators=50),
    #'SupportVectorMachine': LinearSVC(class_weight='balanced'),
    #'KNearestNeighbor': KNeighborsClassifier(n_neighbors=5)
}


params1 = {
    #'DecisionTree': {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]},
    'NeuralNetwork': {'validation_fraction': [0.1, 0.25, 0.33, 0.5, 0.75, 0.9]},
    #'GradientBoosting': {'max_depth': [1, 2, 3]},
    #'SupportVectorMachine': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    #'KNearestNeighbor': {'n_neighbors': [3,7,11]}
}


'''
class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=10, n_jobs=-1, verbose=5, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X, y)
            self.grid_searches[key] = gs


    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            return pd.Series({**params, **d})

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                for k in self.keys
                for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        print(df[columns])
        return df[columns]
'''




def plot_complexity_curve(estimator, title, X, y, param_name, param_range, cv=None,
                        n_jobs=1):
    plt.figure()
    plt.title(title)
    plt.title("Validation Curves")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=3, scoring="roc_auc", n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


'''
def complexity():
    helper1 = EstimatorSelectionHelper(models, params2)
    all_data = get_all_data.get_all_data()
    train, target = get_all_data.process_data(all_data)
    training_features, test_features, \
    training_target, test_target, = train_test_split(train, target, test_size=0.33, random_state=778)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_target)
    helper1.fit(X_train, y_train, scoring='f1', n_jobs=1)
    helper1.score_summary(sort_by='min_score')
'''



all_data = get_all_data.get_all_data()
train, target = get_all_data.process_data(all_data)
samples = train[0:4500]
print(samples)

distribution = mimic.Distribution(samples)
print('distribution', distribution)
distribution._generate_bayes_net()



for node_ind in distribution.bayes_net.nodes():
    print(distribution.bayes_net.node[node_ind])

pos = nx.spring_layout(distribution.spanning_graph)

edge_labels = dict(
        [((u, v,), d['weight'])
         for u, v, d in distribution.spanning_graph.edges(data=True)])

nx.draw_networkx(distribution.spanning_graph, pos)
nx.draw_networkx_edge_labels(
    distribution.spanning_graph,
    pos,
    edge_labels=edge_labels)

plt.show()





'''
for model in models:
    title = model
    cv = ShuffleSplit(n_splits=5, test_size=0.33)
    print(title)
    #plot_learning_curve(models[model], title, train, target, cv=cv, n_jobs=1)
    plot_complexity_curve(models[model], title, train, target, list(params1[model].keys())[0], list(params1[model].values())[0], cv=3, n_jobs=-1)
    plt.show()
'''



