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
import matplotlib.pyplot as plt
from mimicry.mimicry import mimic as mimic
import networkx as nx
import random
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM
from collections import defaultdict
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix
from util import process_data
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.dnn  import DNNClassifier
from tensorflow.contrib.layers import real_valued_column
import time
from time import clock
from itertools import product
from array import *
import jpype as jp
sys.path.append("C:/MOOCs/CS 7641/proj2/ABAGAIL.jar")
jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
jp.java.io.FileReader
jp.java.io.File
jp.java.lang.String
jp.java.lang.StringBuffer
jp.java.lang.Boolean
jp.java.util.Random
jp.java.dist.DiscreteDependencyTree
jp.java.dist.DiscreteUniformDistribution
jp.java.opt.DiscreteChangeOneNeighbor
jp.java.opt.EvaluationFunction
jp.java.opt.EvaluationFunction
jp.java.opt.HillClimbingProblem
jp.java.opt.NeighborFunction
jp.java.opt.RandomizedHillClimbing
jp.java.opt.SimulatedAnnealing
jp.java.opt.example.FourPeaksEvaluationFunction
jp.java.opt.ga.CrossoverFunction
jp.java.opt.ga.SingleCrossOver
jp.java.opt.ga.DiscreteChangeOneMutation
jp.java.opt.ga.GenericGeneticAlgorithmProblem
jp.java.opt.GenericHillClimbingProblem
jp.java.opt.ga.GeneticAlgorithmProblem
jp.java.opt.ga.MutationFunction
jp.java.opt.ga.StandardGeneticAlgorithm
jp.java.opt.ga.UniformCrossOver
jp.java.opt.prob.GenericProbabilisticOptimizationProblem
jp.java.opt.prob.MIMIC
jp.java.opt.prob.ProbabilisticOptimizationProblem
jp.java.shared.FixedIterationTrainer
jp.java.opt.example.ContinuousPeaksEvaluationFunction


ContinuousPeaksEvaluationFunction = jp.JPackage('opt').example.ContinuousPeaksEvaluationFunction
DiscreteUniformDistribution = jp.JPackage('dist').DiscreteUniformDistribution
DiscreteChangeOneNeighbor = jp.JPackage('opt').DiscreteChangeOneNeighbor
DiscreteChangeOneMutation = jp.JPackage('opt').ga.DiscreteChangeOneMutation
SingleCrossOver = jp.JPackage('opt').ga.SingleCrossOver
DiscreteDependencyTree = jp.JPackage('dist').DiscreteDependencyTree
GenericHillClimbingProblem = jp.JPackage('opt').GenericHillClimbingProblem
GenericGeneticAlgorithmProblem = jp.JPackage('opt').ga.GenericGeneticAlgorithmProblem
GenericProbabilisticOptimizationProblem = jp.JPackage('opt').prob.GenericProbabilisticOptimizationProblem
RandomizedHillClimbing = jp.JPackage('opt').RandomizedHillClimbing
FixedIterationTrainer = jp.JPackage('shared').FixedIterationTrainer
SimulatedAnnealing = jp.JPackage('opt').SimulatedAnnealing
StandardGeneticAlgorithm = jp.JPackage('opt').ga.StandardGeneticAlgorithm
MIMIC = jp.JPackage('opt').prob.MIMIC



def baseline_nn(n, X, y):
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



def evaluate_baselineNN(input, labels):
    test_size=600
    print(classification_report(labels['multi_class'][-test_size:], res.predict(input[-test_size:])))
    print(confusion_matrix(labels['multi_class'][-test_size:], res.predict(input[-test_size:])))

    labels['predicted_action'] = list(map(lambda x: -1 if x < 5 else 0 if x == 5 else 1, res.predict(input)))
    print(confusion_matrix(labels['class'][-test_size:], labels['predicted_action'][-test_size:]))

    labels['pred_return'] = labels['predicted_action'] * labels['return']

    Res = labels[-test_size:][['return', 'act_return', 'pred_return']].cumsum()
    Res[0] = 0
    Res.plot()



class DNNModel():
    def __init__(self):
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=[])
        with tf.variable_scope("ff"):
            droped_input = tf.nn.dropout(self.input_data, keep_prob=self.dropout_prob)

            layer_1 = tf.contrib.layers.fully_connected(
                num_outputs=hidden_1_size,
                inputs=droped_input,
            )
            layer_2 = tf.contrib.layers.fully_connected(
                num_outputs=hidden_2_size,
                inputs=layer_1,
            )
            self.logits = tf.contrib.layers.fully_connected(
                num_outputs=num_classes,
                activation_fn=None,
                inputs=layer_2,
            )
        with tf.variable_scope("loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_data, logits=self.logits)
            mask = (1 - tf.sign(1 - self.target_data))  # Don't give credit for flat days
            mask = tf.cast(mask, tf.float32)
            self.loss = tf.reduce_sum(self.losses)

        with tf.name_scope("train"):
            opt = tf.train.AdamOptimizer(lr)
            gvs = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(gvs, global_step=global_step)

        with tf.name_scope("predictions"):
            self.probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.probs, 1)
            correct_pred = tf.cast(tf.equal(self.predictions, tf.cast(self.target_data, tf.int64)), tf.float64)
            self.accuracy = tf.reduce_mean(correct_pred)





if __name__ == '__main__':
    # for MIMIC in continuous space
    datapath = 'util/stock_dfs/'
    all = process_data.merge_all_data(datapath)
    inputdf, targetdf = process_data.embed(all)
    labeled = process_data.process_target(targetdf)

    domain = [(0, 1)] * len(inputdf)
    m = mimic.Mimic(domain, sum, samples=500)
    for i in range(25):
        print(np.average([sum(sample) for sample in m.fit()[:5]]))
        m.fit()
    results = m.fit()
    print(results)

    '''
    # for neural network in sklearn
    datapath = 'util/stock_dfs/'
    all = process_data.merge_all_data(datapath)
    inputdf, targetdf = process_data.embed(all)
    labeled = process_data.process_target(targetdf)

    clf, score, gs = baseline_nn(len(inputdf.columns), inputdf, labeled['multi_class'])


    
    # for baseline dnn in tensorflow
    labeled['tf_class'] = labeled['multi_class']
    num_features = len(inputdf.columns)
    dropout = 0.2
    hidden_1_size = 1000
    hidden_2_size = 250
    num_classes = labeled.tf_class.nunique()
    NUM_EPOCHS = 100
    BATCH_SIZE = 50
    lr = 0.0001
    test_size=600
    train = (inputdf[:-test_size].values, labeled.tf_class[:-test_size].values)
    val = (inputdf[-test_size:].values, labeled.tf_class[-test_size:].values)
    NUM_TRAIN_BATCHES = int(len(train[0]) / BATCH_SIZE)
    NUM_VAL_BATCHES = int(len(val[1]) / BATCH_SIZE)

    with tf.Graph().as_default():
        model = DNNModel()
        input_ = train[0]
        target = train[1]
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run([init])
            epoch_loss = 0
            for e in range(NUM_EPOCHS):
                if epoch_loss > 0 and epoch_loss < 1:
                    break
                epoch_loss = 0
                for batch in range(0, NUM_TRAIN_BATCHES):
                    start = batch * BATCH_SIZE
                    end = start + BATCH_SIZE
                    feed = {
                        model.input_data: input_[start:end],
                        model.target_data: target[start:end],
                        model.dropout_prob: 0.9
                    }

                    _, loss, acc = sess.run(
                        [
                            model.train_op,
                            model.loss,
                            model.accuracy,
                        ]
                        , feed_dict=feed
                    )
                    epoch_loss += loss
                print('step - {0} loss - {1} acc - {2}'.format((1 + batch + NUM_TRAIN_BATCHES * e), epoch_loss, acc))

            print('done training')
            final_preds = np.array([])
            final_probs = None
            for batch in range(0, NUM_VAL_BATCHES):

                start = batch * BATCH_SIZE
                end = start + BATCH_SIZE
                feed = {
                    model.input_data: val[0][start:end],
                    model.target_data: val[1][start:end],
                    model.dropout_prob: 1
                }

                acc, preds, probs = sess.run(
                    [
                        model.accuracy,
                        model.predictions,
                        model.probs
                    ]
                    , feed_dict=feed
                )
                print(acc)
                final_preds = np.concatenate((final_preds, preds), axis=0)
                if final_probs is None:
                    final_probs = probs
                else:
                    final_probs = np.concatenate((final_probs, probs), axis=0)
            prediction_conf = final_probs[np.argmax(final_probs, 1)]
'''