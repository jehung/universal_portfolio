"""
RHC NN training on my classification problem

"""

"""
Do this before running the code in terminal / command line:
git clone https://github.com/originell/jpype.git
cd jpype
python setup.py install'

Additional reference doc:
https://stackoverflow.com/questions/35736763/practical-use-of-java-class-jar-in-python
"""

import os
import csv
import time
import sys
sys.path.append('/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
import jpype as jp
from util import process_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from array import *
from itertools import product
jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
jp.java.lang.System.out.println("hello world")
jp.java.func.nn.backprop.BackPropagationNetworkFactory
jp.java.func.nn.backprop.RPROPUpdateRule
jp.java.func.nn.backprop.BatchBackPropagationTrainer
jp.java.shared.SumOfSquaresError
jp.java.shared.DataSet
jp.java.shared.Instance
jp.java.opt.SimulatedAnnealing
jp.java.opt.example.NeuralNetworkOptimizationProblem
jp.java.opt.RandomizedHillClimbing
jp.java.ga.StandardGeneticAlgorithm
jp.java.func.nn.activation.RELU
jp.java.opt.example.NeuralNetworkEvaluationFunction



BackPropagationNetworkFactory = jp.JPackage('func').nn.backprop.BackPropagationNetworkFactory
DataSet = jp.JPackage('shared').DataSet
SumOfSquaresError = jp.JPackage('shared').SumOfSquaresError
NeuralNetworkOptimizationProblem = jp.JPackage('opt').example.NeuralNetworkOptimizationProblem
RandomizedHillClimbing = jp.JPackage('opt').RandomizedHillClimbing
Instance = jp.JPackage('shared').Instance
RELU = jp.JPackage('func').nn.activation.RELU
NeuralNetworkEvaluationFunction = jp.JPackage('opt').example.NeuralNetworkEvaluationFunction
DiscreteDependencyTree = jp.JPackage('dist').DiscreteDependencyTree



INPUT_LAYER = 450
OUTPUT_LAYER = 3
TRAINING_ITERATIONS = 2001
OUTFILE = 'MIMIC_LOG.txt'
N=100
T=49
maxIters = 20001
numTrials=5
fill = [2] * N
ranges = array('i', fill)





def get_cv_set(data):
    train, val = train_test_split(data, test_size=0.2)
    return train, val


def initialize_instances(data, label):
    """Read the train.csv CSV data into a list of instances."""
    instances = []

    '''
    # Read in the CSV file
    with open(infile, "r") as dat:
        next(dat)
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)
    '''
    for i in range(len(data)):
        #instance = Instance([float(value) for value in data[i][:-1]])
        instance = Instance([float(value) for value in data[i]])
        instance.setLabel(Instance(float(label[i])))
        instances.append(instance)

    return instances


def run_mimic():
    """Run this experiment"""
    datapath = 'util/stock_dfs/'
    outfile = 'Results/randopts_@ALG@_@N@_LOG.txt'
    all = process_data.merge_all_data(datapath)
    train_set, val_set = get_cv_set(all)

    train_inputdf, train_targetdf = process_data.embed(train_set)
    train_labeled = process_data.process_target(train_targetdf)
    training_ints = initialize_instances(train_inputdf, train_labeled['multi_class'])
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    relu = RELU()
    #rule = RPROPUpdateRule()
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, OUTPUT_LAYER],relu)
    data_set = DataSet(training_ints)
    times = [0]
    for t in range(numTrials):
        for samples, keep, m in product([100], [50], [0.7, 0.9]):
            fname = outfile.replace('@ALG@', 'MIMIC{}_{}_{}'.format(samples, keep, m)).replace('@N@', str(t + 1))
            with open(fname, 'w') as f:
                f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
            ef = NeuralNetworkEvaluationFunction(classification_network, data_set, measure)
            odd = DiscreteUniformDistribution(ranges)
            df = DiscreteDependencyTree(m, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            mimic = MIMIC(samples, keep, pop)
            fit = FixedIterationTrainer(mimic, 10)
            times = [0]
            for i in range(0, maxIters, 10):
                start = clock()
                fit.train()
                elapsed = time.clock() - start
                times.append(times[-1] + elapsed)
                fevals = ef.fevals
                score = ef.value(mimic.getOptimal())
                ef.fevals -= 1
                st = '{},{},{},{},{},{},{},{},{}\n'.format('MIMIC', t, i, samples, keep, m, score, times[-1], fevals)
                print(st)
                with open(fname, 'a') as f:
                    f.write(st)


if __name__ == "__main__":
    run_mimic()
    jp.shutdownJVM()
