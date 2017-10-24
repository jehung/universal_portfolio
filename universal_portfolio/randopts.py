"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
import os
import csv
import time
import sys
sys.path.append('/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
print(sys.path)
import jpype as jp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from util import process_data_choice
#jp.startJVM(jp.getDefaultJVMPath(), "-ea")

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


BackPropagationNetworkFactory = jp.JPackage('func').nn.backprop.BackPropagationNetworkFactory
DataSet = jp.JPackage('shared').DataSet
SumOfSquaresError = jp.JPackage('shared').SumOfSquaresError
NeuralNetworkOptimizationProblem = jp.JPackage('opt').example.NeuralNetworkOptimizationProblem
RandomizedHillClimbing = jp.JPackage('opt').RandomizedHillClimbing
SimulatedAnnealing = jp.JPackage('opt').SimulatedAnnealing
StandardGeneticAlgorithm = jp.JPackage('opt').ga.StandardGeneticAlgorithm
Instance = jp.JPackage('shared').Instance
RELU = jp.JPackage('func').nn.activation.RELU


INPUT_LAYER = 2232
HIDDEN_LAYER1 = 1000
HIDDEN_LAYER2 = 250
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 2000
OUTFILE = 'Results/randopts_LOG.txt'




def get_cv_set(data, label):
    traindata, trainlabel, testdata, testlabel = train_test_split(data, label, test_size=0.2)
    return traindata, trainlabel, testdata, testlabel


def initialize_instances(data, label):
    """Read the train.csv CSV data into a list of instances."""
    instances = []
    for i in range(len(data)):
        #instance = Instance([float(value) for value in data[i][:-1]])
        instance = Instance([float(value) for value in data[i]])
        instance.setLabel(Instance(float(label[i])))
        instances.append(instance)

    return instances


def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc


def train(oa, network, oaName, training_ints,testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print("\nError results for %s\n---------------------------" % (oaName,))
    times = [0]
    for iteration in range(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
        times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
            MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{},{}\n'.format(oaName, iteration,MSE_trg,MSE_tst,acc_trg,acc_tst,times[-1]);print(txt)
            with open(OUTFILE,'a+') as f:
                f.write(txt)

def main():
    """Run this experiment"""
    datapath = 'util/stock_dfs/'
    test_size=600
    all = process_data_choice.merge_all_data(datapath)
    inputdf, labeled = process_data_choice.embed(all)
    train_inputdf, val_inputdf, train_labeled, val_labeled = get_cv_set(inputdf.values, labeled)

    training_ints = initialize_instances(train_inputdf, train_labeled)
    testing_ints = initialize_instances(val_inputdf, val_labeled)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    #rule = RPROPUpdateRule()
    classification_network = factory.createRegressionNetwork([len(inputdf.columns), HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER],relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    rhc = RandomizedHillClimbing(nnop)
    #sa = SimulatedAnnealing(1E10, .95, nnop)
    #ga = StandardGeneticAlgorithm(100, 20, 10, nnop)
    with open(OUTFILE, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('RHC', 'iteration', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'))
    train(rhc, classification_network, 'RHC', training_ints,testing_ints, measure)
    
    #with open(OUTFILE, 'w') as f:
    #    f.write('{},{},{},{},{},{},{}\n'.format('SA', 'iteration', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'))
    #train(sa, classification_network, 'SA', training_ints, testing_ints, measure)
    #with open(OUTFILE, 'w') as f:
    #    f.write('{},{},{},{},{},{},{}\n'.format('GA', 'iteration', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'))
    #train(ga, classification_network, 'GA', training_ints,testing_ints, measure),


if __name__ == "__main__":
    main()
    jp.shutdownJVM()
