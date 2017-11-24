# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)
import os
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
from sklearn.externals import joblib
import quandl
import random, timeit
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

'''
Name:        The Self Learning Quant, Example 3

Author:      Daniel Zakrisson

Created:     30/03/2016
Copyright:   (c) Daniel Zakrisson 2016
Licence:     BSD

Requirements:
Numpy
Pandas
MatplotLib
scikit-learn
TA-Lib, instructions at https://mrjbq7.github.io/ta-lib/install.html
Keras, https://keras.io/
Quandl, https://www.quandl.com/tools/python
backtest.py from the TWP library. Download backtest.py and put in the same folder

/plt create a subfolder in the same directory where plot files will be saved

'''


def get_ticker(x):
    return x.split('/')[-1].split('.')[0]


def read_file(file, test=None):
    scaler = preprocessing.MinMaxScaler()
    d = pd.read_csv(file).set_index('Date')
    d.fillna(0, inplace=True)
    ticker = get_ticker(file)
    d['ticker'] = ticker
    d.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
                      'Volume (BTC)': 'volume'},
             inplace=True)

    x_train = d.iloc[:-100, ]
    x_test = d.iloc[-100:, ]
    if test:
        return x_test, ticker
    else:
        return x_train, ticker


# Initialize first state, all items are placed deterministically
def init_state(file, test):
    d, ticker = read_file(file, test=test)
    xdata = pd.DataFrame()
    scaler = preprocessing.StandardScaler()
    xdata['adj_close'] = d['adj_close']  # .values
    xdata['diff'] = xdata['adj_close'].diff(periods=1)
    xdata['diff'].fillna(0, inplace=True)
    xdata['sma15'] = SMA(d, timeperiod=15)
    xdata['sma60'] = SMA(d, timeperiod=60)
    xdata['rsi'] = RSI(d, timeperiod=14)
    xdata['atr'] = ATR(d, timeperiod=14)
    xdata.fillna(0, inplace=True)

    # --- Preprocess data
    # xdata = np.column_stack((close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr))
    xdata = pd.DataFrame(scaler.fit_transform(xdata), columns=xdata.columns)
    xdata['ticker'] = ticker
    pivot_columns = xdata.columns[0:-1]

    pivot = xdata.pivot_table(index=d.index, columns='ticker', values=pivot_columns)  # Make a pivot table from the data
    pivot.columns = [s1 + '-' + s2 for (s1, s2) in pivot.columns.tolist()]

    return pivot


def all_init_data(test=False):
    filepath = 'util/stock_dfs/'
    all = []
    scaler = preprocessing.StandardScaler()
    for f in os.listdir(filepath):
        datapath = os.path.join(filepath, f)
        if datapath.endswith('.csv'):
            # print(datapath)
            Res = init_state(datapath, test=test)
            all.append(Res)
    all = pd.concat(all, axis=1)
    all.fillna(0, inplace=True)

    closecol = [col for col in all.columns if 'adj_close' in col]
    close = all[closecol].values

    # xdata = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi, atr))
    xdata = np.vstack(all.values)
    xdata = np.nan_to_num(xdata)
    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    else:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    state = xdata[0:1, 0:1, :]

    return state, xdata, close


# Take Action
def take_action(state, xdata, action, signal, time_step):
    # this should generate a list of trade signals that at evaluation time are fed to the backtester
    # the backtester should get a list of trade signals and a list of price data for the assett

    # make necessary adjustments to state and then return it
    time_step += 1

    # if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == xdata.shape[0]:
        state = xdata[time_step - 1:time_step, 0:1, :]
        terminal_state = 1
        signal.loc[time_step] = 0

        return state, time_step, signal, terminal_state

    # move the market data window one step forward
    state = xdata[time_step - 1:time_step, 0:1, :]
    # take action
    if action == 1:
        signal.loc[time_step] = 100
    elif action == 2:
        signal.loc[time_step] = -100
    else:
        signal.loc[time_step] = 0
    # print(state)
    terminal_state = 0
    # print(signal)

    return state, time_step, signal, terminal_state


# Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)

    if eval == False:
        try:
            bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata[time_step - 2:time_step]], index=signal[time_step - 2:time_step].index.values),
                          signal[time_step - 2:time_step], signalType='shares')
            reward = np.max((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2]) * bt.data['shares'].iloc[-1])
        except:
            pass
    if terminal_state == 1 and eval == True:
        bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata], index=signal.index.values), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]
        plt.figure(figsize=(3, 4))
        bt.plotTrades()
        plt.axvline(x=400, color='black', linestyle='--')
        plt.text(250, 400, 'training data')
        plt.text(450, 400, 'test data')
        plt.suptitle(str(epoch))
        plt.savefig('plt/' + 'knapsack_' + str(epoch) + '.png')
        plt.close('all')

        '''
        # save a figure of the test set
        plt.figure(figsize=(10, 25))
        for i in range(xdata.T.shape[0]):
        #frame = pd.concat(btFrame, axis=1)
            bt = twp.Backtest(pd.Series(data=[x for x in xdata.T[i]], index=signal.index.values), signal, signalType='shares')
            reward += np.max(bt.pnl.iloc[-1])
            bt.plotTrades()
        #plt.axvline(x=400, color='black', linestyle='--')
        #plt.text(250, 400, 'training data')
        #plt.text(450, 400, 'test data')
        #plt.suptitle(str(epoch))
        plt.savefig('plt/' + 'knapsack_' + str(epoch) + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
        plt.close('all')
        '''
    # print(time_step, terminal_state, eval, reward)

    return reward


def evaluate_Q(eval_data, eval_model, epoch=0):
    # This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = all_init_data()
    status = 1
    terminal_state = 0
    time_step = 1
    while (status == 1):
        # We start in state S
        qval = eval_model.predict(state, batch_size=batch_size)
        action = (np.argmax(qval))
        # Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        # Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True,
                                 epoch=epoch)
        state = new_state
        if terminal_state == 1:  # terminal state
            status = 0

    return eval_reward


if __name__ == "__main__":
    # This neural network is the the Q-function, run it like this:
    # model.predict(state.reshape(1,64), batch_size=1)
    batch_size = 7
    num_features = 2544
    epochs = 10
    gamma = 0.95  # since the reward can be several time steps away, make gamma high
    epsilon = 1
    batchSize = 100
    buffer = 200
    replay = []
    learning_progress = []

    model = Sequential()
    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=True,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=False,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    rms = RMSprop()
    adam = Adam()
    model.compile(loss='mse', optimizer=adam)

    start_time = timeit.default_timer()

    # read_convert_data(symbol='XBTEUR') #run once to read indata, resample and convert to pickle

    astate, xdata, aprice_data = all_init_data()
    bstate, test_data, test_price_data = all_init_data(test=True)
    '''
    bstate, test_data, test_price_data = all_init_data(test=True)
    print(astate.shape)
    print(bstate.shape)
    print(xdata.shape)
    print(test_data.shape)
    print(price_data.shape)
    print(test_price_data.shape)
    '''

    # stores tuples of (S, A, R, S')
    h = 0
    # signal = pd.Series(index=market_data.index)
    signal = pd.Series(index=np.arange(len(xdata)))
    for i in range(epochs):
        if i == epochs - 1:  # the last epoch, use test data set
            state, xdata, price_data = all_init_data()
        else:
            state, xdata, price_data = all_init_data(test=True)
        status = 1
        terminal_state = 0
        time_step = 5
        # while game still in progress
        while (status == 1):
            # We are in state S
            # Let's run our Q function on S to get Q values for all possible actions
            print('epoch ' + str(i))
            qval = model.predict(state, batch_size=batch_size)
            if (random.random() < epsilon):  # choose random action
                action = np.random.randint(0, 3)  # assumes 4 different actions
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(qval))
            # Take action, observe new state S'
            new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
            # Observe reward

            reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state)
            print('new_state', new_state)
            print('reward', reward)

            # Experience replay storage
            if (len(replay) < buffer):  # if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
                # print(time_step, reward, terminal_state)
            else:  # if buffer full, overwrite old values
                if (h < (buffer - 1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                # randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    # Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    old_qval = model.predict(old_state, batch_size=batch_size)
                    newQ = model.predict(new_state, batch_size=batch_size)
                    maxQ = np.max(newQ)
                    y = np.zeros((1, 3))
                    y[:] = old_qval[:]
                    if terminal_state == 0:  # non-terminal state
                        update = (reward + (gamma * maxQ))
                    else:  # terminal state
                        update = reward
                    # print('rewardbase', reward)

                    # print('update', update)
                    y[0][action] = update
                    # print(time_step, reward, terminal_state)
                    X_train.append(old_state)
                    y_train.append(y.reshape(3, ))

                X_train = np.squeeze(np.array(X_train), axis=(1))
                y_train = np.array(y_train)
                model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)

                state = new_state
            if terminal_state == 1:  # if reached terminal state, update epoch status
                status = 0


        eval_reward = evaluate_Q(test_data, model, i)
        # eval_reward = value_iter(test_data, epsilon, epochs)
        learning_progress.append(eval_reward)
        print("Epoch #: %s Reward: %f Epsilon: %f" % (i, eval_reward, epsilon))
        # learning_progress.append((reward))
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1.0 / epochs)


    elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
    print("Completed in %f" % (elapsed,))

    bt = twp.Backtest(pd.Series(data=[x[0] for x in test_price_data]), signal, signalType='shares')
    bt.data['delta'] = bt.data['shares'].diff().fillna(0)


    print(bt.data)
    bt.data.to_csv('plt/knapsack_data.csv')
    unique, counts = np.unique(filter(lambda v: v == v, signal.values), return_counts=True)
    print(np.asarray((unique, counts)).T)

    plt.figure()
    plt.subplot(3, 1, 1)
    bt.plotTrades()
    plt.subplot(3, 1, 2)
    bt.pnl.plot(style='x-')
    plt.subplot(3, 1, 3)
    plt.plot(learning_progress)
    print('to plot', learning_progress)

    plt.savefig('plt/q_summary' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)
    plt.show()


