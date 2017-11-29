# -*- coding: utf-8 -*-
import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt


class TradingRRL(object):
    def __init__(self, T=1000, M=300, N=424, init_t=10000, mu=10000, sigma=0.04, rho=1.0, n_epoch=10):
        self.T = T
        self.M = M
        self.N = N
        self.init_t = init_t
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.all_t = None
        self.all_p = None
        self.t = None
        self.p = None
        self.r = None
        self.x = np.zeros([T, M + 2])
        self.F = np.zeros((T + 1, N))
        self.R = np.zeros((N,T))
        self.w = np.ones((M + 2, N))
        self.w_opt = np.ones((M + 2, N))
        self.epoch_S = np.empty(0)
        self.n_epoch = n_epoch
        self.progress_period = 100
        self.q_threshold = 0.5

    def load_csv(self, fname):
        tmp = pd.read_csv(fname, header=0, low_memory=False)
        print(tmp.head())
        print(tmp.shape)
        # tmp.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        tmp_tstr = tmp['Unnamed: 0']
        tmp_t = [dt.strptime(tmp_tstr[i], '%Y-%m-%d') for i in range(len(tmp_tstr))]
        tmp_p = tmp.iloc[:, 1:]
        print(tmp_p)
        self.all_t = np.array(tmp_t)
        self.all_p = np.array(tmp_p).reshape((1, -1))[0]
        print('shape', self.all_p)

    def quant(self, f):
        fc = f.copy()
        fc[np.where(np.abs(fc) < self.q_threshold)] = 0
        return np.sign(fc)

    def set_t_p_r(self):
        self.t = self.all_t[self.init_t:self.init_t + self.T + self.M + 1]
        self.p = self.all_p[self.init_t:self.init_t + self.T + self.M + 1]
        self.r = -np.diff(self.p)

    def set_x_F(self):
        for i in range(self.T - 1, -1, -1):
            self.x[i] = np.zeros(self.M + 2)
            self.x[i][0] = 1.0
            print(i)
            print(self.F.shape)
            self.x[i][self.M + 2 - 1] = self.F[-1, -1]
            for j in range(1, self.M + 2 - 1, 1):
                self.x[i][j] = self.r[i + j - 1]
        self.F = np.tanh(np.dot(self.x, self.w))
        print('f dimension', self.F.shape)
        print('x dimension', self.x.shape)

    def calc_R(self):
        #self.R = self.mu * (self.F[1:,:] * self.r[:self.T,:] - self.sigma * np.abs(-np.diff(self.F)))
        self.R = self.mu * (np.dot(self.r[:self.T], self.F[:,1:]) - self.sigma * np.abs(-np.diff(self.F, axis=1)))
        print('r dimension', self.R)

    def calc_sumR(self):
        self.sumR = np.cumsum(self.R, axis=1)
        self.sumR2 = np.cumsum((self.R ** 2)[::-1,:], axis=1)[::-1]
        print('sumr', self.sumR.shape)
        print('sumr2', self.sumR2.shape)

    def calc_dSdw(self):
        self.set_x_F()
        print('i am here')
        self.calc_R()
        self.calc_sumR()

        for i in range(self.N - 2, -1, -1):
            self.A = np.sum(self.sumR[:,i]) / self.T
            self.B = np.sum(self.sumR2[:,i]) / self.T
            self.S = self.A / np.sqrt(self.B - self.A ** 2)
            self.dSdA = self.S * (1 + self.S ** 2) / self.A
            self.dSdB = -self.S ** 3 / 2 / self.A ** 2
            self.dAdR = 1.0 / self.T
            self.dBdR = 2.0 / self.T * self.R[:,i]
            self.dRdF = -self.mu * self.sigma * (np.sign(-np.diff(self.F,axis=1)))
            self.dRdFp = self.mu * self.r[i] + self.mu * self.sigma * np.sign(-np.diff(self.F,axis=1))
            self.dFdw = np.zeros(self.M + 2)
            self.dFpdw = np.zeros(self.M + 2)
            self.dSdw = np.zeros((self.M + 2, self.N))
            self.dSdw_j = np.zeros(self.M + 2)
            for j in range(self.T - 1, -1, -1):
                if j != self.T - 1:
                    self.dFpdw = self.dFdw.copy()
                self.dFdw = (1 - self.F[j,i] ** 2) * (self.x[j] + self.w[self.M + 2 - 1,i] * self.dFpdw)
                self.dSdw_j += (self.dSdA * self.dAdR + self.dSdB * self.dBdR[j]) * (self.dRdF[j,i] * self.dFdw + self.dRdFp[j,i] * self.dFpdw)

            self.dSdw[:,i] = self.dSdw_j

    def update_w(self):
        self.w += self.rho * self.dSdw

    def fit(self):

        pre_epoch_times = len(self.epoch_S)

        self.calc_dSdw()
        print("Epoch loop start. Initial sharp's ratio is " + str(self.S) + ".")
        self.S_opt = self.S

        tic = time.clock()
        for e_index in range(self.n_epoch):
            print('e_index', e_index)
            self.calc_dSdw()
            if self.S > self.S_opt:
                self.S_opt = self.S
                self.w_opt = self.w.copy()
            self.epoch_S = np.append(self.epoch_S, self.S)
            self.update_w()
            if e_index % self.progress_period == self.progress_period - 1:
                toc = time.clock()
                print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" + str(
                    self.n_epoch + pre_epoch_times) + ". Shape's ratio: " + str(self.S) + ". Elapsed time: " + str(
                    toc - tic) + " sec.")
        toc = time.clock()
        print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" + str(
            self.n_epoch + pre_epoch_times) + ". Shape's ratio: " + str(self.S) + ". Elapsed time: " + str(
            toc - tic) + " sec.")
        self.w = self.w_opt.copy()
        self.calc_dSdw()
        print("Epoch loop end. Optimized sharp's ratio is " + str(self.S_opt) + ".")

    def save_weight(self):
        pd.DataFrame(self.w).to_csv("w.csv", header=False, index=False)
        pd.DataFrame(self.epoch_S).to_csv("epoch_S.csv", header=False, index=False)
        pd.DataFrame(self.F).to_csv("f.csv", header=False, index=False)

    def load_weight(self):
        tmp = pd.read_csv("w.csv", header=None)
        self.w = tmp.T.values[0]


def get_ticker(x):
    return x.split('/')[-1].split('.')[0]


def read_file(file, test=None):
    d = pd.read_csv(file).set_index('Date')
    d.fillna(0, inplace=True)
    ticker = get_ticker(file)
    d.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
                      'Volume': 'volume'},
             inplace=True)

    d.drop(labels=['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

    return d


def all_init_data():
    filepath = '../../util/stock_dfs/'
    alldata = []
    for f in os.listdir(filepath):
        datapath = os.path.join(filepath, f)
        if datapath.endswith('.csv'):
            # print(datapath)
            Res = read_file(datapath)
            alldata.append(Res)
    alldata = pd.concat(alldata, axis=1)
    alldata.fillna(0, inplace=True)
    alldata
    alldata.to_csv('all_data.csv')


def plot_hist(n_tick, R):
    rnge = max(R) - min(R)
    tick = rnge / n_tick
    tick_min = [min(R) - tick * 0.5 + i * tick for i in range(n_tick)]
    tick_max = [min(R) + tick * 0.5 + i * tick for i in range(n_tick)]
    tick_center = [min(R) + i * tick for i in range(n_tick)]
    tick_val = [0.0] * n_tick
    for i in range(n_tick):
        tick_val[i] = len(
            set(np.where(tick_min[i] < np.array(R))[0].tolist()).intersection(np.where(np.array(R) <= tick_max[i])[0]))
    plt.bar(tick_center, tick_val, width=tick)
    plt.grid()
    plt.show()


def main():
    #fname = '../../util/stock_dfs/A.csv'
    # fname = 'USDJPY30.csv'
    fname = 'all_data.csv'
    # all_init_data()

    init_t = 1001

    T = 1000
    M = 300
    N = 424
    mu = 1000
    sigma = 0.03
    rho = 1.0
    n_epoch = 10

    # RRL agent with initial weight.
    ini_rrl = TradingRRL(T, M, N, init_t, mu, sigma, rho, n_epoch)
    ini_rrl.load_csv(fname)
    ini_rrl.set_t_p_r()
    print('starting derivative calc')
    ini_rrl.calc_dSdw()
    print('finished 1')
    # RRL agent for training
    rrl = TradingRRL(T, M, N, init_t, mu, sigma, rho, n_epoch)
    rrl.all_t = ini_rrl.all_t
    rrl.all_p = ini_rrl.all_p
    rrl.set_t_p_r()
    rrl.fit()
    print('finished 2')
    rrl.save_weight()

    # Plot results.
    # Training for initial term T.
    plt.plot(range(len(rrl.epoch_S)), rrl.epoch_S)
    plt.title("Sharp's ratio optimization")
    plt.xlabel("Epoch times")
    plt.ylabel("Sharp's ratio")
    plt.grid(True)
    plt.savefig("sharp's ratio optimization.png", dpi=300)
    plt.close

    fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
    t = np.linspace(1, rrl.T, rrl.T)[::-1]
    ax[0].plot(t, rrl.p[:rrl.T])
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("USDJPY")
    ax[0].grid(True)

    ax[1].plot(t, ini_rrl.F[:rrl.T], color="blue", label="With initial weights")
    ax[1].plot(t, rrl.F[:rrl.T], color="red", label="With optimized weights")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("F")
    ax[1].legend(loc="upper left")
    ax[1].grid(True)

    ax[2].plot(t, ini_rrl.sumR, color="blue", label="With initial weights")
    ax[2].plot(t, rrl.sumR, color="red", label="With optimized weights")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("Sum of reward[yen]")
    ax[2].legend(loc="upper left")
    ax[2].grid(True)
    plt.savefig("rrl_train.png", dpi=300)
    fig.clear()

    # Prediction for next term T with optimized weight.
    # RRL agent with initial weight.
    ini_rrl_f = TradingRRL(T, M, N, init_t - T, mu, sigma, rho, n_epoch)
    ini_rrl_f.all_t = ini_rrl.all_t
    ini_rrl_f.all_p = ini_rrl.all_p
    ini_rrl_f.set_t_p_r()
    ini_rrl_f.calc_dSdw()
    # RRL agent with optimized weight.
    rrl_f = TradingRRL(T, M, N, init_t - T, mu, sigma, rho, n_epoch)
    rrl_f.all_t = ini_rrl.all_t
    rrl_f.all_p = ini_rrl.all_p
    rrl_f.set_t_p_r()
    rrl_f.w = rrl.w
    rrl_f.calc_dSdw()

    fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
    t_f = np.linspace(rrl.T + 1, rrl.T + rrl.T, rrl.T)[::-1]
    ax[0].plot(t_f, rrl_f.p[:rrl_f.T])
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("USDJPY")
    ax[0].grid(True)

    ax[1].plot(t_f, ini_rrl_f.F[:rrl_f.T], color="blue", label="With initial weights")
    ax[1].plot(t_f, rrl_f.F[:rrl_f.T], color="red", label="With optimized weights")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("F")
    ax[1].legend(loc="lower right")
    ax[1].grid(True)

    ax[2].plot(t_f, ini_rrl_f.sumR, color="blue", label="With initial weights")
    ax[2].plot(t_f, rrl_f.sumR, color="red", label="With optimized weights")
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("Sum of reward[yen]")
    ax[2].legend(loc="lower right")
    ax[2].grid(True)
    plt.savefig("rrl_prediction.png", dpi=300)
    fig.clear()


if __name__ == "__main__":
    main()