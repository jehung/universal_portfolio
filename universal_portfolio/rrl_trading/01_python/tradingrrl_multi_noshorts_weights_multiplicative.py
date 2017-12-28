import os
import time
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import heapq


class TradingRRL(object):
    def __init__(self, T=1000, M=300, N=424, init_t=10000, mu=10000, sigma=0.04, rho=1.0, n_epoch=10):
        self.T = T
        self.M = M
        self.N = N
        self.TOP = 30
        self.threshold = 0.0
        self.init_t = init_t
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.all_t = None
        self.all_p = None
        self.t = None
        self.p = None
        self.bench = None
        self.r = None
        self.x = np.zeros([T, M + 2])
        self.F = np.zeros((T + 1, N))
        self.FS = np.zeros((T + 1, N))
        self.R = np.zeros((T, N))
        self.w = np.ones((M + 2, N))
        self.w_opt = np.ones((M + 2, N))
        self.epoch_S = pd.DataFrame()
        self.n_epoch = n_epoch
        self.progress_period = 100
        self.q_threshold = 0.5
        self.b = np.ones((T+1, N))
        self.total = None
        self.bench = None


    def load_bench(self, bench):
        tmp = pd.read_csv(bench, header=0, low_memory=False)
        tmp.set_index('Date', inplace=True)
        self.bench = self.mu*(1+tmp.pct_change()).cumprod()
        #self.bench = self.mu * np.diff(tmp, axis=0).cumsum()
        print('bench', self.bench)
        pd.DataFrame(self.bench).to_csv('bench.csv')

    def load_csv(self, fname):
        tmp = pd.read_csv(fname, header=0, low_memory=False)
        print(tmp.head())
        print(tmp.shape)
        # tmp.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        tmp_tstr = tmp['Unnamed: 0']
        #tmp_t = [dt.strptime(tmp_tstr[i], '%Y.%m.%d') for i in range(len(tmp_tstr))]
        tmp_t = [dt.strptime(tmp_tstr[i], '%m/%d/%y') for i in range(len(tmp_tstr))]
        #tmp_t = [dt.strptime(tmp_tstr[i], '%Y-%m-%d') for i in range(len(tmp_tstr))]
        tmp_p = tmp.iloc[:, 1:]
        self.all_t = np.array(tmp_t) # [::-1]
        self.all_p = np.array(tmp_p)#.reshape((1, -1))[0] # [::-1]
        print('all_p shape', self.all_p.shape)

    def load_csv_test(self, fname):
        tmp = pd.read_csv(fname, header=0, low_memory=False)
        print(tmp.head())
        print(tmp.shape)
        tmp.replace(0, np.nan, inplace=True)
        tmp.dropna(axis=1, how='any', inplace=True)
        print('effect check', tmp.shape)
        # tmp.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        tmp_tstr = tmp['Unnamed: 0']
        #tmp_t = [dt.strptime(tmp_tstr[i], '%Y.%m.%d') for i in range(len(tmp_tstr))]
        #tmp_t = [dt.strptime(tmp_tstr[i], '%m/%d/%y') for i in range(len(tmp_tstr))]
        tmp_t = [dt.strptime(tmp_tstr[i], '%Y-%m-%d') for i in range(len(tmp_tstr))]
        tmp_p = tmp.iloc[:, 1:]
        self.all_t = np.array(tmp_t) # [::-1]
        self.all_p = np.array(tmp_p)#.reshape((1, -1))[0] # [::-1]
        print('all_p shape', self.all_p.shape)

    def quant(self, f):
        fc = f.copy()
        fc[np.where(np.abs(fc) < self.q_threshold)] = 0
        #return np.sign(fc)
        return fc

    def softmax(self, x):
        l2_norm = np.sqrt(x*x).sum()
        return x/l2_norm
        #e_x = np.exp(x)
        #return e_x / e_x.sum()

    def set_t_p_r(self):
        #self.t = self.all_t[self.init_t:self.init_t + self.T + self.M + 1]
        #self.p = self.all_p[self.init_t:self.init_t + self.T + self.M + 1,:] ## TODO: add column dimension for assets > 1
        self.t = self.all_t[self.init_t:self.init_t + self.T + self.M + 1]
        self.p = self.all_p[self.init_t:self.init_t + self.T + self.M + 1,
                 :]  ## TODO: add column dimension for assets > 1
        print('p dimension', self.p)
        #self.r = -np.diff(self.p, axis=0)
        firstr = np.zeros((1, self.p.shape[1]))
        self.r = np.diff(self.p, axis=0)/self.p[:-1]
        self.r = np.concatenate((firstr, self.r), axis=0)
        print('r dimension', self.r.shape)
        print('r dimension', self.r)
        pd.DataFrame(self.r).to_csv("smallr.csv", header=False, index=False)

    def set_x_F(self):
        for i in range(self.T - 1, -1, -1):
            self.x[i] = np.zeros(self.M + 2)
            self.x[i][0] = 1.0
            self.x[i][self.M + 2 - 1] = self.F[i+1,-1] ## TODO: i used -1 on column
            for j in range(1, self.M + 2 - 1, 1):
                #self.x[i][j] = self.r[i+ j - 1,0] ## TODO: i used -1 on column:
                self.x[i,j] = self.r[i + (j-1), -1]  ## TODO: i used -1 on column; and must deal with j
            self.F[i] = self.quant(np.tanh(np.dot(self.x[i], self.w)+self.b[i]))   ## TODO: test this

    def calc_R(self):
        #self.R = self.mu * (np.dot(self.r[:self.T], self.F[:,1:]) - self.sigma * np.abs(-np.diff(self.F, axis=1)))
        #self.R = self.mu * (self.r[:self.T] * self.F[1:]) - self.sigma * np.abs(-np.diff(self.F, axis=0))
        #self.R = self.mu * (np.multiply(self.F[1:,], np.reshape(self.r[:self.T], (self.T, -1)))) * (self.sigma) * np.abs(-np.diff(self.F, axis=0))
        self.R = ((np.multiply(self.F[1:, ], np.reshape(0+self.r[:self.T], (self.T, -1)))) * (1-self.sigma * np.abs(-np.diff(self.F, axis=0))))
        pd.DataFrame(self.R).to_csv('R.csv')

    def calc_sumR(self):
        self.sumR = np.cumsum(self.R[::-1], axis=0)[::-1] ## TODO: cumsum axis
        #self.sumR = np.cumprod(self.R[::-1], axis=0)[::-1]  ## TODO: cumsum axis
        self.sumR2 = np.cumsum((self.R[::-1] ** 2), axis=0)[::-1] ## TODO: cumsum axis
        #self.sumR2 = np.cumprod((self.R[::-1] ** 2), axis=0)[::-1]  ## TODO: cumsum axis
        #print('cumprod', self.sumR)

    def calc_dSdw(self):
        self.set_x_F()
        self.calc_R()
        self.calc_sumR()

        self.Sall = np.empty(0)  # a list of period-to-date sharpe ratios, for all n investments
        self.dSdw = np.zeros((self.M + 2, self.N))
        for j in range(self.N):
            self.A = self.sumR[0,j] / self.T
            self.B = self.sumR2[0,j] / self.T
            #self.A = self.sumR / self.T
            #self.B = self.sumR2 / self.T
            self.S = self.A / np.sqrt(self.B - (self.A ** 2))
            #self.S = ((self.B[1:,j]*np.diff(self.A[:,j], axis=0)-0.5*self.A[1:,j]*np.diff(self.B[:,j], axis=0))/ (self.B[1,j] - (self.A[1,j] ** 2))**(3/2))[1]
            #self.S = (self.B[1,j] - (self.A[1,j] ** 2))**(3/2)
            #print('sharpe checl', np.isnan(self.r).sum())
            self.dSdA = self.S * (1 + self.S ** 2) / self.A
            self.dSdB = -self.S ** 3 / 2 / self.A ** 2
            self.dAdR = 1.0 / self.T
            self.dBdR = 2.0 / self.T * self.R[:,j]
            self.dRdF = -self.mu * self.sigma * np.sign(-np.diff(self.F, axis=0))
            self.dRdFp = self.mu * self.r[:self.T] + self.mu * self.sigma * np.sign(-np.diff(self.F, axis=0))  ## TODO: r needs to be a matrix if assets > 1
            self.dFdw = np.zeros(self.M + 2)
            self.dFpdw = np.zeros(self.M + 2)
            #self.dSdw = np.zeros((self.M + 2, self.N))  ## TODO: should not have put this here. this resets everytime
            self.dSdw_j = np.zeros(self.M + 2)
            for i in range(self.T - 1, -1, -1):
                if i != self.T - 1:
                    self.dFpdw = self.dFdw.copy()
                self.dFdw = (1 - self.F[i,j] ** 2) * (self.x[i] + self.w[self.M + 2 - 1,j] * self.dFpdw)
                self.dSdw_j += (self.dSdA * self.dAdR + self.dSdB * self.dBdR[i]) * (
                    self.dRdF[i,j] * self.dFdw + self.dRdFp[i,j] * self.dFpdw)
            self.dSdw[:, j] = self.dSdw_j
            self.Sall = np.append(self.Sall, self.S)

    def update_w(self):
        self.w += self.rho * self.dSdw

    def get_investment_weights(self):
        for i in range(self.FS.shape[0]):
            self.FS[i] = np.multiply(self.F[i], self.Sall)
        tmp = np.apply_along_axis(self.softmax, 1, self.FS)
        return np.apply_along_axis(self.select_n, 1, tmp)

    def select_n(self, array):
        threshold = max(heapq.nlargest(self.TOP, array)[-1], self.threshold)
        new_array = [x if x >= threshold else 0 for x in array]
        return new_array

    def fit(self):
        pre_epoch_times = len(self.epoch_S)

        self.calc_dSdw()
        print("Epoch loop start. Initial sharp's ratio is " + str(np.mean(self.Sall)) + ".")
        print('s len', len(self.Sall))
        self.S_opt = self.Sall

        tic = time.clock()
        for e_index in range(self.n_epoch):
            self.calc_dSdw()
            if np.sum(self.Sall) >= np.sum(self.S_opt):
                self.S_opt = self.Sall
                self.w_opt = self.w.copy()
            #self.Sall = np.apply_along_axis(self.select_n, 0, self.Sall) # TODO: don't do this here
            self.epoch_S[e_index] = np.array(self.Sall)
            self.update_w()
            if e_index % self.progress_period == self.progress_period - 1:
                toc = time.clock()
                print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" + str(
                    self.n_epoch + pre_epoch_times) + ". Shape's ratio: " + str(self.Sall[self.Sall.nonzero()].mean()) + ". Elapsed time: " + str(
                    toc - tic) + " sec.")
        toc = time.clock()
        print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" + str(
            self.n_epoch + pre_epoch_times) + ". Shape's ratio: " + str(self.Sall[self.Sall.nonzero()].mean()) + ". Elapsed time: " + str(
            toc - tic) + " sec.")
        self.w = self.w_opt.copy()
        self.calc_dSdw()
        print("Epoch loop end. Optimized sharp's ratio is " + str(self.Sall[self.Sall.nonzero()].mean()) + ".")
        print('first check', self.Sall)
        print('now check', self.epoch_S)
        print('R dimension', self.R.shape)


    def save_weight(self):
        self.F1 = self.get_investment_weights()

        pd.DataFrame(self.w).to_csv("w.csv", header=False, index=False)
        self.epoch_S.to_csv("epoch_S.csv", header=False, index=False)
        pd.DataFrame(self.F).to_csv("f.csv", header=False, index=False)
        pd.DataFrame(self.FS).to_csv("fs.csv", header=False, index=False)
        pd.DataFrame(self.F1).to_csv("f1.csv", header=False, index=False)

    def load_weight(self):
        tmp = pd.read_csv("w.csv", header=None)
        self.w = tmp.T.values[0]

    def get_investment_sum(self, train_phase=True):
        firstR = np.zeros((1,self.p.shape[1]))
        self.R = np.concatenate((firstR, self.R), axis=0)
        tmp = np.multiply(self.R, self.F1)
        self.total = self.mu * ((1+tmp.sum(axis=1)).cumprod(axis=0))
        print('iam here', self.total.shape, self.total)
        if train_phase:
            pd.DataFrame(self.total).to_csv('investment_sum.csv')
        else:
            pd.DataFrame(self.total).to_csv('investment_sum_testphase.csv')


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
    #fname = 'USDJPY30.csv'
    bench = 'SPY.csv'
    fname = 'all_data.csv'

    init_t = 1001 #1001

    T = 1000
    M = 200
    N = 371 # TODO: no magic numbers!!! 371
    mu = 100
    sigma = 0.04
    rho = 1.0
    n_epoch = 100

    # RRL agent with initial weight.
    ini_rrl = TradingRRL(T, M, N, init_t-T, mu, sigma, rho, n_epoch)
    ini_rrl.load_csv_test(fname)
    ini_rrl.load_bench(bench)
    ini_rrl.set_t_p_r()
    ini_rrl.calc_dSdw()
    # RRL agent for training
    rrl = TradingRRL(T, M, N, init_t-T, mu, sigma, rho, n_epoch)
    rrl.all_t = ini_rrl.all_t
    rrl.all_p = ini_rrl.all_p
    rrl.set_t_p_r()
    rrl.fit()
    rrl.save_weight()
    rrl.get_investment_sum()

    # Plot results.
    # Training for initial term T.
    '''
    plt.plot(range(len(rrl.epoch_S)), rrl.epoch_S)
    plt.title("Sharp's ratio optimization")
    plt.xlabel("Epoch times")
    plt.ylabel("Sharp's ratio")
    plt.grid(True)
    plt.savefig("sharp's ratio optimization.png", dpi=300)
    plt.close
    '''

    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
    ax[0].plot(ini_rrl.bench[:rrl.T], color='red', label='Benchmark')
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("SPY")
    ax[0].grid(True)

    ax[1].plot(ini_rrl.bench[:rrl.T], color='red', label='Benchmark')
    ax[1].plot(rrl.total, color="blue", label="With optimized weights")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Total Invested")
    ax[1].legend(loc="upper left")
    ax[1].grid(True)

    plt.savefig("rrl_training.png", dpi=300)

'''
    # Prediction for next term T with optimized weight.
    # RRL agent with initial weight.
    ini_rrl_f = TradingRRL(T, M, N, init_t, mu, sigma, rho, n_epoch)
    ini_rrl_f.all_t = ini_rrl.all_t
    ini_rrl_f.all_p = ini_rrl.all_p
    ini_rrl_f.set_t_p_r()
    ini_rrl_f.calc_dSdw()
    # RRL agent with optimized weight.
    rrl_f = TradingRRL(T, M, N, init_t, mu, sigma, rho, n_epoch)
    rrl_f.all_t = ini_rrl.all_t
    rrl_f.all_p = ini_rrl.all_p
    rrl_f.set_t_p_r()
    rrl_f.w = rrl.w
    rrl_f.F1 = rrl.F1
    rrl_f.calc_dSdw()
    rrl_f.get_investment_sum(train_phase=False)


    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
    ax[0].plot(ini_rrl.bench[:rrl.T], color='red', label='Benchmark')
    ax[0].plot(ini_rrl.bench[rrl.T:], color='purple', label='Benchmark')
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("USDJPY: benchmark")
    ax[0].grid(True)

    ax[1].plot(ini_rrl.bench[:rrl.T], color='red', label='Benchmark: before day 1000')
    ax[1].plot(ini_rrl.bench[rrl.T:], color='purple', label='Benchmark: after day 1000')
    ax[1].plot(rrl.total, color="blue", label="With optimized weights: before day 1000")
    ax[1].plot(rrl_f.total, color="blue", label="With optimized weights: after day 1000")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("Total Investment")
    ax[1].legend(loc="upper left")
    ax[1].grid(True)

    plt.savefig("rrl_prediction.png", dpi=300)
    fig.clear()
'''

if __name__ == "__main__":
    main()