import pandas as pd
from scipy.ndimage.interpolation import shift
import pandas_datareader.data as web
import datetime
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
from yahoo_finance import Share
import re


def initialize(start, tickers):
    start = start

    X = pd.DataFrame()

    for ticker in tickers:
        print ticker
        #try:
        data = web.DataReader(ticker, 'yahoo', start)
        data["Ticker"] = ticker
        data["Ratio"] = np.log(data["Adj Close"]).diff(periods=1)+1
        X[ticker] = data["Ratio"]

    (r, c) = X.shape
    return X.fillna(1), r, c

tickers = ['F00000GZCI.TO', 'XEI.TO', 'TLT', 'FIE.TO']
start = datetime.datetime(2015, 12, 29)



# Credits to smart implementaion by the user 'bar' from stackoverflow
# http://stackoverflow.com/questions/6750298/efficient-item-binning-algorithm-itertools-numpy
def binnings(n, k, cache={}):
    if n == 0:
        return np.zeros((1, k))
    if k == 0:
        return np.empty((0, 0))
    args = (n, k)
    if args in cache:
        return cache[args]
    a = binnings(n - 1, k, cache)
    a1 = a + (np.arange(k) == 0)
    b = binnings(n, k - 1, cache)
    b1 = np.hstack((np.zeros((b.shape[0], 1)), b))
    result = np.vstack((a1, b1))
    cache[args] = result
    return result


def permute(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def build_dist(df):
    df = df.apply(np.random.normal, axis=0)
    return df

def resample(df, n=None):
    if n is None:
        n = len(X)
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X.iloc[resample_i, :]
    X_resample = X_resample.reset_index(drop=True)
    return X_resample

#X, r, c = initialize(start, tickers)
#x1 = resample(X, r)
#print x1


def analyze(df):
    S = np.ones(r) * 0
    n = len(tickers)-1
    num_bin = 20
    B = binnings(num_bin, n)/num_bin
    S_alldays = np.cumprod(np.dot(B, df.iloc[:, 1:len(tickers)].transpose()), axis=1)
    num = np.dot(S_alldays.transpose(), B)
    denom = np.sum(S_alldays.transpose(), axis=1)
    bk = num/denom[:, None]
    return bk

X, r, c = initialize(start, tickers)

print 'ans', X
bk = analyze(X.fillna(0))
print bk


def update_weights(tickers, r, c, seed, bk, df):
    weights = np.ones((r, c))/(len(tickers)-1)
    traditional_values = seed*weights[:, 1:len(tickers)]*np.cumprod(df.iloc[:, 1:len(tickers)], axis=0)
    bench = seed*(1)*1*np.cumprod(df.iloc[:, 0], axis=0)
    #bench = seed * (1) * 1 * np.cumprod(benchmark.iloc[:,7], axis=0)
    universal_values = seed*bk*np.cumprod(df.iloc[:, 1:len(tickers)], axis=0)
    portfolio_value_t = np.sum(traditional_values, axis=1)
    portfolio_value_u = np.sum(universal_values, axis=1)
    vol_portfolio_value_t = pd.expanding_std(portfolio_value_t, min_periods=1)/pd.expanding_mean(portfolio_value_t, min_periods=1)
    vol_portfolio_value_u = pd.expanding_std(portfolio_value_u, min_periods=1)/pd.expanding_mean(portfolio_value_t, min_periods=1)
    vol_bench = pd.expanding_std(bench, min_periods=1)/pd.expanding_mean(portfolio_value_t, min_periods=1)
    #print 'vol_portfolio_value_t', vol_portfolio_value_t
    #print 'vol_portfolio_value_u', vol_portfolio_value_u
    #cbcprint 'vol_bench', vol_bench
    return traditional_values, universal_values, bench, portfolio_value_t, portfolio_value_u, vol_portfolio_value_t, vol_portfolio_value_u, vol_bench

#X1 = permute(X)
#bk = analyze(X1)
#traditional_values, universal_values, bench, portfolio_value_t, portfolio_value_u, vol_portfolio_value_t, vol_portfolio_value_u, vol_bench = update_weights(tickers, r, c, 12500, bk, X1)


no_sim = 100

for i in range(no_sim):
    #X1 = permute(X)
    #X1 = build_dist(X)
    X1 = resample(X, r)
    #bk = analyze(X)
    bk = analyze(X1)
    traditional_values, universal_values, bench, portfolio_value_t, portfolio_value_u, vol_portfolio_value_t, vol_portfolio_value_u, vol_bench = update_weights(tickers, r, c, 12500, bk, X1)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(portfolio_value_t, color='blue', label="Traditional portfolio")
    plt.plot(portfolio_value_u, color='green', label="Universal Portfolio")
    plt.plot(bench, color='red', label="Benchmark")
    # plt.plot(universal_values[tickers[0]], label=tickers[0])
    # plt.plot(universal_values[tickers[1]], label=tickers[1])
    # plt.plot(universal_values[tickers[2]], label=tickers[2])
    # plt.plot(universal_values[tickers[3]], label=tickers[3])
    #pylab.legend(loc='upper left')
    plt.subplot(212)
    plt.plot(vol_portfolio_value_t, color='blue', label="Vol-Traditional portfolio")
    plt.plot(vol_portfolio_value_u, color='green', label="Vol-Universal Portfolio")
    plt.plot(vol_bench, color='red', label="Vol-Benchmark")
plt.show()
