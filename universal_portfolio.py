import pandas as pd
from scipy.ndimage.interpolation import shift
import pandas_datareader.data as web
import datetime
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pylab


def initialize(start, end, tickers):
    start = start
    end = end

    X = pd.DataFrame()

    for ticker in tickers:
        try:
            data = web.DataReader(ticker, 'yahoo', start, end)
        except IOError:
            pass

        data["Ticker"] = ticker
        data["Ratio"] = data["Close"].astype("float64") / data["Open"].astype("float64")
        X[ticker] = data["Ratio"]
        X.fillna(0)

    (r, c) = X.shape
    return (X, r)

tickers = ["COST", "MSFT", "SBUX"]
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2016, 6, 30)


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


def analyze(df):
    S = np.ones(r) * 0
    n = len(tickers)
    num_bin = 20
    B = binnings(num_bin, n)/num_bin
    S = np.prod(np.dot(B, X.transpose()), axis=1)
    S_alldays = np.cumprod(np.dot(B, X.transpose()), axis=1)
    num = np.dot(S_alldays.transpose(), B)
    #denom = sum(S.transpose())
    denom = np.sum(S_alldays.transpose(), axis=1)
    bk = num/denom[:, None]
    i = 0
    return B, S, bk

X, r = initialize(start, end, tickers)
B, S, bk = analyze(X)
print bk

def update_weights(tickers, bk, seed):
    weights = dict.fromkeys(tickers, 1/len(tickers))
    value = dict.fromkeys(tickers, seed/len(tickers))
    portfolio_value = []
    share_values = {}
    i = 0
    for i in range(0, r):
        for ticker in tickers:
            share_value = []
            if seed*weights[ticker] - seed*max(0, bk[i, tickers.index(ticker)]) < 100:
                value[ticker] = seed*X[ticker][0:i, ]
                share_value.append(value[ticker])
            else:
                weights[ticker] = max(0, bk[i, tickers.index(ticker)])
                value[ticker] = sum(value.values())*weights[ticker]
                share_value.append(value[ticker])
            portfolio_value.append(sum(value.values()))
            share_values[ticker] = share_value
        i += 1

    return weights, value, portfolio_value, share_values

weights, value, portfolio_value, share_values = update_weights(tickers, bk, 12500)

'''
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(S, label="Universal Portfolio")
plt.plot(share_values['COST'], label="Costco")
plt.plot(share_values['MSFT'], label="Microsoft")
plt.plot(share_values['SBUX'], label="Starbucks")
pylab.legend(loc='upper left')
plt.show()

#print share_values['COST']
#print type(share_values['COST'])
'''
