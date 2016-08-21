import pandas as pd
from scipy.ndimage.interpolation import shift
import pandas_datareader.data as web
import datetime
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

# How to Get a List of all NASDAQ Securities as CSV file using Python?
# +tested in Python 3.5.0b2
#
# (c) 2015 QuantAtRisk.com, by Pawel Lachowicz

'''
import os
import random
os.system("curl --ftp-ssl anonymous:jupi@jupi.com "
          "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt "
          "> nasdaq.lst")


os.system("tail -n +9 nasdaq.lst | cat | sed '$d' | sed 's/|/ /g' > "
          "nasdaq.lst2")

os.system("awk '{print $1}' nasdaq.lst2 > nasdaq.csv")
os.system("echo; head nasdaq.csv; echo '...'; tail nasdaq.csv")
'''
'''
import pandas as pd
data = pd.read_csv("nasdaq.csv", index_col=None, header=None)
data.columns=["Ticker"]
#print type(data)
dl = data['Ticker'].tolist()


def screen_ticker(x, start=datetime.datetime(2013, 1, 2), end=datetime.datetime.now()):
    result = []
    start = datetime.datetime(2013, 1, 3)
    end = datetime.datetime(2016, 7, 29)
    for xi in x:
        try:
            a = web.DataReader(xi, 'yahoo', start, end)
            if a.index[0] == start and a.index[-1] == end:
                result.append(xi)
        except:
            pass
    return result

data = screen_ticker(dl)
print data
import pickle
with open("data.bin", "wb") as f:
    pickle.dump(data, f)
'''

import pickle
with open("data.bin", "rb") as data:
    all_tickers = pickle.load(data)


#random.seed(576)
def sample(x, k):
    result = []
    for i in range(k):
        ki = random.randint(0, len(x))
        result.append(x[ki])
    return result
tickers = sample(all_tickers, 4)
#bench = 'F0CAN05OAE'
bench = '^GSPC'
tickers.append(bench)
print 'here', tickers


'''
def sample(iterator, k):
    """
    Samples k elements from an iterable object.

    :param iterator: an object that is iterable
    :param k: the number of items to sample
    """
    it = iter()
    # fill the reservoir to start
    result = [next(iterator) for _ in range(k)]

    n = k - 1
    for item in iterator:
        n += 1
        s = random.randint(0, n)
        if s < k:how tot
            result[s] = item

    return result
'''




def initialize(start, end, tickers):
    start = start
    end = end

    X = pd.DataFrame()

    for ticker in tickers:
        print ticker
        #try:
        data = web.DataReader(ticker, 'yahoo', start, end)
        data["Ticker"] = ticker
        data["Ratio"] = np.log(data["Close"]).diff(periods=1).fillna(0)+1
        X[ticker] = data["Ratio"]
        #except IOError:
        #    pass

        #data["Ticker"] = ticker
        #data["Ratio"] = data["Close"].astype("float64") / data["Open"].astype("float64")
        #X[ticker] = data["Ratio"]
        #X.fillna(0)

    (r, c) = X.shape
    return X, r, c

#tickers = ['COST', 'MSFT', 'SBUX', 'AMZN', '^GSPC']
#bench = 'F0CAN05OAE'
start = datetime.datetime(2013, 1, 2)
end = datetime.datetime(2016, 7, 29)


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
    n = len(tickers)-1
    num_bin = 20
    B = binnings(num_bin, n)/num_bin
    #S = np.prod(np.dot(B, X.transpose()), axis=1)
    S_alldays = np.cumprod(np.dot(B, X.iloc[:, :-1].transpose()), axis=1)
    num = np.dot(S_alldays.transpose(), B)
    #denom = sum(S.transpose())
    denom = np.sum(S_alldays.transpose(), axis=1)
    bk = num/denom[:, None]
    i = 0
    return bk

X, r, c = initialize(start, end, tickers)
print 'X', X

bk = analyze(X)
print 'bk', bk

def update_weights(tickers, r, c, seed, bk):
    weights = np.ones((r, c))/(len(tickers)-1)
    print weights
    traditional_values = seed*weights[:, :-1]*np.cumprod(X.iloc[:, :-1], axis=0)
    print 'a', traditional_values
    bench = seed*(1)*1*np.cumprod(X.iloc[:, -1], axis=0)
    print bench
    universal_values = seed*bk*np.cumprod(X.iloc[:, :-1], axis=0)
    print 'b', universal_values
    portfolio_value_t = np.sum(traditional_values, axis=1)
    portfolio_value_u = np.sum(universal_values, axis=1)
    return traditional_values, universal_values, bench, portfolio_value_t, portfolio_value_u

traditional_values, universal_values, bench, portfolio_value_t, portfolio_value_u = update_weights(tickers, r, c, 12500, bk)

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(portfolio_value_t, label="Traditional portfolio")
plt.plot(portfolio_value_u, label="Universal Portfolio")
plt.plot(bench, label="Benchmark")
#plt.plot(universal_values[tickers[0]], label=tickers[0])
#plt.plot(universal_values[tickers[1]], label=tickers[1])
#plt.plot(universal_values[tickers[2]], label=tickers[2])
#plt.plot(universal_values[tickers[3]], label=tickers[3])
pylab.legend(loc='upper left')
plt.show()

