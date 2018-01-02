import os
import pandas as pd

def get_ticker(x):
    return x.split('/')[-1].split('.')[0]


def get_benchmark():
    return 'SPY'


def read_file(file, test=None):
    d = pd.read_csv(file).set_index('Date')
    d.fillna(0, inplace=True)
    ticker = get_ticker(file)
    d.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': ticker,
                      'Volume': 'volume'},
             inplace=True)

    d.drop(labels=['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

    return d


def all_init_data():
    #filepath = stage2
    filepath = '/Users/Shared/Jenkins/Home/workspace/Test1/'
    alldata = []
    for f in os.listdir(filepath):
        datapath = os.path.join(filepath, f)
        print(datapath)
        if datapath.endswith('.csv') and 'SPY' not in datapath:
            print(datapath)
            Res = read_file(datapath)
            Res.reset_index(inplace=True)
            Res.drop_duplicates('Date', keep='first', inplace=True)
            Res.set_index('Date', inplace=True)
            alldata.append(Res)
    alldata = pd.concat(alldata, axis=1)
    alldata.fillna(0, inplace=True)
    alldata
    #alldata.to_csv('../rrl_trading/01_python/all_data_todate.csv')
    alldata.to_csv('all_data_todate.csv')


all_init_data()