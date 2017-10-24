import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



datapath = '../util/stock_dfs/'


def get_ticker(x):
    return x.split('/')[-1].split('.')[0]


def ret(x, y):
    return np.log(y/x)


def get_zscore(x):
    return (x -x.mean())/x.std()


def make_inputs(filepath):
    D = pd.read_csv(filepath).set_index('Date')
    #D.index = pd.to_datetime(D.index,format='%Y-%m-%d') # Set the indix to a datetime
    Res = pd.DataFrame()
    ticker = get_ticker(filepath)

    #Res['c_2_o'] = get_zscore(ret(D.Open,D.Close))
    Res['h_2_o'] = get_zscore(ret(D.Open,D.High))
    Res['l_2_o'] = get_zscore(ret(D.Open,D.Low))
    #Res['c_2_h'] = get_zscore(ret(D.High,D.Close))
    Res['h_2_l'] = get_zscore(ret(D.High,D.Low))
    Res['c1_c0'] = ret(D.Close,D.Close.shift(-1)).fillna(0) #Tommorows return
    Res['vol'] = get_zscore(D.Volume)
    Res['ticker'] = ticker
    return Res


def merge_all_data(datapath):
    all = pd.DataFrame()
    for f in os.listdir(datapath):
        filepath = os.path.join(datapath,f)
        if filepath.endswith('.csv'):
            print(filepath)
            Res = make_inputs(filepath)
            all = all.append(Res)

    return all


def embed(df):
    "str: choice of return, class, multi_class"
    pivot_columns = df.columns[:-1]
    P = df.pivot_table(index=df.index, columns='ticker', values=pivot_columns)  # Make a pivot table from the data
    tmp = P.stack(0).reset_index()
    tmp.index = tmp.apply(lambda x:x['Date']+x['level_1'], axis=1)
    inputDF = tmp.dropna(axis=1)
    inputDF.drop(['Date', 'level_1'], axis=1, inplace=True)
    print(inputDF.head())
    targetDF = inputDF.apply(lambda x: np.percentile(x, 80), axis=1)
    #targetDF = targetDF.apply(lambda x:labeler(x))  ## convert to classification problem
    print(targetDF.head())

    return inputDF, targetDF



def labeler(x):
    if x>0:
        return 1
    else:
        return 0



if __name__ == "__main__":
    all = merge_all_data(datapath)
    a, b = embed(all)

