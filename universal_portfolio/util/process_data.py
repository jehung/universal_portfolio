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

    Res['c_2_o'] = get_zscore(ret(D.Open,D.Close))
    Res['h_2_o'] = get_zscore(ret(D.Open,D.High))
    Res['l_2_o'] = get_zscore(ret(D.Open,D.Low))
    Res['c_2_h'] = get_zscore(ret(D.High,D.Close))
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
            Res = make_inputs(filepath)
            all = all.append(Res)

    pivot_columns = all.columns[:-1]
    P = all.pivot_table(index=all.index, columns='ticker', values=pivot_columns)  # Make a pivot table from the data
    return P


def embed(df):
    mi = df.columns.tolist()
    new_ind = pd.Index(e[1] + '_' + e[0] for e in mi)
    df.columns = new_ind
    clean_and_flat = df.dropna(1)
    target_cols = list(filter(lambda x: 'c1_c0' in x, clean_and_flat.columns.values))
    input_cols = list(filter(lambda x: 'c1_c0' not in x, clean_and_flat.columns.values))
    inputDF = clean_and_flat[input_cols]
    targetDF = clean_and_flat[target_cols]

    return inputDF, targetDF


def labeler(x):
    if x>0.0029:
        return 1
    if x<-0.00462:
        return -1
    else:
        return 0



def process_target(df):
    TotalReturn = ((1 - np.exp(df)).sum(axis=1)) / len(df.columns)  # If i put one dollar in each stock at the close, this is how much I'd get back

    Labeled = pd.DataFrame()
    Labeled['return'] = TotalReturn
    Labeled['class'] = TotalReturn.apply(labeler, 1)
    Labeled['multi_class'] = pd.qcut(TotalReturn, 11, labels=range(11))
    pd.qcut(TotalReturn, 5).unique()

    return Labeled



if __name__ == "__main__":
    all = merge_all_data(datapath)
    inputdf, targetdf = embed(all)
    labeled = process_target(targetdf)

    print(inputdf.head())
    print(labeled.head())
