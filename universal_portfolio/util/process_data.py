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
    pivot_columns = df.columns[:-1]
    P = df.pivot_table(index=df.index, columns='ticker', values=pivot_columns)  # Make a pivot table from the data

    mi = P.columns.tolist()
    new_ind = pd.Index(e[1] + '_' + e[0] for e in mi)
    P.columns = new_ind
    clean_and_flat = P.dropna(axis=1)
    print(clean_and_flat.head())
    target_cols = list(filter(lambda x: 'c1_c0' in x, clean_and_flat.columns.values))
    input_cols = list(filter(lambda x: 'c1_c0' not in x, clean_and_flat.columns.values))
    print('target_col', target_cols)
    print('input_cols', input_cols)
    inputDF = clean_and_flat[input_cols]
    targetDF = clean_and_flat[target_cols]

    TotalReturn = ((1 - np.exp(targetDF)).sum(axis=1)) / len(targetDF.columns)  # If i put one dollar in each stock at the close, this is how much I'd get back

    Labeled = pd.DataFrame()
    Labeled['return'] = TotalReturn
    Labeled['class'] = TotalReturn.apply(labeler, 1)
    Labeled['multi_class'] = pd.qcut(TotalReturn, 11, labels=range(11))
    pd.qcut(TotalReturn, 5).unique()

    return inputDF, Labeled


def labeler(x):
    if x>0.0029:
        return 1
    if x<-0.00462:
        return -1
    else:
        return 0


'''
if __name__ == "__main__":
    all = merge_all_data(datapath)
    inputdf, targetdf = embed(all)
    labeled = process_target(targetdf)

    print(inputdf.head())
    print(labeled.head())
'''