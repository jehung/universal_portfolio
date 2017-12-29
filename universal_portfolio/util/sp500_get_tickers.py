import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    return tickers

tickers = save_sp500_tickers()
print(tickers)


def get_benchmark_from_yahoo(bench='SPY'):
    start = dt.datetime(2008, 1, 1)
    end = dt.datetime(2017, 12, 28)
    df = web.DataReader(bench, "yahoo", start, end)
    df.to_csv('../rrl_trading/01_python/SPY.csv'.format(bench))
    return df

def get_data_from_yahoo(reload_sp500=False):
    ticker_count = 0
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2008, 1, 1)
    end = dt.datetime(2017, 12, 28)

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        try:
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
            ticker_count += 1
        except:
            continue

    print('Total number of tickers (S&P500): ' + str(len(tickers)))
    print('Total number of processed tickers (S&P500): ' + str(ticker_count))


get_benchmark_from_yahoo()
get_data_from_yahoo(True)
