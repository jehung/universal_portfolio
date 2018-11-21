#stage2 = '/Users/jennyhung/MathfreakData/Research/MyResearch/UniversalPortfolio/Code/universal_portfolio/universal_portfolio/util/stock_dfs'
#stage3 = '/Users/jennyhung/MathfreakData/Research/MyResearch/UniversalPortfolio/Code/universal_portfolio/universal_portfolio/rrl_trading/01_python'
import sys
import os
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
#sys.path.append(os.environ['WORKSPACE'] +
#                '/anaconda/bin/python')
#sys.path.append(stage2)
#sys.path.append(stage3)
print(sys.path)

import bs4 as bs
import datetime as dt
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


def get_benchmark_from_yahoo(bench='SPY'):
    start = '2008-01-01'
    end = dt.datetime.today().strftime('%Y-%m-%d')
    #df = web.DataReader(bench, "yahoo", start, end)
    df = web.get_data_yahoo(bench, start, end)
    #df.to_csv(stage3+'/SPY.csv'.format(bench))
    df.to_csv('SPY.csv'.format(bench))
    return df

def get_data_from_yahoo(reload_sp500=True):
    ticker_count = 0
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = '2008-01-01'
    end = dt.datetime.today().strftime('%Y-%m-%d')

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        try:
            df = web.get_data_yahoo(ticker, start, end)
            #df.to_csv(stage2+'/{}.csv'.format(ticker))
            df.to_csv('{}.csv'.format(ticker))
            ticker_count += 1
        except:
            continue

    print('Total number of tickers (S&P500): ' + str(len(tickers)))
    print('Total number of processed tickers (S&P500): ' + str(ticker_count))


today = dt.datetime.now()
tickers = save_sp500_tickers()
print(tickers)

get_benchmark_from_yahoo()
get_data_from_yahoo(True)
