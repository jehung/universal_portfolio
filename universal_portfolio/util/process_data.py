import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def load_data(results_path):
    """
    Process all downloaded files into one dataframe
    :param results_path: type string, denotes the name of directory where downloaded files are
    :return: pandas dataframe
    """
    files = os.listdir(results_path)
    results_dicts = []
    for file_name in files:

        full_path = os.path.join(results_path, file_name)
        with open(full_path, 'rb') as file_obj:
            data = pd.read_csv(file_obj)
            data.set_index('Date', inplace=True)
            data['Price_delta'] = data['Close']/data['Open']
            data['Vol_delta'] = data['Volume'].diff()
            data['Ticker'] = data.Volume.apply(lambda x:file_name.split('.')[0])
            results_dicts.append(data)
    a = pd.concat(results_dicts)

    return a



def set_params():
    """"

    :return:
    """

    pass


if __name__ == "__main__":
    data_path = os.path.join(os.curdir,'stock_dfs') ## this line to be changed to do different analysis
    ans = load_data(data_path)

