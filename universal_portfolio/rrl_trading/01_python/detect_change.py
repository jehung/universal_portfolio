# File               :   schangepoint.py
# Author             :   Sheheryar Sewani (@Sheysrebellion)
# Description        :   first attempt at implementing change point detection
#                   :   using CUMSUM estimation.  Based on the discussions on:
#                   :   http://www.variation.com/cpa/tech/changepoint.html
# Created            :   August 19, 2009
# temp

import numpy as np, random, sys
import pandas as pd
import matplotlib.pyplot as plt


def readfile(filename):
    """
    Description:    Expects a list of numbers seperated by the new line character
    Arguments:      filename
    Notes:          Does not check for errors
    """

    #lines = [float(line.strip()) for line in file(filename)]
    #return lines

    tmp = pd.read_csv(filename, header=0, low_memory=False)
    #tmp.set_index('Date', inplace=True)
    print(tmp.adj_close)
    return tmp.adj_close


def cumsums(data):
    """
    Description:    Calculates the cumulative sums
    Arguments:      data - a list of floats
    """

    series_average = np.average(data)
    csums = []
    csums.append(0)

    for i in range(0, len(data)):
        csums.append((data[i] - series_average) + csums[i])

    return csums


def randomize(data):
    # Magnus L Hetland's solution, (s4)
    # copied from
    # http://mail.python.org/pipermail/python-list/1999-August/009741.html
    # original was destructive, new makes a copy of the list.
    temp_list = []
    temp_list.extend(data)

    result = []
    for i in range(len(temp_list)):
        element = random.choice(temp_list)
        temp_list.remove(element)
        result.append(element)
    return result


def bootstrap(data, iterations=1000):
    """
    Description:    used, along with the confidence interval,
                    to detect if a change occurred int he series.
                    Creates 100 bootsrapped series, shuffles the original data list
                    then calculates the cumulative sum for each shuffled data series
    Arguments:      data - a list of floats
    Returns:        returns the confidence interval
    """

    cumsum_original = cumsums(data)
    sdiff_original = max(cumsum_original) - min(cumsum_original)

    # boot strap n samples
    bootstrapped_series = [randomize(data) for i in range(iterations)]

    # find cumumlative sums for the bootstrapped samples
    bootstrapped_cumsums = [cumsums(bootstrapped_series[i]) for i in range(len(bootstrapped_series))]
    x = [max(bootstrapped_cumsums[i]) - min(bootstrapped_cumsums[i]) for i in range(len(bootstrapped_series))]

    # find the number of bootstrapped series where
    # S_diff is < S_diff of the original series
    n = 0
    for i in range(len(x)):
        if (x[i] < sdiff_original):
            n = n + 1

    s = (n / float(iterations)) * 100.0
    return s


def find_index_of_maximum(cumsum):
    """
    Description:  Find the index of the maximum value from the cummulative sums
    """
    max_number = sys.float_info.min
    max_index = 0
    abs_vals = [abs(x) for x in cumsum]

    for (i, num) in enumerate(abs_vals):
        if num > max_number:
            max_number = num
            max_index = i

    return max_index


def get_changepoints(data, change_points, confidence_level=90, offset=0):
    """
    Description:    Call the function by passing a data series
                    Once a change has been detected, break the data into two segments,
                    one each side of the change-point, and the analysis repeated for each segment.
    Returns:        Indexes of change points detected in the data series
    """
    if not change_points:
        change_points = []

    confidence = bootstrap(data, 1000)
    if (confidence > confidence_level):
        cumsum = cumsums(data)
        max_index = find_index_of_maximum(cumsum)
        print('max_index', max_index)

        # add change point found to list
        # use offset to find the correct index based on original data
        change_points.extend([max_index + offset])

        # split the data into two, and calculate change points
        get_changepoints(data[:max_index], change_points, confidence_level, offset)
        print('change point first-part at ',  max_index)

        try:
            get_changepoints(data[max_index:], change_points, confidence_level, offset + max_index - 1)
        except:
            print('exception secondpart at ', max_index)
            pass
    return change_points


def main():
    fname = 'SPY.csv'
    data = readfile(fname)

    data.plot()
    plt.show()

    change_points = []
    points = get_changepoints(data, change_points)
    print(points)


if __name__ == '__main__':
    main()