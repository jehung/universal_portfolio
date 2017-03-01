# universal_portfolio

The universal_portfolio is a Python-based script that allows users to specify a number of stock/mutual fund/exchanged-traded funds over an arbitrary period of time, to specify an investment strategy, and compare investment outcomes. 

This Python code seeks to find the investment outcomes of different strategies such as old-fashioned buy-and-hold, as well as active rebalancing - while comparing each of the investment outcome against a benchmark. This benchmark can be arbitrarily specified by a user.

## Required Packages

To run the Python code, you must have already installed `NumPy`, `Pandas`, `Matpotlib`, as well as `pandas-datareader`, and `yahoo-finance`.

## How to Run the Code

### Python IDE
Load the `.py` file in any Python IDE, and run.

### Command Line
`cd` into the directory where the package is saved or downloaded, and 
`python [filename.py]`

## How to Use the Code

In `universal_beat_this.py`, you can specify your own beanchmark investment, and further specify a list of tickers to invest in, as well as a starting date of the analysis. These values have been filled in by the project default values, but you are free to change it.

## Technical Background

The technical background was inspired by Thomas M. Cover's Universal Portfolio theory. He demonstrated an algorithm for portfolio selection that outperforms the best stock in the market by actively rebalancing the weights of the investments daily. However, his algorithm had not taken into consideration the transaction costs.

This Python code builds upon Cover's theory of actively balancing, and seeks to find the investment outcomes of different strategies such as old-fashioned buy-and-hold, as well as active rebalancing - while comparing each of the investment outcome against a benchmark.

## How to Contribute

I welcome your contribution! Portfolio selection is an NP-hard problem, and that can partially explain that the selection process is often an subjective process. Over time, the goal of this quantitative finance project is to show that it is very difficult, if not impossible, to outperform a diversified basket of investment under any strategy, and therefore mitigating the need for a choice set of investments.

## License 

The content of this repository is licensed under a
[Creative Commons Attribution License](http://creativecommons.org/licenses/by/3.0/us/)
