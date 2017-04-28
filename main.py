﻿from datetime import datetime
import os
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
from pandas_datareader._utils import RemoteDataError
import matplotlib.pyplot as plt
#wating on pull request to work with python 3 for now can install from https://github.com/woodsjc/finsymbols
import finsymbols

START = datetime(2017,4,14)
END = datetime.today()

def get_data(stockName='HPE'):
    try:
        data = pdr.DataReader(stockName, 'yahoo', START, END)
    except RemoteDataError:
        data = pdr.DataReader(stockName.replace('.', '-'), 'yahoo', START, END)
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    data.to_csv('./data/{}.csv'.format(stockName))
    print("Wrote ./data/{}.csv".format(stockName))
    data.describe()
    return data

def load_data(name='hpe'):
    files = os.listdir('./data')
    for f in files:
        if len(name) <= len(f) and name == f[:len(name)]:
            return pd.read_csv('./data/' + f, parse_dates=['Date'], index_col=['Date'], dayfirst=True)
    return None

def adj_y_limits():
    a,b = plt.ylim()
    if a>0:
        plt.ylim(0,b+1)
    elif b<0:
        plt.ylim(a-1,0)
    plt.pause(.001)

def moving_avg_window(stockData, timeScale=5):
    movingAvg = np.zeros(shape=(len(stockData)-timeScale, 1))
    start = 0
    for x in range(timeScale, len(stockData)):
        movingAvg[start] = stockData.ix[start:x].sum() / timeScale
        start += 1 
    return pd.DataFrame(data=movingAvg,
                        index=stockData[-start:].index,
                        columns=["moving average: {}".format(stockData.name)])

def get_moving_avg(stockData, timescale=5):
    if len(stockData) < timescale:
        raise ValueError
    return stockData[-timescale:].sum() / timescale

def download_sp500_data():
    sp500 = finsymbols.get_sp500_symbols()
    data = []

    for stock in sp500:
        if 'symbol' in stock:
            data.append((get_data(stock['symbol']), stock['symbol']))

    return data

def load_sp500_data():
    sp500 = finsymbols.get_sp500_symbols()
    spData = []

    for stock in sp500:
        if 'symbol' in stock:
            symbol = stock['symbol']
            symbolData = (load_data(symbol), symbol)
            if symbolData is None:
                spData.append((get_data(symbol), symbol))
            else:
                spData.append(symbolData)

    return spData

def buy_or_sell(stock_list, current_bids, total_money):
    '''Start off simple with decision based on 5 day moving average and expectation to regress to 
    the mean. Then work on trends and other indicators. Eventually move to different file. Prices
    seem very stable for the little testing I did (less than 10% off moving average). Only 5% 
    picked up anything.
    '''
    for stock, symbol in stock_list:
        if stock is None:
            continue
        cur_price = stock.ix[-1]['Close']
        mov_avg = get_moving_avg(stock.ix[:-1]['Close'], 5)
        if cur_price < .95 * mov_avg:
            print("Buy: {} at ${:.2f} vs ${:.2f}".format(symbol, cur_price, mov_avg))
        elif cur_price > 1.05 * mov_avg:
            print("Sell: {} at ${:.2f} vs ${:.2f}".format(symbol, cur_price, mov_avg))


stock_name = 'hpe'

#data = get_data(stock_name)
data = load_data(stock_name)

data["High"].subtract(data["Low"]).plot()
data["High"].subtract(data["Low"]).describe()
adj_y_limits()
plt.close()

data.ix[-10:]['Close'].plot()
pdr.get_quote_yahoo(stock_name)
plt.close()

plt.plot(moving_avg_window(data.ix[-10:]['Close']))
plt.close()

#sp500 = download_sp500_data()
sp500 = load_sp500_data()

buy_or_sell(sp500, None, None)

##TODO##
#Add in logic for multi week comparisons so different algorithms/schemes can
#be tested with prior data