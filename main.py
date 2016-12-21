import pandas_datareader.data as pdr
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

#wating on pull request to work with python 3 for now can install from https://github.com/woodsjc/finsymbols
import finsymbols

START = datetime(2016,11,1)
END = datetime.today()

def get_data(name='HPE'):
    data = pdr.DataReader(name, 'yahoo', START, END)
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    data.to_csv('./data/' + name + '.csv')
    print("Wrote ./data" + name + '.csv')
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

def moving_avg_window(d, timescale=5):
    tmp = np.zeros(shape=(len(d)-timescale,1))
    start = 0
    for x in range(timescale, len(d)):
        tmp[start] = d.ix[start:x].sum() / timescale
        start += 1 
    return pd.DataFrame(data=tmp, index=d[start:].index, columns=["moving average: " + d.name])

def dl_sp500_data():
    sp500 = finsymbols.get_sp500_symbols()
    data = []

    for s in sp500:
        if 'symbol' in s:
            data.append(get_data(s['symbol']))

    return data

def load_sp500_data():
    sp500 = finsymbols.get_sp500_symbols()
    data = []

    for s in sp500:
        if 'symbol' in s:
            symbol = s['symbol']
            tmp = load_data(symbol)
            if tmp is None:
                data.append(get_data(symbol))
            else:
                data.append(tmp)

    return data


stock_name = 'hpe'

#data = get_data(stock_name)
data = load_data(stock_name)

data["High"].subtract(data["Low"]).plot()
data["High"].subtract(data["Low"]).describe()
adj_y_limits()
plt.pause(.001)
plt.pause(.001)
plt.close()
plt.pause(.001)

data.ix[-10:]['Close'].plot()
plt.pause(.001)
pdr.get_quote_yahoo(stock_name)
plt.close()

plt.plot(moving_avg_window(data.ix[-10:]['Close']))
plt.pause(.001)
plt.pause(.001)
plt.close()
plt.pause(.001)

#sp500 = dl_sp500_data()
sp500 = load_sp500_data()

