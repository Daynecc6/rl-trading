import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(assets, start, end):
    data = yf.download(assets, start=start, end=end)['Adj Close']
    data.dropna(how='any', inplace=True)
    returns = data.pct_change().dropna()
    return data, returns

def split_data(returns_df, train_end="2020-12-31"):
    train = returns_df.loc[:train_end]
    test  = returns_df.loc[train_end:]
    return train, test
