import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def train_test(time_series):
    time_series['Lagged_returns'] = time_series.Returns.shift()
    time_series.dropna(inplace=True)
    train = time_series[:'2017']
    test = time_series['2018':]
    X_train = train.Lagged_returns.to_frame().dropna()
    X_test = test.Lagged_returns.to_frame().dropna()
    Y_train = train.Returns.dropna()
    Y_test = test.Returns.dropna()
    return X_train, X_test, Y_train, Y_test