import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def LR_model(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    out_sample = Y_test.to_frame()
    out_sample["Predicted_return"] = model.predict(X_test)
    in_sample = Y_train.to_frame()
    in_sample["Predicted_return"] = model.predict(X_train)
    return in_sample, out_sample


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

# mean_squared_error (MSE) & root-mean-squared error (RMSE, i.e. sqrt of MSE)
def rmse(in_sample, out_sample):
    in_rmse = mean_squared_error(
        in_sample['Returns'],
        in_sample['Predicted_return'])
    in_sample_rmse = np.sqrt(in_rmse)
    print(f'In-sample root mean squared error is {in_sample_rmse}')
    out_rmse = mean_squared_error(
        out_sample['Returns'],
        out_sample['Predicted_return'])
    out_of_sample_rmse = np.sqrt(out_rmse)
    print(f'Out-of-sample root mean squared error is {out_of_sample_rmse}')
    return in_rmse, out_rmse