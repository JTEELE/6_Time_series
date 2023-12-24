import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster
from scalecast import GridGenerator
import pytz
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt, timedelta as td
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM
prediction_days = 5

def load_and_preprocess_data(filename, prediction_days):
    df = pd.read_csv(filename).set_index('Ticker')
    convert_to_date = lambda s: dt.fromisoformat(s).astimezone(pytz.utc).date()
    try:
        df[df.columns[0]] = df[df.columns[0]].apply(convert_to_date)
    except:
        pass
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['values'].values.reshape(-1, 1))
    X_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler, df

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    return model

def train_model(X_train, y_train, epochs=50, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, history

def scalecast_model(dataframe):
    f = Forecaster(
    y = dataframe['values'], # required
    current_dates = dataframe['date'], # required
    future_dates = 24, # length of the forecast horizon
    test_length = 0, # set a test set length or fraction to validate all models if desired
    cis = False, # choose whether or not to evaluate confidence intervals for all models
    metrics = ['rmse','mae','mape','r2'], # the metrics to evaluate when testing/tuning models
    )
    f.set_estimator('xgboost') # select an estimator
    f.auto_Xvar_select() # find best look-back, trend, and seasonality for your series
    f.cross_validate(k=3) # tune model hyperparams using time series cross validation
    f.auto_forecast() # automatically forecast with the chosen Xvars and hyperparams
    results = f.export(['lvl_fcsts','model_summaries']) # extract results
    print(results)

if __name__ == "__main__":
    name = input("Enter stock ticker: ")
    filename = f"Data/{name}.csv"
    prediction_days = 10
    
    X_train, y_train, scaler, df = load_and_preprocess_data(filename, prediction_days)
    
    model, history = train_model(X_train, y_train, epochs=50, batch_size=32)
    model.save(f"ML/{name}.h5")
    # Make predictions using the trained model
    y_pred = model.predict(X_train)
    # Inverse transform the scaled values to the original scale
    y_pred_original = scaler.inverse_transform(y_pred)
    y_actual_original = scaler.inverse_transform(y_train.reshape(-1, 1))
    
    # Create a plot to visualize actual vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual_original, label="Actual")
    plt.plot(y_pred_original, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Actual vs. Predicted Values")
    plt.legend()
    # plt.show()

    # Load model and predict next 3 days
    # model = tf.keras.models.load_model("ML/GOOG.h5")
    # y_pred = model.predict(X_train)
    # # Inverse transform the scaled values to the original scale
    # y_pred_original = scaler.inverse_transform(y_pred)
    # y_actual_original = scaler.inverse_transform(y_train.reshape(-1, 1))
    
    # # Create a plot to visualize actual vs. predicted values
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_actual_original, label="Actual")
    # plt.plot(y_pred_original, label="Predicted")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.title("Actual vs. Predicted Values")
    # plt.legend()
    # plt.show()

    loaded_model = load_model(f"ML/{name}.h5")
    # Predict the value for the next 3 days
    last_10_days = X_train[-1]  # Get the last 10 days of data
    predictions = []

    for _ in range(prediction_days):
        # Reshape the last_10_days to match the input shape of the model
        last_10_days = last_10_days.reshape(1, prediction_days, 1)
        # Use the loaded model to predict the next day's value
        next_day_prediction = loaded_model.predict(last_10_days)
        # Inverse transform the scaled value to the original scale
        next_day_prediction_original = scaler.inverse_transform(next_day_prediction)[0][0]
        # Append the prediction to the list of predictions
        predictions.append(next_day_prediction_original)
        # Update last_10_days for the next iteration
        last_10_days = np.append(last_10_days[0][1:], next_day_prediction).reshape(1, prediction_days, 1)
    # Print the predicted values for the next 3 days
    for i, prediction in enumerate(predictions, start=1):
        date = dt.now(pytz.utc) + td(days=i)
        print(f"Predicted value {i} day(s) from today ({date.date()}): {prediction}")

    
    scalecast_model(df)