import pandas as pd
import numpy as np
import pytz
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime as dt, timedelta as td

# Function to load and preprocess data
def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    df.rename(columns={'time': 'DATE'}, inplace=True)
    convert_to_date = lambda s: dt.fromisoformat(s).astimezone(pytz.utc).date()
    try:
        df[df.columns[0]] = df[df.columns[0]].apply(convert_to_date)
    except:
        pass
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df[df.columns[0]] = df[df.columns[0]].dt.strftime("%Y%m%d") # sn.heatmap(df.corr()), df.set_index('DATE', inplace=True)
    test_sample_size = int(df.shape[0] * 0.0057)
    return df, test_sample_size

# Function to prepare multivariate data
def prepare_multivariate_data(df, window_size, test_sample_size):
    data = df.iloc[:, 1:4].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X_train, y_train = [], []
    for i in range(window_size, len(data_scaled) - test_sample_size):
        X_train.append(data_scaled[i - window_size:i, :])
        y_train.append(data_scaled[i, 2])
    
    X, y = np.array(X_train), np.array(y_train)
    X = np.reshape(X, (X.shape[0], X.shape[1], 3))
    
    test_data = data_scaled[-test_sample_size:, :]
    test_set_scaled = test_data[:, 0:2]
    
    return X, y, test_set_scaled, scaler

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=70, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=70, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=70, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=70))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to make predictions
def make_predictions(model, X_test, scaler):
    predictions_scaled = model.predict(X_test)
    predictions_scaled = predictions_scaled.reshape(predictions_scaled.shape[0], -1)
    X_test_reshaped = X_test[:, -1, :-1]
    predictions = scaler.inverse_transform(np.hstack((X_test_reshaped, predictions_scaled)))
    return predictions[:, -1]

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

if __name__ == '__main__':
    data_path = "Data\Electricity+Consumption.csv"
    window_size = 24
    epochs = 80
    batch_size = 32
    
    # Load and preprocess data
    df, test_sample_size = load_and_preprocess_data(data_path)
    X, y, test_set_scaled, scaler = prepare_multivariate_data(df, window_size, test_sample_size)
    input_shape = (X.shape[1], X.shape[2])
    
    # Build the LSTM model
    model = build_lstm_model(input_shape)
    
    # Callbacks for saving the best model
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    callbacks = [checkpoint]
    
    # Training the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    
    # Load the best model
    best_model = load_model('best_model.h5')
    
    # Make predictions
    prediction_test = []
    Batch_one = X[-1]  # Start with the last window of training data
    for i in range(test_sample_size):
        First_pred = best_model.predict(Batch_one.reshape(1, window_size, 3))[0][0]
        prediction_test.append(First_pred)
        New_var = test_set_scaled[i, :]
        New_var = New_var.reshape(1, 2)
        New_test = np.insert(New_var, 2, [First_pred], axis=1)
        New_test = New_test.reshape(1, 1, 3)
        Batch_one = np.append(Batch_one[:, 1:, :], New_test, axis=1)
    
    # Inverse transform and plot predictions
    predictions = scaler.inverse_transform(np.array(prediction_test).reshape(-1, 1))
    real_values = df.iloc[-test_sample_size:, 3].values  # Assuming the 3rd column is the target variable
    plt.plot(real_values, color='red', label='Actual Electrical Consumption')
    plt.plot(predictions, color='blue', label='Predicted Values')
    plt.title('Electrical Consumption Prediction')
    plt.xlabel('Time (hr)')
    plt.ylabel('Electrical Demand (MW)')
    plt.legend()
    plt.show()
    
    # Calculate RMSE and MAPE
    RMSE = math.sqrt(mean_squared_error(real_values, predictions))
    MAPE = np.mean(np.abs((real_values - predictions) / real_values)) * 100
    
    print(f"Root Mean Squared Error (RMSE): {RMSE}")
    print(f"Mean Absolute Percentage Error (MAPE): {MAPE:.2f}%")
