#JT MODEL
# LSTM - Multivariate Stock Price Prediction
import pandas as pd
import numpy as np
import pytz
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from datetime import datetime as dt, timedelta as td

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    df.rename(columns = {'time':'DATE'}, inplace = True)   
    convert_to_date = lambda s: dt.fromisoformat(s).astimezone(pytz.utc).date()
    try:
        df[df.columns[0]] = df[df.columns[0]].apply(convert_to_date)
    except:
        pass
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df[df.columns[0]] = df[df.columns[0]].dt.strftime("%Y%m%d") # sn.heatmap(df.corr()), df.set_index('DATE', inplace=True)

    test_sample_size = int(df.shape[0]*.0057)
    return df, test_sample_size

def prepare_multivariate_data(df, window_size, test_sample_size):
    global training_set_scaled, scaler, test_set_scaled, training_set, test_set
    training_set = df.iloc[:df.shape[0]-test_sample_size, 1:4].values
    test_set = df.iloc[df.shape[0]-test_sample_size:, 1:4].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(training_set)
    test_set_scaled = scaler.fit_transform(test_set)
    test_set_scaled = test_set_scaled[:, 0:2]
    X_train, y_train = [], []
    for i in range(window_size, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - window_size:i, 0:3])
        y_train.append(training_set_scaled[i, 2])
    X, y = np.array(X_train), np.array(y_train)
    X = np.reshape(X,(X.shape[0], X.shape[1], 3))

    return X, y, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=70, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=70, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 70, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=70))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error') #optimizer = Adam(learning_rate=0.001)
    return model

def make_predictions(model, X_test, scaler):
    predictions_scaled = model.predict(X_test)
    predictions_scaled = predictions_scaled.reshape(predictions_scaled.shape[0], -1) # Reshape predictions_scaled to match the shape of X_test[:, -1, :-1]
    X_test_reshaped = X_test[:, -1, :-1] # Reshape X_test to have the same number of features as predictions_scaled
    assert X_test_reshaped.shape[0] == predictions_scaled.shape[0] # Ensure the number of rows matches
    predictions = scaler.inverse_transform(np.hstack((X_test_reshaped, predictions_scaled)))
    return predictions[:, -1]

def evaluate_model(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Main Function
if __name__ == '__main__':
    data_path = "Data\Electricity+Consumption.csv"#'Data/LTCUSD.csv'
    # data_path = "Data\COINBASE_LTCUSD, 1D_beed4.csv"#'Data/LTCUSD.csv'
    window_size = 24
    epochs = 80 #100
    batch_size = 32 #32
    df, test_sample_size = load_and_preprocess_data(data_path)
    X, y, scaler = prepare_multivariate_data(df, window_size, test_sample_size)
    input_shape = (X.shape[1], X.shape[2])
    model = build_lstm_model(input_shape)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)


    plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.show()
    model.save('LSTM - Multivariate')
    Model = load_model('LSTM - Multivariate')
    prediction_test = []
    Batch_one = training_set_scaled[-24:]
    Batch_New = Batch_one.reshape((1,24,3))

    for i in range(test_sample_size):
        First_pred = Model.predict(Batch_New)[0]
        prediction_test.append(First_pred)
        New_var = test_set_scaled[i,:]
        New_var = New_var.reshape(1,2)
        New_test = np.insert(New_var, 2, [First_pred], axis =1)
        New_test = New_test.reshape(1,1,3)
        Batch_New = np.append(Batch_New[:,1:,:], New_test, axis=1)
    prediction_test = np.array(prediction_test)
    SI = MinMaxScaler(feature_range = (0,1))
    y_Scale = training_set[:,2:3]
    SI.fit_transform(y_Scale)
    predictions = SI.inverse_transform(prediction_test)
    real_values = test_set[:, 2] #Problem child
    plt.plot(real_values, color = 'red', label = 'Actual Electrical Consumption')
    plt.plot(predictions, color = 'blue', label = 'Predicted Values')
    plt.title('Electrical Consumption Prediction')
    plt.xlabel('Time (hr)')
    plt.ylabel('Electrical Demand (MW)')
    plt.legend()
    plt.show()

    RMSE = math.sqrt(mean_squared_error(real_values,predictions))
    def mean_absolute_percentage_error (y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    MAPE = mean_absolute_percentage_error(real_values,predictions)
