# !pip install pandas yfinance seaborn matplotlib tensorflow keras sklearn plotly

#### Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 2: Download stock data and visualise the data
# Check the file download path
#   %pwd

# Download the stock data for NFLX using yfinance.
data = yf.download('NFLX', start='2016-01-01', end='2023-06-22')
data.to_csv("NFLX_2023_06_22.CSV", index=True)

# Visualize the ‘Close Price History’ data using Plotly
##### You can use seaborn, matplotlib or plotly to visualise the data. In this example, I used plotly to illustrate the close price history of stock ‘NFLX’.
# Create the figure and add a line trace
fig = go.Figure(data=go.Scatter(x=data.index, y=data['Close'], mode='lines'))

# Set the layout
fig.update_layout(
    title='Stock:NFLX - Close Price History',
    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title='Close Price USD ($)', showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False,

)

# Show the figure
fig.show()

# Step 3: Prepare the data for LSTM
#### Before feeding the data into an LSTM model, you need to scale it and create a training data set:
# Create a new dataframe with only the 'Close' column
data = data.filter(['Close'])
​
# Convert the dataframe to a numpy array
dataset = data.values
​
# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * .95 ))
​
# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
​
# Create the training data set
train_data = scaled_data[0:int(training_data_len), :]
​
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
​
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Step 4: Build and train the LSTM model
# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=7)

# Step 5: Test the model accuracy on existing data and visualise tested data
#### After training the model, you can test its accuracy on the existing data:
# Prepare test data
test_data = scaled_data[training_data_len - 60:, :]

# Prepare the input sequences and target values for testing
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
    
# Convert the input sequences and target values to NumPy arrays
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Use the model to make predictions on the test data
predictions = model.predict(x_test)

# Transform the predictions and actual values back to their original scales
predictions = scaler.inverse_transform(predictions)
y_test = np.asarray(y_test).reshape(-1,1)
y_test = scaler.inverse_transform(y_test)

# Calculate the root mean square error (rmse)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))

print(f'rmse:{rmse}')

# Visualise the trained model with tested data using Plotly
#### Finally, you can plot the data to see how well the model fits the existing data:
# Splitting the Data and adding Predictions to Validation Data:
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Creating the Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Val'))
fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predictions'))

# Set Layout
fig.update_layout(
    title='LSTM Model - Stock:NFLX - Trained Model',
    xaxis=dict(title='Date',showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title='Close Price USD ($)',showgrid=True, gridcolor='lightgray'),
    legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,

)

fig.show()

# Step 6: Predict Future Stock Price
# Get the last 60 day closing price
last_60_days = dataset[-60:]

# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

# Create an empty list
X_test = []

# Append the past 60 days
X_test.append(last_60_days_scaled)

# Convert the X_test data set to a numpy array
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price
pred_price = model.predict(X_test)

# Undo the scaling
pred_prices = scaler.inverse_transform(np.array(pred_price).reshape(-1, 1))

# Print the predicted prices
print(pred_prices)


# Visualise predicted stock price
#### Please note that this is a very basic LSTM model for time series prediction. There are many ways to improve this model, such as adding more layers to the model, using more epochs for training, or using more features for training data.
# Create traces
trace1 = go.Scatter(
    x=train.index,
    y=train['Close'],
    mode='lines',
    name='Train'
)
trace2 = go.Scatter(
    x=valid.index,
    y=valid['Close'],
    mode='lines',
    name='Val'
)
trace3 = go.Scatter(
    x=valid.index,
    y=valid['Predictions'],
    mode='lines',
    name='Predictions'
)

# Create a list of dates for the x-axis
dates = pd.date_range(start=pd.Timestamp.today(), periods=len(pred_prices))

# Add a scatter plot for the predicted prices
trace4 = go.Scatter(
    x=dates,
    y=pred_prices.flatten(),
    mode='lines',
    name='Predicted Prices'
)

data = [trace1, trace2, trace3, trace4]

# Edit the layout
layout = dict(
    title='LSTM Model - Stock:NFLX - Price Prediction',
    xaxis=dict(title='Date',showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title='Close Price USD ($)',showgrid=True, gridcolor='lightgray'),
    legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

fig = dict(data=data, layout=layout)
go.Figure(fig).show()