import pandas as pd
from scalecast.Forecaster import Forecaster
from scalecast import GridGenerator
from datetime import datetime as dt, timedelta as td
import pytz
from fbprophet import Prophet
import pandas as pd


df = pd.read_csv("Data\GOOGLE.csv")
convert_to_date = lambda s: dt.fromisoformat(s).astimezone(pytz.utc).date()
try:
    df[df.columns[0]] = df[df.columns[0]].apply(convert_to_date)
except:
    pass
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])


# Sample data: Replace this with your actual DataFrame
df.columns = ['ds', 'y'] # Your DataFrame should have at least two columns: 'ds' for dates and 'y' for the values

# Initialize a Prophet model
model = Prophet()

# Fit the model with your DataFrame
model.fit(df)

# Create a DataFrame for future predictions
# For example, predicting for the next 30 days
future = model.make_future_dataframe(periods=30)

# Make predictions
forecast = model.predict(future)

# forecast is a DataFrame with a 'ds' column for dates and a 'yhat' column for predictions
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plotting (optional, requires matplotlib)
fig1 = model.plot(forecast)
