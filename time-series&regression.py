"""
Fx Rate Forecasting for Japanese Yen
"""
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from arch import arch_model
# %matplotlib inline
import warnings
import statsmodels.api as sm
import statsmodels.api as sm
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA 
import arch
from arch import arch_model
warnings.simplefilter(action='ignore', category=FutureWarning)
# This is the continuous chain of the futures contracts that are 1 month to expiration
yen_futures = pd.read_csv(
    Path("Data\drafts\Instructions\Starter_Code\yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head()


"""
 ### Return Forecasting: Initial Time-Series Plotting
"""
# Plot just the "Settle" column from the dataframe:
px.line(yen_futures['Settle'],title='Yen-Futures Settlement Price')


"""
### Decomposition Using a Hodrick-Prescott Filter
"""
# Split settle price between Trend & Noise
settle_noise, settle_trend = sm.tsa.filters.hpfilter(yen_futures["Settle"])
# Create a dataframe of just the settle price, and add columns for "noise" and "trend" series from above:
decomposition = pd.concat([yen_futures["Settle"],settle_noise,settle_trend], axis='columns', join='inner')
decomposition.dropna(inplace=True)
# Plot the Settle Price vs. the Trend for 2015 to the present
decomposition.loc['2015-01-01':].plot(kind='scatter', x='Settle', y='Settle_trend', figsize=(10,10), title='Settle PRICE since 2015')

# Plot the Settle Price vs. the Trend for 2015 to the present

fig = make_subplots(specs=[[{"secondary_y": True}]])


fig.add_trace(
    go.Scatter(x=decomposition['2015-01-01':].index, y=decomposition['Settle'], name="Settle"),
    row=1, col=1, secondary_y=False)

fig.add_trace(
    go.Scatter(x=decomposition['2015-01-01':].index, y=decomposition['Settle_trend'], name="Settle_trend"),
    row=1, col=1, secondary_y=True,
)
fig.show()


# Plot the Settle Price vs. the Trend for 2015 to the present
px.scatter(decomposition.loc['2015-01-01':], x="Settle", y="Settle_trend", title='Settle PRICE (using HVPLOT) since 2015')


# Plot the Settle Noise
px.line(decomposition, x=decomposition.index, y=decomposition['Settle'],title='Settle NOISE (using HVPLOT) since 2015')

"""
# Forecasting Returns using an ARMA Model

### Using futures Settle *Returns*, estimate an ARMA model
"""
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()

# Use the updated ARIMA model instead of ARMA
model = ARIMA(returns.values, order=(2, 0, 1))  # Note: ARIMA(p,d,q) where d is the degree of differencing
result = model.fit()
print(result.summary())

# Plot the 5 Day Returns Forecast
forecast = result.get_forecast(steps=5)
forecast_values = forecast.predicted_mean
px.line(forecast_values, title='Five Day ARIMA Forecast')




"""
### Forecasting the Settle Price using an ARIMA Model
"""
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA  # Import the updated ARIMA model

# Assuming yen_futures is a DataFrame already defined
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()

# Use the updated ARIMA model
arima = ARIMA(returns.values, order=(5, 1, 1))  # Updated order to (5, 1, 1) as per your code
arima_results = arima.fit()

# Output model summary results
print(arima_results.summary())

# Plot the 5 Day Price Forecast
forecast = arima_results.get_forecast(steps=5)
forecast_values = forecast.predicted_mean

# Creating a date range for forecast without 'closed' argument
last_date = returns.index[-1]
forecast_index = pd.date_range(start=last_date, periods=6, freq='B')[1:]  # 'B' stands for business day frequency

# Ensure the data is in a DataFrame for Plotly
forecast_df = pd.DataFrame(forecast_values, index=forecast_index, columns=['Forecast'])

# Use Plotly Express to plot the forecast
px.line(forecast_df, y='Forecast', title='Five Day ARIMA Forecast')

"""
### Volatility Forecasting with GARCH

Forecasting near-term **volatility** of Japanese Yen futures returns. 
"""

# Assuming returns is a DataFrame already defined
garch = arch_model(returns, mean='Zero', vol='GARCH', p=2, q=1)
garch_results = garch.fit(disp='off')
print(garch_results.summary())

# Calculate the last day of the dataset
last_day = returns.index.max()
# Create a 5-day forecast of volatility starting from the last day
forecast_horizon = 5
forecast = garch_results.forecast(start=last_day, horizon=forecast_horizon)

# Annualize the forecast
intermediate = np.sqrt(forecast.variance.dropna() * 252)

# Transpose the forecast for easier plotting
final = intermediate.dropna().T

# Ensure the index is a date range for Plotly
final.index = pd.date_range(start=last_day, periods=forecast_horizon, freq='B')

# Plot the final forecast using Plotly Express
px.line(final, title='Five Day Volatility Forecast')


print("""# Conclusions:
The ARIMA model suggests a futher decline in the prices over the predicted period.
However, the models have a significantly higher p-value, indicating the results are not reliable.
There is not enough information to determine whether buying the yen right now is a good decision.
""")




"### Regression Analysis: Seasonal Effects with Sklearn Linear Regression"


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import plotly.express as px
from _functions import *
yen_futures = pd.read_csv(
    Path("Data\drafts\Instructions\Starter_Code\yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures['Returns'] = (yen_futures['Settle'].pct_change()*100)
yen_futures['Returns'] = yen_futures['Returns'].replace(-np.inf, np.nan).dropna()
# print(f' Lagged returns is the independent variable (X), data type: {type(yen_futures['Returns']}'')
X_train, X_test, Y_train, Y_test = train_test(yen_futures)

in_sample, out_sample = LR_model(X_train,Y_train, X_test, Y_test)
in_rmse, out_rmse = rmse(in_sample,out_sample)


"### In-Sample Performance"
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=in_sample.index[-20:], y=in_sample['Returns'].tail(20), name="Actual Returns"),
    row=1, col=1, secondary_y=False)
fig.add_trace(
    go.Scatter(x=in_sample.index[-20:], y=in_sample['Predicted_return'].tail(20), name="Predicted_Return"),
    row=1, col=1, secondary_y=True,
)
fig.show()


"### Out-of-Sample Performance"

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=out_sample.index[-20:], y=out_sample['Returns'].tail(20), name="Actual Returns"),
    row=1, col=1, secondary_y=False)
fig.add_trace(
    go.Scatter(x=out_sample.index[-20:], y=out_sample['Predicted_return'].tail(20), name="Predicted_Return"),
    row=1, col=1, secondary_y=True,
)
fig.show()