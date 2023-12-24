import pandas as pd
from scalecast.Forecaster import Forecaster
from scalecast import GridGenerator
from datetime import datetime as dt, timedelta as td
import pytz
GridGenerator.get_example_grids() # example hyperparameter grids

df = pd.read_csv("Data\GOOGLE.csv")
# df.rename(columns = {'time':'DATE'}, inplace = True)   
convert_to_date = lambda s: dt.fromisoformat(s).astimezone(pytz.utc).date()
try:
    df[df.columns[0]] = df[df.columns[0]].apply(convert_to_date)
except:
    pass
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
print(df[df.columns[0]].min(), df[df.columns[0]].max())
f = Forecaster(
   y = df['values'], # required
   current_dates = df['date'], # required
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