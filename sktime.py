import pandas as pd
import numpy as np
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.base import ForecastingHorizon

# Sample data generation (replace this with your actual dataset)
np.random.seed(0)
dates = pd.date_range(start="2023-05-10", periods=2711, freq="2H")
data = np.random.rand(2711, 3)  # Assuming the 3 features are numeric
df = pd.DataFrame(data, index=dates, columns=["Feature1", "Feature2", "Feature3"])

# Assuming the first feature is the target variable
y = df["Feature1"]

# Splitting the data into train and test sets
y_train, y_test = temporal_train_test_split(y, test_size=0.2)

# Create a forecaster - RandomForest wrapped as a forecaster
forecaster = make_reduction(RandomForestRegressor(), window_length=15, strategy="recursive")

# Fit the forecaster to the training data
forecaster.fit(y_train)

# Define the forecasting horizon (2 days into the future, 2-hour increments)
fh = ForecastingHorizon(pd.date_range(start=y_test.index[-1], periods=24, freq="2H"), is_relative=False)

# Generate predictions
y_pred = forecaster.predict(fh)

# Display the predictions
print(y_pred)
