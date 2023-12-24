print("""
      ## GENERAL LR MODELS
      ## Multiple Linear Regression
""")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Simulated data
np.random.seed(0)
num_samples = 1000
num_features = 3
X = np.random.rand(num_samples, num_features)  # Features: bedrooms, square footage, neighborhood income
true_coefficients = np.array([30000, 150, 1000])  # True coefficients for features
noise = np.random.normal(0, 2000, num_samples)  # Adding some noise
y = X.dot(true_coefficients) + noise  # Simulated house prices
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predicting house prices
y_pred = model.predict(X_test)
# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

print("""
## Polynomial Regression
""")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Simulated data
np.random.seed(0)
num_samples = 50
temperature = np.random.uniform(10, 40, num_samples)  # Simulated outdoor temperature
sales = 200 + 5 * temperature - 0.2 * temperature**2 + np.random.normal(0, 50, num_samples)  # Simulated sales
# Reshape the data
temperature = temperature.reshape(-1, 1)
sales = sales.reshape(-1, 1)
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(temperature, sales, test_size=0.2, random_state=42)
# Creating polynomial features (quadratic)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Creating and training the Polynomial Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)
# Predicting ice cream sales
y_pred = model.predict(X_test_poly)
# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Plotting the results
plt.scatter(temperature, sales, color='blue', label='Actual Sales')
plt.plot(temperature, model.predict(poly.transform(temperature)), color='red', label='Predicted Sales')
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.legend()
plt.show()
print("""
## Support Vector Regression
""")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# Simulated data
np.random.seed(0)
num_samples = 100
earnings = np.random.uniform(1, 10, num_samples)  # Simulated earnings indicator
dividends = np.random.uniform(0.1, 1, num_samples)  # Simulated dividends indicator
volatility = np.random.uniform(0.05, 0.2, num_samples)  # Simulated volatility indicator
stock_price = 50 + 10 * earnings - 5 * dividends + 20 * volatility + np.random.normal(0, 15, num_samples)  # Simulated stock price
# Combine indicators into a feature matrix
X = np.column_stack((earnings, dividends, volatility))
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, stock_price, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Creating and training the Support Vector Regression model
model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X_train_scaled, y_train)
# Predicting stock prices
y_pred = model.predict(X_test_scaled)
# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Plotting the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Stock Price')
plt.ylabel('Predicted Stock Price')
plt.show()

print("""
## Decision Tree Regression
""")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
# Simulated data
np.random.seed(0)
num_samples = 100
occupants = np.random.randint(1, 10, num_samples)  # Simulated number of occupants
temperature = np.random.uniform(10, 30, num_samples)  # Simulated outdoor temperature
time_of_day = np.random.choice(['morning', 'afternoon', 'evening'], num_samples)  # Simulated time of day
energy_consumption = 100 + 5 * occupants + 3 * temperature + np.random.normal(0, 20, num_samples)  # Simulated energy consumption
# Encode categorical feature
time_of_day_encoded = np.array([0 if t == 'morning' else 1 if t == 'afternoon' else 2 for t in time_of_day]).reshape(-1, 1)
# Combine features into a feature matrix
X = np.column_stack((occupants, temperature, time_of_day_encoded))
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, energy_consumption, test_size=0.2, random_state=42)
# Creating and training the Decision Tree Regression model
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
# Predicting energy consumption
y_pred = model.predict(X_test)
# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Visualizing the Decision Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['Occupants', 'Temperature', 'TimeOfDay'], filled=True)
plt.show()



print("""
## Random Forest Regression
""")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
# Simulated data
np.random.seed(0)
num_samples = 1000
contract_type = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], num_samples)  # Simulated contract type
monthly_charges = np.random.uniform(20, 100, num_samples)  # Simulated monthly charges
customer_satisfaction = np.random.randint(1, 6, num_samples)  # Simulated customer satisfaction scores
churn_probability = 0.3 * (customer_satisfaction - 3) + 0.2 * (contract_type == 'Month-to-Month') + np.random.normal(0, 0.2, num_samples)  # Simulated churn probability
# Encode categorical feature
contract_type_encoded = np.array([0 if c == 'Month-to-Month' else 1 if c == 'One Year' else 2 for c in contract_type]).reshape(-1, 1)
# Combine features into a feature matrix
X = np.column_stack((contract_type_encoded, monthly_charges, customer_satisfaction))
# Generate churn labels based on churn probability
y = (churn_probability > 0.5).astype(int)
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and training the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
# Predicting churn probabilities
y_pred = model.predict(X_test)
# Converting probabilities to binary churn predictions
y_pred_binary = (y_pred > 0.5).astype(int)
# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")