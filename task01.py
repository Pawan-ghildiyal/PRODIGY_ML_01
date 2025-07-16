# Task 01: Linear Regression for House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('train.csv')  
# Explore data
print(data.head())
print(data.info())
print(data.describe())

# Select relevant features from the Kaggle dataset
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_test_values = y_test.values      # Actual prices
y_pred_values = y_pred             # Predicted prices

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print(data.columns)

# Plot results
plt.figure(figsize=(10,6))
plt.scatter(range(len(y_test_values)), y_test_values, color='blue', label='Actual Price', alpha=0.6)
plt.scatter(range(len(y_pred_values)), y_pred_values, color='red', label='Predicted Price', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.title('Actual vs Predicted House Prices (Scatter Plot)')
plt.legend()
plt.tight_layout()
plt.show()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()
