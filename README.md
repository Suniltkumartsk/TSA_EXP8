# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 28.10.2025

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv('usedcarssold.csv')

# Generate random Date for each Year
def random_date_in_year(year):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    random_days = np.random.randint(0, (end - start).days + 1)
    return start + timedelta(days=random_days)

data['Date'] = data['Year'].apply(random_date_in_year)

# Convert Date column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use Sold_Cars as target variable
close_data = data[['Sold_Cars']]
print("Shape of the dataset:", close_data.shape)
print("First 10 rows of the dataset:")
print(close_data.head(10))

# Plot original Sold Cars data
plt.plot(close_data['Sold_Cars'], label='Original Used Cars Sold Data')
plt.title('Original Used Cars Sold Data')
plt.xlabel('Date')
plt.ylabel('Number of Cars Sold')
plt.legend()
plt.grid()
plt.show()

# Moving Averages
rolling_mean_5 = close_data['Sold_Cars'].rolling(window=5).mean()
rolling_mean_10 = close_data['Sold_Cars'].rolling(window=10).mean()

rolling_mean_5.head(10)
rolling_mean_10.head(20)

# Plot moving averages
plt.plot(close_data['Sold_Cars'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Used Cars Sold Data')
plt.xlabel('Date')
plt.ylabel('Number of Cars Sold')
plt.legend()
plt.grid()
plt.show()

# Resample to monthly frequency
data_monthly = data.resample('MS').sum()
data_monthly_close = data_monthly['Sold_Cars']

# Scale data slightly to avoid zero values
scaled_data = pd.Series(data_monthly.values.reshape(-1, 1).flatten())
scaled_data = scaled_data + 1

# Split train/test
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Train model (additive trend and additive seasonal)
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast
test_predictions_add = model_add.forecast(steps=len(test_data))

# Visual evaluation
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual Evaluation - Used Cars Sold')
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Square Error (RMSE):", rmse)

# Variance and mean
print("Variance:", np.sqrt(scaled_data.var()), "Mean:", scaled_data.mean())

# Final model for prediction (additive seasonal)
model = ExponentialSmoothing(data_monthly_close, trend='add', seasonal='add', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly_close) / 4))

# Plot final predictions
ax = data_monthly_close.plot()
predictions.plot(ax=ax)
ax.legend(["data_monthly_close", "predictions"])
ax.set_xlabel('Date')
ax.set_ylabel('Number of Cars Sold')
ax.set_title('Used Car Sales Forecast')
plt.show()

```

### OUTPUT:
Original Dataset:
<img width="461" height="261" alt="image" src="https://github.com/user-attachments/assets/1d4a9ac8-1457-4a0e-9f69-0a4bef374a9d" />

Plot Transform Dataset:
<img width="654" height="461" alt="image" src="https://github.com/user-attachments/assets/5bcdcc94-de23-4ed3-a546-4c6af7e16fd3" />


Performance metrics:
<img width="988" height="253" alt="image" src="https://github.com/user-attachments/assets/a02f841f-2226-4a47-bdd1-1df5c4224473" />


Exponential Smoothing:
<img width="752" height="449" alt="image" src="https://github.com/user-attachments/assets/2bd44972-0dc2-4862-b0c7-b993bed73801" />


Prediction:

<img width="661" height="464" alt="image" src="https://github.com/user-attachments/assets/7c8a5d5e-7d36-4af2-b3dd-6b4e5ff49d80" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
