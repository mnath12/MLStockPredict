import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------
# 1. Download QQQ Data
# -------------------------------
# We download data starting December 2017 so that the first forecast for January 2018 can use December 2017 data.
ticker = 'QQQ'
data = yf.download(ticker, start="2017-9-01", end="2023-01-01")
price = data['Close']

# Calculate daily returns (percentage change)
returns = price.pct_change().dropna()

# -------------------------------
# 2. Rolling ARIMA Forecast on Returns
# -------------------------------
# We'll use a window of ~21 trading days (approx. one month)
window = 63

forecasts = []
actuals = []
dates = []

# Loop over the returns series: each iteration uses the previous 21 days to forecast the next day's return.
for i in range(window, len(returns)):
    train = returns.iloc[i-window:i]
    try:
        # Fit an ARIMA(1,0,1) model on the training window
        model = ARIMA(train, order=(5, 1, 3))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        forecasts.append(forecast.iloc[-1])
    except Exception as e:
        forecasts.append(np.nan)
    actuals.append(returns.iloc[i])
    dates.append(returns.index[i])

# Compile the forecasts and actual returns into a DataFrame
df_forecast = pd.DataFrame({'Forecast': forecasts, 'Actual': actuals}, index=dates)

# -------------------------------
# 3. Evaluate Forecast Accuracy
# -------------------------------
# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = np.mean(np.abs(df_forecast['Forecast'] - df_forecast['Actual']))
mse = np.mean((df_forecast['Forecast'] - df_forecast['Actual'])**2)
print("Mean Absolute Error: {:.6f}".format(mae))
print("Mean Squared Error: {:.6f}".format(mse))
# -------------------------------
# -------------------------------
# 4. Directional Accuracy Calculation
# -------------------------------
# Compute the sign of each element using apply with a lambda function.
df_forecast['Forecast_Sign'] = df_forecast['Forecast'].apply(lambda x: np.sign(x))
df_forecast['Actual_Sign'] = df_forecast['Actual'].apply(lambda x: np.sign(x))

# Now compute whether the forecast sign matches the actual sign.
df_forecast['Match'] = df_forecast['Forecast_Sign'] == df_forecast['Actual_Sign']

# Count the number of matches and compute the percentage.
match_count = df_forecast['Match'].sum()
total_days = len(df_forecast)
match_percent = (match_count / total_days) * 100

print(f"Directional Accuracy: Forecast matched actual sign on {match_count} out of {total_days} days ({match_percent:.2f}%).")

# -------------------------------
# 4. Plot Forecasts vs. Actual Returns
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_forecast.index, df_forecast['Actual'], label='Actual Returns', color='blue')
plt.plot(df_forecast.index, df_forecast['Forecast'], label='Forecasted Returns', color='red', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('ARIMA Forecast vs. Actual Daily Returns')
plt.legend()
plt.grid(True)
plt.show()
