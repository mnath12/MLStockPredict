import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Get Time-Series Data for QQQ
# -------------------------------
ticker = 'QQQ'
data = yf.download(ticker, start="2018-01-01", end="2023-01-01")
# Using the 'Close' column since adjusted prices are in 'Close' by default.
price = data['Close']

# -------------------------------
# 2. Calculate Daily Returns and Volatility
# -------------------------------
# Calculate daily returns as percentage change
returns = price.pct_change().dropna()

# Calculate a 30-day rolling volatility.
# Multiplying by sqrt(252) annualizes the volatility (assuming 252 trading days per year).
rolling_window = 30
volatility = returns.rolling(window=rolling_window).std() * np.sqrt(252)

# -------------------------------
# 3. Plot Stock Price, Returns, and Volatility
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(price.index, price, label='Price')
plt.title('QQQ Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(returns.index, returns, label='Daily Returns', color='orange')
plt.title('QQQ Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(volatility.index, volatility, label='30-Day Rolling Volatility (Annualized)', color='green')
plt.title('QQQ Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Download QQQ data
ticker = 'QQQ'
data = yf.download(ticker, start="2022-01-01", end="2023-01-01")
price = data['Close']

# Calculate daily returns
returns = price.pct_change().dropna()

# Use the previous month's data (approx. 21 trading days)
recent_returns = returns[-21:]

# Fit an ARIMA model (order can be tuned based on your data)
model = ARIMA(recent_returns, order=(1, 0, 1))
model_fit = model.fit()

# Forecast today's return (next day forecast)
forecast = model_fit.forecast(steps=1)
print("Forecasted return for today:", forecast.iloc[-1])
