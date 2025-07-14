import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from arch import arch_model

# -------------------------------
# 1. Download QQQ Data and Compute Returns
# -------------------------------
ticker = 'QQQ'
data = yf.download(ticker, start="2017-12-01", end="2023-01-01")
price = data['Close']
returns = price.pct_change().dropna()

# -------------------------------
# 2. Rolling ARIMA-GARCH Forecasting
# -------------------------------
# Define a training window (e.g., 63 days)
window = 63

# Lists to store forecasts and actuals
forecasted_returns = []
forecasted_volatilities = []
actual_returns = []
dates = []

# Loop over the returns series.
for i in range(window, len(returns)):
    train = returns.iloc[i - window:i]
    try:
        # Fit an ARIMA model on the training window using auto_arima.
        arima_model = pm.auto_arima(train, seasonal=False, stepwise=True, trace=False)
        # Forecast the next day's mean return.
        predicted_mu = arima_model.predict(n_periods=1).item()
        # Extract ARIMA residuals.
        arima_residuals = arima_model.arima_res_.resid
        
        # Fit a GARCH(1,1) model on the ARIMA residuals.
        # Use mean='Zero' since residuals should be centered.
        garch_model = arch_model(arima_residuals, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
        garch_fitted = garch_model.fit(disp='off')
        # Forecast the next day's variance.
        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_variance = garch_forecast.variance.iloc[-1, 0]
        predicted_sigma = np.sqrt(predicted_variance)
    except Exception as e:
        predicted_mu = np.nan
        predicted_sigma = np.nan

    forecasted_returns.append(predicted_mu)
    forecasted_volatilities.append(predicted_sigma)
    actual_returns.append(returns.iloc[i])
    dates.append(returns.index[i])

# Compile forecasts and actual returns into a DataFrame.
df_forecast = pd.DataFrame({
    'Forecast_Return': forecasted_returns,
    'Forecast_Volatility': forecasted_volatilities,
    'Actual_Return': actual_returns
}, index=dates)

# For a simple proxy of realized volatility, use the absolute value of the actual return.
df_forecast['Realized_Volatility'] = np.abs(df_forecast['Actual_Return'])

# -------------------------------
# 3. Plot Forecasted vs Actual Returns
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_forecast.index, df_forecast['Actual_Return'], label='Actual Return', color='blue')
plt.plot(df_forecast.index, df_forecast['Forecast_Return'], label='Forecasted Return', color='red', linestyle='--')
plt.title('Actual vs Forecasted Return (ARIMA-GARCH)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 4. Plot Forecasted vs Realized Volatility
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_forecast.index, df_forecast['Realized_Volatility'], label='Realized Volatility (|Return|)', color='blue')
plt.plot(df_forecast.index, df_forecast['Forecast_Volatility'], label='Forecasted Volatility', color='red', linestyle='--')
plt.title('Realized vs Forecasted Volatility (ARIMA-GARCH)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()
