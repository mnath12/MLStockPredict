import yfinance as yf
import pandas as pd
import numpy as np
import pmdarima as pm
from arch import arch_model

# -------------------------------
# 1. Download QQQ Data and Compute Returns
# -------------------------------
ticker = 'QQQ'
# Download data from December 2017 to have a training period for January 2018 onward.
data = yf.download(ticker, start="2017-12-01", end="2023-01-01")
price = data['Close']
returns = price.pct_change().dropna()

# -------------------------------
# 2. Fit an ARIMA Model on Returns
# -------------------------------
# Automatically select the best ARIMA model using pmdarima's auto_arima.
arima_model = pm.auto_arima(returns, seasonal=False, stepwise=True, trace=True)
print("Selected ARIMA model order:", arima_model.order)

# Extract residuals from the ARIMA model.
arima_residuals = arima_model.arima_res_.resid

# -------------------------------
# 3. Fit a GARCH(1,1) Model on ARIMA Residuals
# -------------------------------
# We set the mean to 'Zero' because ARIMA residuals should be centered around zero.
garch_model = arch_model(arima_residuals, mean='Zero', vol='GARCH', p=1, q=1, dist='normal')
garch_fitted = garch_model.fit(disp='off')
print(garch_fitted.summary())

# -------------------------------
# 4. Forecast Next-Day Return and Volatility
# -------------------------------
# Forecast the ARIMA model for the next period's mean return.
predicted_mu = arima_model.predict(n_periods=1).item()

# Forecast the volatility using the GARCH model.
garch_forecast = garch_fitted.forecast(horizon=1)
# The forecasted variance for the next period:
predicted_variance = garch_forecast.variance.iloc[-1, 0]
predicted_sigma = np.sqrt(predicted_variance)

# In this ARIMA-GARCH setup, the one-step ahead return forecast is:
# y(t+1) = predicted_mu + predicted_error, with E[predicted_error]=0,
# so the best forecast is just predicted_mu.
prediction = predicted_mu

print("Forecasted Mean Return (ARIMA):", predicted_mu)
print("Forecasted Volatility (GARCH):", predicted_sigma)
print("Combined Prediction (Return forecast):", prediction)
