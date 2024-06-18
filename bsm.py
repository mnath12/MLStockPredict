import numpy as np
from scipy.stats import norm
import yfinance as yf

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Fetching stock data for a specific ticker (e.g., AAPL)
ticker = 'AAPL'
stock = yf.Ticker(ticker)

# Current stock price
S = stock.history(period='1d')['Close'].iloc[-1]

# Option chain data
option_chain = stock.option_chain()

# Example: Fetch the first call and put option data
call_option = option_chain.calls.iloc[0]
put_option = option_chain.puts.iloc[0]

# Extract necessary parameters
K_call = call_option['strike']
K_put = put_option['strike']
T = (call_option['expiration'] - call_option['lastTradeDate']).days / 365.0
r = 0.01  # Risk-free rate (example)
sigma = 0.2  # Implied volatility (example)

# Calculate the call and put prices using the Black-Scholes formula
calculated_call_price = black_scholes_call(S, K_call, T, r, sigma)
calculated_put_price = black_scholes_put(S, K_put, T, r, sigma)

# Real market prices
market_call_price = call_option['lastPrice']
market_put_price = put_option['lastPrice']

# Print results
print(f"Calculated Call Price: {calculated_call_price:.2f}")
print(f"Market Call Price: {market_call_price:.2f}")
print(f"Calculated Put Price: {calculated_put_price:.2f}")
print(f"Market Put Price: {market_put_price:.2f}")
