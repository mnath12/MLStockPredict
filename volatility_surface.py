#!/usr/bin/env python
"""
Minimal Volatility Surface Testing Script

This script creates a volatility surface for a hardcoded option using QuantLib and Alpaca data.
Less than 100 lines of code with everything hardcoded for easy testing.
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import QuantLib as ql
import requests

# Import DataHandler from the backtesting module
from backtesting_module.data_handler import DataHandler

# Import configuration
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY, FRED_API_KEY

# =============================================================================
# HARDCODED PARAMETERS - Modify these as needed
# =============================================================================

# Stock and Option Parameters
STOCK_TICKER = "AAPL"
OPTION_TICKER = "AAPL251031C00140000"  # AAPL Call expiring 2025-04-18, strike $200
DATE_RANGE_START = "2025-10-01"
DATE_RANGE_END = "2025-10-17"

# Risk Parameters
RISK_FREE_RATE = None  # Will be fetched from FRED
DIVIDEND_YIELD = 0.0

# FRED API Configuration
FRED_SERIES_ID = "TB3MS"  # 3-Month Treasury Bill: Secondary Market Rate

# Use keys from config
POLYGON_KEY = POLYGON_API_KEY

# =============================================================================
# FRED DATA FETCHING
# =============================================================================

def get_risk_free_rate_from_fred():
    """Fetch the current 3-month Treasury bill rate from FRED"""
    try:
        # FRED API endpoint for latest observation
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': FRED_SERIES_ID,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': 1
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        observations = data.get('observations', [])
        
        if observations:
            # Get the latest rate and convert from percentage to decimal
            latest_rate = float(observations[0]['value'])
            risk_free_rate = latest_rate / 100.0  # Convert from percentage to decimal
            
            print(f"📊 Current 3-Month T-Bill Rate: {latest_rate:.2f}% ({risk_free_rate:.4f} decimal)")
            return risk_free_rate
        else:
            print("⚠️ No data available from FRED, using fallback rate of 5%")
            return 0.05
            
    except Exception as e:
        print(f"⚠️ Error fetching risk-free rate from FRED: {e}")
        print("Using fallback rate of 5%")
        return 0.05

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"🔬 Building Volatility Surface for {STOCK_TICKER}")
    print(f"📅 Date Range: {DATE_RANGE_START} to {DATE_RANGE_END}")
    print(f"🎯 Option: {OPTION_TICKER}")
    
    # Get current risk-free rate from FRED
    print(f"\n🏦 Fetching current risk-free rate from FRED...")
    risk_free_rate = get_risk_free_rate_from_fred()
    
    # Initialize DataHandler
    data_handler = DataHandler(
        alpaca_api_key=ALPACA_API_KEY,
        alpaca_secret=ALPACA_SECRET_KEY,
        polygon_key=POLYGON_KEY
    )
    
    # Get stock price data
    print(f"\n📈 Fetching stock data for {STOCK_TICKER}...")
    stock_bars = data_handler.get_stock_bars(
        STOCK_TICKER, 
        DATE_RANGE_START, 
        DATE_RANGE_END, 
        "1Day"
    )
    
    if stock_bars.empty:
        print("❌ No stock data found")
        return
    
    print(f"✅ Retrieved {len(stock_bars)} stock bars")
    
    # Get option price data
    print(f"\n📊 Fetching option data for {OPTION_TICKER}...")
    try:
        option_bars = data_handler.get_option_aggregates(
            OPTION_TICKER,
            DATE_RANGE_START,
            DATE_RANGE_END,
            "day"
        )
        
        if option_bars.empty:
            print("❌ No option data found")
            return
            
        print(f"✅ Retrieved {len(option_bars)} option bars")
        
    except Exception as e:
        print(f"❌ Error fetching option data: {e}")
        return
    
    # Build volatility surface data
    print(f"\n🔬 Building volatility surface...")
    
    # Simple implied volatility calculation using QuantLib
    today = ql.Date.todaysDate()
    day_counter = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    
    # Set up QuantLib objects
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, day_counter))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, DIVIDEND_YIELD, day_counter))
    vol_q = ql.SimpleQuote(0.20)  # Initial volatility guess
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_q), day_counter)
    )
    
    # Extract option details from ticker
    strike_price = 200.0  # From OPTION_TICKER
    expiration_date = dt.date(2025, 4, 18)  # From OPTION_TICKER
    option_type = ql.Option.Call  # From OPTION_TICKER
    
    # Create QuantLib option
    expiry_date = ql.Date(expiration_date.day, expiration_date.month, expiration_date.year)
    exercise = ql.EuropeanExercise(expiry_date)
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    option = ql.VanillaOption(payoff, exercise)
    
    # Build surface data points
    surface_data = []
    
    for i, (stock_date, stock_row) in enumerate(stock_bars.iterrows()):
        if stock_date.date() in option_bars.index.date:
            # Get option price for this date
            option_date = option_bars.index[option_bars.index.date == stock_date.date()][0]
            option_price = option_bars.loc[option_date, 'close']
            
            if pd.notna(option_price) and option_price > 0:
                # Calculate time to expiry
                stock_date_dt = pd.to_datetime(stock_date).date()
                ttm = (expiration_date - stock_date_dt).days / 365.25
                
                if ttm > 0:
                    # Set up process with current stock price
                    spot_price = stock_row['close']
                    spot_q = ql.QuoteHandle(ql.SimpleQuote(spot_price))
                    process = ql.BlackScholesMertonProcess(spot_q, div_ts, rf_ts, vol_ts)
                    
                    # Set pricing engine
                    engine = ql.AnalyticEuropeanEngine(process)
                    option.setPricingEngine(engine)
                    
                    # Calculate implied volatility using simple bisection
                    try:
                        iv = calculate_implied_vol(option, process, option_price)
                        if not np.isnan(iv):
                            surface_data.append({
                                'date': stock_date,
                                'spot_price': spot_price,
                                'option_price': option_price,
                                'ttm': ttm,
                                'strike': strike_price,
                                'iv': iv
                            })
                    except:
                        continue
    
    if not surface_data:
        print("❌ No valid volatility data points found")
        return
    
    # Create DataFrame and display results
    surface_df = pd.DataFrame(surface_data)
    print(f"\n✅ Built volatility surface with {len(surface_df)} data points")
    print("\n📊 Surface Data Preview:")
    print(surface_df[['date', 'spot_price', 'option_price', 'ttm', 'iv']].head())
    
    # Plot the surface
    plot_volatility_surface(surface_df, risk_free_rate)
    
    # Save results
    filename = f"{STOCK_TICKER}_vol_surface_{DATE_RANGE_START}_{DATE_RANGE_END}.csv"
    surface_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to {filename}")

def calculate_implied_vol(option, process, market_price, tolerance=1e-4):
    """Calculate implied volatility using simple bisection method"""
    vol_quote = process.blackVolatility().link.volatilityQuote()
    
    def price_diff(vol):
        vol_quote.setValue(vol)
        return option.NPV() - market_price
    
    # Bisection search
    low, high = 0.01, 3.0
    for _ in range(50):
        mid = (low + high) / 2
        diff = price_diff(mid)
        if abs(diff) < tolerance:
            return mid
        elif diff > 0:
            high = mid
        else:
            low = mid
    
    return np.nan

def plot_volatility_surface(surface_df, risk_free_rate):
    """Plot the volatility surface"""
    try:
        fig = plt.figure(figsize=(10, 6))
        
        # 2D plot of IV over time
        plt.subplot(1, 2, 1)
        plt.plot(surface_df['date'], surface_df['iv'], 'b-o', markersize=4)
        plt.title(f'Implied Volatility Over Time\n{STOCK_TICKER} {OPTION_TICKER}\nRisk-free Rate: {risk_free_rate:.2%}')
        plt.xlabel('Date')
        plt.ylabel('Implied Volatility')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3D surface plot if we have enough data points
        if len(surface_df) > 3:
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(surface_df['ttm'], surface_df['strike'], surface_df['iv'], 
                      c=surface_df['iv'], cmap='viridis', s=50)
            ax.set_xlabel('Time to Expiry (years)')
            ax.set_ylabel('Strike Price')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('Volatility Surface')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

if __name__ == "__main__":
    main()
