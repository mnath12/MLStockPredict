#!/usr/bin/env python
"""
fit_days_vol_surf.py - Plot today's volatility surface for TSLA
Uses DataHandler for option data and QuantLib for IV calculations
"""

import sys
import os
from pathlib import Path
from datetime import date, timedelta, datetime
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from backtesting_module.quantlib import compute_iv_quantlib
from backtesting_module.data_handler import DataHandler

# Configuration
TICKER = "TSLA"
RISK_FREE_RATE = 0.05  # 5% annual
DIVIDEND_YIELD = 0.0
EXERCISE_STYLE = "american"
TREE_TYPE = "crr"
STEPS = 1000
MONEYNESS_MIN = 0.75
MONEYNESS_MAX = 1.25
DAYS_LIMIT = 60

# Import configuration
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY

# Use keys from config
POLYGON_KEY = POLYGON_API_KEY
ALPACA_SECRET = ALPACA_SECRET_KEY


def decode_option_symbol(symbol: str) -> Optional[Dict]:
    """Decode Alpaca option symbol format: ROOTYYMMDDCSTRIKE"""
    try:
        root = symbol[:-15]
        yymmdd = symbol[-15:-9]
        cp = symbol[-9]
        strike_str = symbol[-8:]
        
        year = 2000 + int(yymmdd[:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        expiry = date(year, month, day)
        
        strike = float(strike_str) / 1000.0
        option_type = "call" if cp.upper() == "C" else "put"
        
        return {"root": root, "expiry": expiry, "type": option_type, "strike": strike}
    except Exception as e:
        print(f"Error decoding {symbol}: {e}")
        return None


def fetch_option_data(ticker: str, today: date) -> pd.DataFrame:
    """
    Fetch option data for today's volatility surface
    """
    print(f"\n📡 Fetching option data for {ticker}...")
    
    # Initialize data handler
    data_handler = DataHandler(
        alpaca_api_key=ALPACA_API_KEY,
        alpaca_secret=ALPACA_SECRET,
        polygon_key=POLYGON_KEY
    )
    
    # Get current stock price
    print(f"\n📈 Fetching current {ticker} price...")
    try:
        stock_bars = data_handler.get_stock_bars(
            ticker=ticker,
            start_date=today.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
            timeframe="1D"
        )
        
        if stock_bars.empty:
            print("⚠️  No stock data for today, trying last available day...")
            # Try getting last 5 days
            start = (today - timedelta(days=5)).strftime("%Y-%m-%d")
            stock_bars = data_handler.get_stock_bars(
                ticker=ticker,
                start_date=start,
                end_date=today.strftime("%Y-%m-%d"),
                timeframe="1D"
            )
        
        if not stock_bars.empty:
            spot_price = stock_bars['close'].iloc[-1]
            print(f"✅ Current {ticker} price: ${spot_price:.2f}")
        else:
            print("❌ Could not fetch stock price")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching stock price: {e}")
        return pd.DataFrame()
    
    # Get option contracts
    print(f"\n🔍 Searching for option contracts...")
    try:
        # Get expiration range
        exp_to = (today + timedelta(days=DAYS_LIMIT)).strftime("%Y-%m-%d")
        
        # Get strike range based on moneyness
        strike_min = spot_price * MONEYNESS_MIN
        strike_max = spot_price * MONEYNESS_MAX
        
        option_tickers = data_handler.options_search(
            underlying=ticker,
            exp_to=exp_to,
            strike_min=strike_min,
            strike_max=strike_max,
            as_of=today.strftime("%Y-%m-%d"),
            limit=1000
        )
        
        print(f"✅ Found {len(option_tickers)} option contracts")
        
    except Exception as e:
        print(f"❌ Error fetching option contracts: {e}")
        return pd.DataFrame()
    
    # Process options and get pricing data
    print(f"\n📊 Processing option prices...")
    option_data = []
    
    # Increase limit to get more variety in expiries
    for i, opt_ticker in enumerate(option_tickers[:500]):  # Limit to 200 for more expiries
        if i % 20 == 0:
            print(f"   Processing {i+1}/{min(200, len(option_tickers))} options...")
        
        try:
            # Decode option symbol
            meta = decode_option_symbol(opt_ticker)
            if meta is None:
                continue
            
            # Get days to expiry
            days_to_exp = (meta["expiry"] - today).days
            if days_to_exp < 0 or days_to_exp > DAYS_LIMIT:
                continue
            
            # Get option price data
            opt_prices = data_handler.get_option_price_series(
                option_ticker=opt_ticker,
                start_date=today.strftime("%Y-%m-%d"),
                end_date=today.strftime("%Y-%m-%d"),
                timespan="day",
                price_type="close"
            )
            
            if opt_prices.empty:
                continue
            
            price = opt_prices.iloc[-1]
            
            # Calculate implied volatility
            try:
                iv = compute_iv_quantlib(
                    spot_price=spot_price,
                    option_price=price,
                    strike_price=meta["strike"],
                    days_to_maturity=days_to_exp,
                    risk_free_rate=RISK_FREE_RATE,
                    dividend_yield=DIVIDEND_YIELD,
                    option_type=meta["type"],
                    exercise_style=EXERCISE_STYLE,
                    tree=TREE_TYPE,
                    steps=STEPS
                )
                
                option_data.append({
                    "symbol": opt_ticker,
                    "strike": meta["strike"],
                    "expiry": meta["expiry"],
                    "type": meta["type"],
                    "days_to_expiry": days_to_exp,
                    "price": price,
                    "iv": iv,
                    "moneyness": meta["strike"] / spot_price
                })
                
            except Exception as e:
                continue
                
        except Exception as e:
            continue
    
    print(f"✅ Successfully processed {len(option_data)} options")
    
    if len(option_data) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(option_data)


def get_today_volatility_surface():
    """
    Main function to generate today's volatility surface
    """
    print("=" * 60)
    print("  Real-Time Volatility Surface Generator")
    print("=" * 60)
    print(f"\n📊 Configuration:")
    print(f"   Ticker: {TICKER}")
    print(f"   Risk-free rate: {RISK_FREE_RATE*100:.1f}%")
    print(f"   Exercise style: {EXERCISE_STYLE.title()}")
    print(f"   Moneyness: {MONEYNESS_MIN:.2f} - {MONEYNESS_MAX:.2f}")
    print(f"   Days ahead: {DAYS_LIMIT}")
    
    today = date.today()
    print(f"\n📅 Today: {today.strftime('%Y-%m-%d')}")
    
    # Fetch option data
    option_df = fetch_option_data(TICKER, today)
    
    if option_df.empty:
        print("\n❌ No option data retrieved")
        print("\nPossible issues:")
        print("  1. No options available in the moneyness range")
        print("  2. Date range issue")
        print("  3. API connectivity issues")
        return
    
    # Display summary
    print(f"\n📊 Data Summary:")
    print(f"   Total options: {len(option_df)}")
    print(f"   Calls: {len(option_df[option_df['type'] == 'call'])}")
    print(f"   Puts: {len(option_df[option_df['type'] == 'put'])}")
    print(f"   Strike range: ${option_df['strike'].min():.2f} - ${option_df['strike'].max():.2f}")
    print(f"   IV range: {option_df['iv'].min():.1%} - {option_df['iv'].max():.1%}")
    
    # Plot surface
    plot_surface(option_df, TICKER)


def plot_surface(df: pd.DataFrame, ticker: str):
    """Plot 3D volatility surface or 2D curve if only one expiry"""
    if df.empty or len(df) < 10:
        print("\n⚠️  Insufficient data for surface plot")
        print(f"   Need at least 10 options, got {len(df)}")
        return
    
    # Convert to years
    df['ttm_years'] = df['days_to_expiry'] / 365.25
    
    # Check if we have multiple expiries
    unique_ttm = df['ttm_years'].nunique()
    unique_strikes = df['strike'].nunique()
    
    print(f"\n📊 Data dimensions:")
    print(f"   Unique strikes: {unique_strikes}")
    print(f"   Unique TTM values: {unique_ttm}")
    
    if unique_ttm == 1:
        print(f"\n⚠️  Only one expiry date - plotting 2D curve instead")
        # Plot 2D vol curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sort by strike
        df_sorted = df.sort_values('strike')
        
        # Plot 1: IV vs Strike
        ax1.plot(df_sorted['strike'], df_sorted['iv'], 'o-', color='steelblue', markersize=6, linewidth=2)
        ax1.set_xlabel('Strike Price ($)', fontsize=12)
        ax1.set_ylabel('Implied Volatility', fontsize=12)
        days_to_exp = df_sorted['days_to_expiry'].iloc[0]
        ax1.set_title(f'{ticker} Implied Volatility Curve\n(Expiry: {days_to_exp} days)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Moneyness vs IV
        ax2.plot(df_sorted['moneyness'], df_sorted['iv'], 'o-', color='coral', markersize=6, linewidth=2)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ATM')
        ax2.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax2.set_ylabel('Implied Volatility', fontsize=12)
        ax2.set_title('IV Smile/Skew', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
    else:
        # Plot 3D surface
        print(f"\n🎨 Creating 3D volatility surface plot...")
        
        # Create pivot table
        surface = df.pivot_table(
            values='iv',
            index='ttm_years',
            columns='strike',
            aggfunc='mean'
        )
        
        if surface.empty:
            print("❌ Cannot create surface")
            print(f"   Unique strikes: {df['strike'].nunique()}")
            print(f"   Unique TTM: {df['ttm_years'].nunique()}")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(surface.columns.values, surface.index.values)
        Z = surface.values
        
        surf = ax.plot_surface(
            X, Y, Z, 
            cmap='viridis', 
            alpha=0.9, 
            linewidth=0.1, 
            antialiased=True,
            edgecolors='black',
            linewidths=0.5
        )
        
        ax.set_xlabel('Strike Price ($)', fontsize=12, labelpad=10)
        ax.set_ylabel('Time to Expiry (years)', fontsize=12, labelpad=10)
        ax.set_zlabel('Implied Volatility', fontsize=12, labelpad=10)
        
        today_str = date.today().strftime("%Y-%m-%d")
        ax.set_title(
            f'{ticker} Implied Volatility Surface\n'
            f'{EXERCISE_STYLE.title()} Options - {today_str}',
            fontsize=14,
            pad=20
        )
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Implied Volatility')
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent
    today_str = date.today().strftime("%Y-%m-%d")
    filename = output_dir / f"{ticker}_vol_surface_{today_str.replace('-', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot to {filename}")
    
    # Also save data
    csv_filename = output_dir / f"{ticker}_vol_data_{today_str.replace('-', '')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"💾 Saved data to {csv_filename}")
    
    plt.show()
    
    print(f"\n✅ Complete! Generated plot with {len(df)} options")


if __name__ == "__main__":
    get_today_volatility_surface()
