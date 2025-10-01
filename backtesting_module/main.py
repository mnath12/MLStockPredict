#!/usr/bin/env python3
"""
Main backtesting loop following the architecture diagram.

This implements the main loop that connects all components:
DataHandler -> GreeksEngine -> Portfolio -> Strategy -> PositionSizer -> ExecutionHandler

Key Features:
- Modular, extensible architecture for stock and option backtesting.
- User-driven workflow for selecting stock, date range, and frequency.
- Fetches and displays available option contracts for a given date.
- **LSTM Volatility Forecasting Integration:**
    - Uses pre-trained LSTM model from Colab for volatility predictions
    - Processes hourly realized volatility data matching Colab training format
    - Provides robust error handling with EWMA fallback
    - Tracks model performance metrics during backtesting
- **Parallelized option contract ranking:**
    - When ranking options by available data (number of bars), the code now uses Python's ThreadPoolExecutor to query Polygon's API for all contracts in parallel.
    - This dramatically speeds up the process of finding the top 20 most active/liquid contracts, making the workflow much more responsive for the user.
    - The number of parallel workers is set to 8 by default, balancing speed and API rate limits.
- User selects from the top 20 contracts with the most data, reducing the chance of picking illiquid or inactive options.
- **Dynamic risk-free rate from FRED:**
    - Fetches real-time risk-free rates from Federal Reserve Economic Data (FRED)
    - Uses 3-Month Treasury Bill rate (DGS3MO) as the risk-free rate
    - Falls back to 10-Year Treasury rate (DGS10) if 3-month data unavailable
    - Provides fallback to default rate if FRED API fails

Environment Requirements:
- Run in tf-m1 conda environment for optimal TensorFlow performance
- Ensure TensorFlow >= 2.13.0 is installed
- LSTM model files must be in volatility_models/ folder
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import warnings
import re
import concurrent.futures
import os
import sys

# Environment validation for TensorFlow
def validate_environment():
    """Validate that the environment is properly set up for LSTM volatility forecasting."""
    print("🔍 Validating environment for LSTM volatility forecasting...")
    
    # Check TensorFlow availability
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check if we're in the tf-m1 environment (optional)
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        if 'tf-m1' in conda_env.lower():
            print(f"✅ Running in tf-m1 conda environment: {conda_env}")
        else:
            print(f"⚠️  Current environment: {conda_env}")
            print("   For optimal performance, consider using the tf-m1 conda environment")
        
        # Check GPU availability (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
        else:
            print("ℹ️  No GPU detected, using CPU")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not available")
        print("   Please install TensorFlow: pip install tensorflow>=2.13.0")
        print("   Or activate the tf-m1 conda environment")
        return False
    except Exception as e:
        print(f"❌ Error validating TensorFlow: {e}")
        return False

# Validate environment before importing modules
env_valid = validate_environment()
if not env_valid:
    print("\n⚠️  Environment validation failed. LSTM volatility forecasting may not work properly.")
    print("   The system will fall back to EWMA volatility estimation.")
    print("   To fix this, ensure TensorFlow is installed and properly configured.\n")

from backtesting_module import (
    DataHandler, GreeksEngine, Portfolio, PositionSizer, ExecutionHandler,
    BaseStrategy, LSTMStrategy, Fill
)
from backtesting_module.strategy import BuyAndHoldStrategy
from backtesting_module.data_handler import DataHandler
from backtesting_module.volatility_forecaster import VolatilityForecaster
from backtesting_module.lstm_volatility_forecaster import LSTMVolatilityForecaster, create_lstm_volatility_forecaster

def get_risk_free_rate_from_fred(fred_api_key: Optional[str] = None, fallback_rate: float = 0.02) -> float:
    """
    Fetch the current risk-free rate from FRED.
    
    Args:
        fred_api_key: FRED API key (32 characters)
        fallback_rate: Rate to use if FRED API fails (default: 2%)
    
    Returns:
        float: Current risk-free rate as a decimal (e.g., 0.05 for 5%)
    """
    if not fred_api_key:
        print(f"Warning: No FRED API key provided. Using fallback rate: {fallback_rate*100:.1f}%")
        return fallback_rate
    
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)
        
        # Try 3-Month Treasury Bill rate first (most commonly used for short-term options)
        try:
            dgs3mo = fred.get_series('DGS3MO')
            if not dgs3mo.empty:
                latest_rate = dgs3mo.iloc[-1]
                if not pd.isna(latest_rate):
                    rate = latest_rate / 100.0  # Convert from percentage to decimal
                    print(f"Using 3-Month Treasury Bill rate: {rate*100:.2f}%")
                    return rate
        except Exception as e:
            print(f"Could not fetch 3-Month Treasury rate: {e}")
        
        # Fallback to 10-Year Treasury rate
        try:
            dgs10 = fred.get_series('DGS10')
            if not dgs10.empty:
                latest_rate = dgs10.iloc[-1]
                if not pd.isna(latest_rate):
                    rate = latest_rate / 100.0  # Convert from percentage to decimal
                    print(f"Using 10-Year Treasury rate: {rate*100:.2f}%")
                    return rate
        except Exception as e:
            print(f"Could not fetch 10-Year Treasury rate: {e}")
        
        # If both fail, use fallback rate
        print(f"Could not fetch rates from FRED. Using fallback rate: {fallback_rate*100:.1f}%")
        return fallback_rate
        
    except ImportError:
        print("fredapi package not installed. Install with: pip install fredapi")
        print(f"Using fallback rate: {fallback_rate*100:.1f}%")
        return fallback_rate
    except Exception as e:
        print(f"Error fetching risk-free rate from FRED: {e}")
        print(f"Using fallback rate: {fallback_rate*100:.1f}%")
        return fallback_rate

def parse_freq(freq: str):
    # Remove spaces and lowercase
    freq = freq.replace(' ', '').lower()
    m = re.fullmatch(r"(\d+)([a-z]+)", freq)
    if not m:
        raise ValueError(f"Invalid frequency: {freq}")
    n, unit = int(m.group(1)), m.group(2)
    if unit in ("min", "t"):
        return n, "minute"
    if unit in ("hour", "h"):
        return n, "hour"
    if unit in ("day", "d"):
        return n, "day"
    if unit in ("week", "w"):
        return n, "week"
    if unit in ("month", "m"):
        return n, "month"
    if unit in ("quarter", "q"):
        return n, "quarter"
    if unit in ("year", "y"):
        return n, "year"
    raise ValueError(f"Unsupported frequency unit: {unit}")

def parse_option_ticker(ticker):
    # Remove O: prefix if present
    if ticker.startswith("O:"):
        ticker = ticker[2:]
    # Underlying: all letters up to first digit
    m = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", ticker)
    if not m:
        raise ValueError(f"Could not parse option ticker: {ticker}")
    underlying, date_part, opt_type, strike_part = m.groups()
    expiry = pd.to_datetime(f"20{date_part}", format='%Y%m%d').tz_localize('UTC')
    strike = float(strike_part) / 1000.0
    
    # Convert option type to full name
    if opt_type.upper() == 'C':
        opt_type = 'call'
    elif opt_type.upper() == 'P':
        opt_type = 'put'
    else:
        raise ValueError(f"Invalid option type: {opt_type}")
    
    return underlying, expiry, opt_type, strike

def should_rebalance(current_timestamp: pd.Timestamp, last_rebalance_date: Optional[pd.Timestamp], 
                    rebalancing_freq: str, current_delta: Optional[float] = None, 
                    target_delta: float = 0.0, delta_threshold: float = 0.05) -> bool:
    """
    Determine if we should rebalance based on the rebalancing frequency and delta threshold.
    
    Args:
        current_timestamp: Current timestamp
        last_rebalance_date: Last rebalancing date (None if first time)
        rebalancing_freq: Rebalancing frequency ("daily", "weekly", "monthly", "every_bar")
        current_delta: Current portfolio delta
        target_delta: Target portfolio delta (usually 0 for delta-neutral)
        delta_threshold: Minimum delta deviation to trigger rebalancing
        
    Returns:
        bool: True if we should rebalance
    """
    # Always rebalance on first run
    if last_rebalance_date is None:
        return True
    
    # Check frequency-based rebalancing
    frequency_check = False
    current_date = current_timestamp.date()
    last_date = last_rebalance_date.date()
    
    if rebalancing_freq == "every_bar":
        frequency_check = True
    elif rebalancing_freq == "daily":
        frequency_check = current_date > last_date
    elif rebalancing_freq == "weekly":
        # Check if we've moved to a new week
        current_week = current_date.isocalendar()[1]
        last_week = last_date.isocalendar()[1]
        frequency_check = current_week > last_week
    elif rebalancing_freq == "monthly":
        # Check if we've moved to a new month
        frequency_check = (current_date.year > last_date.year) or \
                         (current_date.year == last_date.year and current_date.month > last_date.month)
    
    # For straddle strategy, we use frequency-based rebalancing AND delta deviation
    if frequency_check:
        # If frequency check passes, also check delta deviation
        if current_delta is not None:
            delta_deviation = abs(current_delta - target_delta)
            return delta_deviation >= delta_threshold
        return True
    
    # For every_bar rebalancing, check delta threshold
    if rebalancing_freq == "every_bar" and current_delta is not None:
        delta_deviation = abs(current_delta - target_delta)
        return delta_deviation >= delta_threshold
    
    return False

class BacktestEngine:
    """
    Main backtesting engine that orchestrates all components.
    """
    def __init__(self, initial_cash: float = 100000):
        self.data_handler: Optional[DataHandler] = None
        self.greeks_engine = GreeksEngine(pricing_model="Black-Scholes")
        self.portfolio = Portfolio(initial_cash)
        self.position_sizer = PositionSizer()
        self.execution_handler = ExecutionHandler()
        self.strategy: Optional[BaseStrategy] = None
        self.big_df: Optional[pd.DataFrame] = None
        self.results = {}

    def setup_data_handler(self, alpaca_key: str = "", alpaca_secret: str = "", polygon_key: str = "") -> None:
        # Always use the real DataHandler
        print("Setting up DataHandler...")
        self.data_handler = DataHandler(alpaca_key, alpaca_secret, polygon_key)

    def prompt_user(self):
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        return symbol, start_date, end_date

    def run_backtest(self) -> Dict[str, Any]:
        if not all([self.data_handler, self.strategy, self.big_df is not None]):
            raise ValueError("Must setup data handler, strategy, and load data before running backtest")

        print("\n" + "="*50)
        print("STARTING MAIN BACKTESTING LOOP")
        print("="*50)

        if self.big_df is None:
            raise ValueError("No data loaded for backtesting")
        
        total_bars = len(self.big_df)
        print(f"Processing {total_bars} bars...")

        for i, (timestamp, row) in enumerate(self.big_df.iterrows()):
            market_data = self._extract_market_data(row)
            greeks_data = self._calculate_greeks(market_data, timestamp)
            market_data.update(greeks_data)
            current_prices = {k: v for k, v in market_data.items() if not k.endswith(('_delta', '_gamma', '_vega', '_theta', '_rho'))}
            self.portfolio.update_portfolio_value(current_prices, pd.Timestamp(timestamp))
            portfolio_view = self.portfolio.portfolio_view()
            if self.strategy is None:
                raise ValueError("No strategy loaded for backtesting")
            targets = self.strategy.on_bar(self.big_df, i, portfolio_view, market_data)
            if targets:
                orders = self.position_sizer.get_orders(targets, self.portfolio, pd.Timestamp(timestamp))
                if orders:
                    fills = self.execution_handler.get_fills(orders, current_prices)
                    for fill in fills:
                        self.portfolio.update_with_fill(fill)
        print("✓ Backtesting completed")
        return {}

    def _extract_market_data(self, row: pd.Series) -> Dict[str, float]:
        market_data = {}
        for col in row.index:
            value = row[col]
            if isinstance(value, (float, int)) and not pd.isna(value):
                if isinstance(col, str) and col.endswith('_close'):
                    symbol = col.replace('_close', '')
                    market_data[symbol] = float(value)
        return market_data

    def _calculate_greeks(self, market_data: Dict[str, float], timestamp: Any) -> Dict[str, float]:
        # Dummy implementation for now
        return {}

def find_best_straddle_combinations(calls, puts, current_stock_price, min_dte_days=10):
    """
    Find the best straddle combinations based on:
    1. Same strike price (symmetric straddle)
    2. Same expiry date
    3. Close to ATM (current stock price)
    4. Good liquidity (number of bars)
    5. Minimum days to expiry (to avoid high gamma)
    
    Returns:
        List of tuples: (call_ticker, put_ticker, strike, expiry, atm_distance, liquidity_score, dte_days)
    """
    from datetime import datetime
    best_combinations = []
    
    for call_ticker, call_bars in calls:
        try:
            call_underlying, call_expiry, call_type, call_strike = parse_option_ticker(call_ticker)
        except:
            continue
            
        for put_ticker, put_bars in puts:
            try:
                put_underlying, put_expiry, put_type, put_strike = parse_option_ticker(put_ticker)
            except:
                continue
            
            # Check if same strike and expiry
            if call_strike == put_strike and call_expiry == put_expiry:
                # Calculate days to expiry
                dte_days = (call_expiry - datetime.now(timezone.utc)).days
                
                # Skip if too close to expiry
                if dte_days < min_dte_days:
                    continue
                
                atm_distance = abs(call_strike - current_stock_price)
                liquidity_score = min(call_bars, put_bars)  # Use the lower of the two
                
                # Calculate a combined score (lower is better)
                # Weight ATM distance heavily, then liquidity, then DTE
                combined_score = (atm_distance * 3 + 
                                (1000 / liquidity_score) if liquidity_score > 0 else float('inf') +
                                max(0, 20 - dte_days) * 0.1)  # Prefer longer DTE
                
                best_combinations.append((
                    call_ticker, put_ticker, call_strike, call_expiry, 
                    atm_distance, liquidity_score, dte_days, combined_score
                ))
    
    # Sort by combined score (best first)
    best_combinations.sort(key=lambda x: x[7])
    
    return best_combinations[:10]  # Return top 10

def main():
    print("🚀 VOLATILITY-TIMING STRADDLE STRATEGY BACKTESTER")
    print("=" * 60)
    print("📋 STRATEGY SUMMARY:")
    print("   • Long straddle when forecasted volatility > implied volatility")
    print("   • Short straddle when forecasted volatility < implied volatility") 
    print("   • Delta-neutral hedging with stock position")
    print("   • Daily rebalancing based on volatility signals")
    print("   • Options control 100 shares each (standard contract size)")
    print("   • Target: Profit from volatility mispricing")
    print("=" * 60)
    print()
    print("Initializing Backtesting Engine...")
    # Prompt for API keys or use empty strings for demo
    alpaca_key = input("Enter Alpaca API key (or leave blank): ").strip()
    alpaca_secret = input("Enter Alpaca API secret (or leave blank): ").strip()
    polygon_key = input("Enter Polygon API key (or leave blank): ").strip()
    fred_key = input("Enter FRED API key (or leave blank for fallback rate): ").strip()

    # Get risk-free rate from FRED
    print("\nFetching current risk-free rate...")
    r = get_risk_free_rate_from_fred(fred_key)
    print(f"Risk-free rate: {r*100:.2f}%\n")

    data_handler = DataHandler(
        alpaca_api_key="PKCLL4TXCDLRN76OGRAB",
        alpaca_secret="ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T",
        polygon_key="ejp0y0ppSQJzIX1W8qSoTIvL5ja3ctO9",
    )
    # Prompt user for stock, date, and frequency
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    today_str = datetime.today().strftime("%Y-%m-%d")
    
    # Calculate earliest available date (2 years ago for Polygon options data)
    earliest_date = (datetime.today() - timedelta(days=2*365)).strftime("%Y-%m-%d")
    print(f"\n📅 Polygon has 2 years of historical options data")
    print(f"   Earliest available date: {earliest_date}")
    
    start_date = input(f"Enter start date (YYYY-MM-DD) [default: {today_str}]: ").strip()
    if not start_date:
        start_date = today_str
    end_date = input(f"Enter end date (YYYY-MM-DD) [default: {today_str}]: ").strip()
    if not end_date:
        end_date = today_str
    freq = input("Enter frequency (e.g., 1D, 5Min) [default: 1D]: ").strip()
    if not freq:
        freq = "1D"
    
    # Rebalancing frequency setup
    print("\n⚖️ Rebalancing Frequency Setup:")
    print("1. Daily rebalancing (recommended)")
    print("2. Weekly rebalancing")
    print("3. Monthly rebalancing")
    print("4. Every bar (high frequency)")
    
    rebal_choice = input("Choose rebalancing frequency (1/2/3/4) [default: 1]: ").strip()
    if not rebal_choice:
        rebal_choice = "1"
    
    # Map choice to rebalancing frequency
    rebalancing_freq_map = {
        "1": "daily",
        "2": "weekly", 
        "3": "monthly",
        "4": "every_bar"
    }
    rebalancing_freq = rebalancing_freq_map.get(rebal_choice, "daily")
    print(f"Using {rebalancing_freq} rebalancing")
    
    # Delta threshold setup (in shares)
    print("\n🎯 Delta Threshold Setup:")
    print("Only rebalance when portfolio delta deviates from target by this amount (in shares)")
    delta_threshold_input = input("Enter delta threshold in shares (10-100) [default: 25]: ").strip()
    if not delta_threshold_input:
        delta_threshold = 25
    else:
        try:
            delta_threshold = float(delta_threshold_input)
            delta_threshold = max(10, min(100, delta_threshold))  # Clamp between 10 and 100 shares
        except ValueError:
            delta_threshold = 25
            print("Invalid input, using default threshold of 25 shares")
    
    print(f"Using delta threshold: {delta_threshold} shares")
    
    # Volatility forecasting setup
    print("\n🔮 Volatility Forecasting Setup:")
    print("1. 🤖 LSTM model from volatility_models folder (recommended)")
    print("   - Uses pre-trained LSTM from Google Colab")
    print("   - Processes hourly realized volatility data")
    print("   - 60-hour memory window for predictions")
    print("   - Automatic fallback to EWMA on errors")
    print("2. 📁 Other local model from volatility_models folder")
    print("3. 📊 Fallback method (exponentially weighted)")
    
    vol_choice = input("Choose volatility forecasting method (1/2/3) [default: 1]: ").strip()
    if not vol_choice:
        vol_choice = "1"
    
    # Show environment status for LSTM
    if vol_choice == "1":
        print(f"\n🔍 LSTM Environment Status:")
        if env_valid:
            print(f"   ✅ TensorFlow environment ready")
            print(f"   ✅ LSTM integration available")
        else:
            print(f"   ⚠️  TensorFlow environment issues detected")
            print(f"   ⚠️  LSTM may fall back to EWMA method")
    
    volatility_forecaster = None
    use_batch_predictions = False
    
    if vol_choice == "1":
        # LSTM model from volatility_models folder
        volatility_models_dir = "volatility_models"
        
        # Check for LSTM model files
        lstm_model_path = os.path.join(volatility_models_dir, "volatility_lstm_model.h5")
        lstm_scaler_path = os.path.join(volatility_models_dir, "volatility_scaler.pkl")
        
        print(f"\n🤖 LSTM Volatility Forecasting Setup")
        print(f"   Model path: {lstm_model_path}")
        print(f"   Scaler path: {lstm_scaler_path}")
        
        if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
            print(f"✅ LSTM model files found")
            
            # Check environment compatibility
            if not env_valid:
                print("⚠️  Environment validation failed earlier")
                print("   LSTM model may not load properly")
                proceed = input("   Continue anyway? (y/N): ").strip().lower()
                if proceed != 'y':
                    print("   Using fallback EWMA method")
                    volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
                    use_batch_predictions = False
                else:
                    print("   Proceeding with LSTM loading...")
            
            try:
                print(f"   Loading LSTM model and scaler...")
                volatility_forecaster = create_lstm_volatility_forecaster(
                    model_path=lstm_model_path,
                    scaler_path=lstm_scaler_path,
                    memory_window=60  # Match the Colab training
                )
                
                if volatility_forecaster.is_model_loaded:
                    print("✅ LSTM volatility model loaded successfully")
                    print("   Using hourly realized volatility forecasting")
                    print("   Memory window: 60 hours")
                    print("   Data format: Matches Colab training")
                    
                    # Show model information
                    model_info = volatility_forecaster.get_model_info()
                    print(f"   Model status: {'✅ Loaded' if model_info['model_loaded'] else '❌ Failed'}")
                    print(f"   Memory window: {model_info['memory_window']} hours")
                    
                    use_batch_predictions = False  # LSTM handles its own batching
                    
                    # Test the model with a simple prediction
                    print(f"   Testing model with sample data...")
                    try:
                        # Create a simple test DataFrame
                        test_dates = pd.date_range(start='2024-01-01', periods=100, freq='h', tz='America/New_York')
                        test_prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
                        test_df = pd.DataFrame({'close': test_prices}, index=test_dates)
                        
                        test_forecast = volatility_forecaster.forecast_volatility(
                            test_df, test_dates[-1], "TEST"
                        )
                        print(f"   ✅ Test forecast: {test_forecast:.4f} ({test_forecast*100:.2f}%)")
                        
                    except Exception as test_error:
                        print(f"   ⚠️  Test prediction failed: {test_error}")
                        print("   Model loaded but may have compatibility issues")
                        
                else:
                    print("❌ LSTM model failed to load")
                    print("   This could be due to:")
                    print("   - TensorFlow version incompatibility")
                    print("   - Model file corruption")
                    print("   - Missing dependencies")
                    print("   Falling back to EWMA method")
                    volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
                    use_batch_predictions = False
                    
            except Exception as e:
                print(f"❌ Error loading LSTM model: {e}")
                print("   Detailed error information:")
                import traceback
                traceback.print_exc()
                print("   Falling back to EWMA method")
                volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
                use_batch_predictions = False
        else:
            print(f"❌ LSTM model files not found:")
            print(f"   Expected: {lstm_model_path}")
            print(f"   Expected: {lstm_scaler_path}")
            print("   Please ensure your Colab-trained model files are in the volatility_models folder")
            print("   Using fallback method for now")
            volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
            use_batch_predictions = False
    
    elif vol_choice == "2":
        # Other local model from volatility_models folder
        volatility_models_dir = "volatility_models"
        
        # Create directory if it doesn't exist
        if not os.path.exists(volatility_models_dir):
            os.makedirs(volatility_models_dir)
            print(f"Created {volatility_models_dir} directory")
        
        # List available models (excluding LSTM files)
        model_files = []
        if os.path.exists(volatility_models_dir):
            for file in os.listdir(volatility_models_dir):
                if file.endswith(('.pkl', '.h5', '.keras', '.json')) and not file.startswith('volatility_lstm'):
                    model_files.append(file)
        
        if model_files:
            print(f"\nAvailable models in {volatility_models_dir}/:")
            for i, model_file in enumerate(model_files):
                print(f"  {i+1}: {model_file}")
            
            model_choice = input(f"Select model (1-{len(model_files)}) or press Enter to skip: ").strip()
            if model_choice and model_choice.isdigit():
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(model_files):
                    model_path = os.path.join(volatility_models_dir, model_files[model_idx])
                    print(f"Loading model: {model_path}")
                    
                    try:
                        volatility_forecaster = VolatilityForecaster(model_path=model_path, fallback_method="ewm")
                        print("✓ Model loaded successfully")
                    except Exception as e:
                        print(f"✗ Could not load model: {e}")
                        print("Falling back to EWMA method")
                        volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
                        use_batch_predictions = False
                    
                    # Batch prediction setup
                    print("\n📊 Batch Prediction Setup:")
                    print("1. Individual predictions (current method)")
                    print("2. Batch predictions (more efficient)")
                    
                    batch_choice = input("Choose prediction method (1/2) [default: 1]: ").strip()
                    if not batch_choice:
                        batch_choice = "1"
                    
                    use_batch_predictions = (batch_choice == "2")
                    
                    if use_batch_predictions:
                        print("Using batch predictions for efficiency")
                        print("Batch predictions will be computed after data loading...")
                    
                else:
                    print("Invalid selection, using fallback method")
                    volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
                    use_batch_predictions = False
            else:
                print("No model selected, using fallback method")
                volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
                use_batch_predictions = False
        else:
            print(f"No model files found in {volatility_models_dir}/")
            print("Please place your trained model files (.pkl, .h5, .keras, .json) in the volatility_models folder")
            print("Using fallback method for now")
            volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
            use_batch_predictions = False
    
    else:
        # Fallback method
        print("Using fallback volatility estimation (exponentially weighted)")
        volatility_forecaster = VolatilityForecaster(fallback_method="ewm")
        use_batch_predictions = False

    # Fetch and print stock time series
    stock_df = data_handler.get_stock_bars(symbol, start_date, end_date, freq)
    print(f"\nStock time series for {symbol} from {start_date} to {end_date} (freq={freq}):")
    print(stock_df)

    # Compute batch volatility forecasts if requested
    volatility_forecasts = {}
    if use_batch_predictions:
        print("\n📊 Computing batch volatility forecasts...")
        # Pre-compute volatility forecasts for the entire period
        all_dates = stock_df.index.unique()
        
        for i, date in enumerate(all_dates):
            if i % 100 == 0:  # Progress indicator
                print(f"  Processing date {i+1}/{len(all_dates)}")
            
            # Get price data up to this date
            price_data_up_to_date = stock_df['close'].loc[:date]
            if len(price_data_up_to_date) > 0:
                try:
                    vol_forecast = volatility_forecaster.forecast_volatility(price_data_up_to_date, date)
                    volatility_forecasts[date] = vol_forecast
                except Exception as e:
                    print(f"Warning: Could not forecast volatility for {date}: {e}")
                    # Use fallback
                    volatility_forecasts[date] = volatility_forecaster._fallback_forecast(price_data_up_to_date)
        
        print(f"✓ Pre-computed {len(volatility_forecasts)} volatility forecasts")

    # Display current price of the stock
    if not stock_df.empty:
        last_price = stock_df['close'].iloc[-1]
        print(f"\033[1;36mCurrent price of {symbol}: {last_price:.2f}\033[0m")
    else:
        last_price = None
        print(f"\033[1;31mWarning: No stock data found for {symbol}.\033[0m")

    # Optionally filter for strike price and expiry date
    print("\nYou can filter options by strike price and expiry date (press Enter to skip any filter).")
    strike_min = input("Enter minimum strike price (or leave blank): ").strip()
    strike_max = input("Enter maximum strike price (or leave blank): ").strip()
    exp_from = input("Enter earliest expiry date (YYYY-MM-DD, or leave blank): ").strip()
    exp_to = input("Enter latest expiry date (YYYY-MM-DD, or leave blank): ").strip()
    strike_min = float(strike_min) if strike_min else None
    strike_max = float(strike_max) if strike_max else None
    exp_from = exp_from if exp_from else None
    exp_to = exp_to if exp_to else None

    # Fetch available options for that day with filters
    options = data_handler.options_search(
        symbol,
        exp_from=exp_from,
        exp_to=exp_to,
        strike_min=strike_min,
        strike_max=strike_max,
        as_of=start_date
    )
    if not options:
        print("No options found for this stock and date.")
        return

    # Print a colored breakdown of the option ticker format
    print("\n\033[1;33mOption Ticker Format:\033[0m")
    print("  \033[1;34mO:\033[0m\033[1;32mAAPL\033[0m\033[1;35m240322\033[0m\033[1;36mC\033[0m\033[1;31m00185000\033[0m")
    print("  |    |      |     |         ")
    print("  |    |      |     |-- Strike price (e.g., 00185000 = $185.00)")
    print("  |    |      |-- Option type (C = Call, P = Put)")
    print("  |    |-- Expiry date (YYMMDD, e.g., 240322 = 2024-03-22)")
    print("  |-- Underlying symbol (e.g., AAPL)")
    print("  O: = Option contract prefix required by Polygon API\n")

    print(f"\033[1;36mCurrent price of {symbol}: {last_price:.2f}\033[0m" if last_price is not None else "")

    print("\nRanking options by available data...")
    multiplier, timespan = parse_freq(freq)

    def get_option_bar_count(opt):
        try:
            ticker = opt if opt.startswith("O:") else f"O:{opt}"
            data = data_handler._poly.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=50000,
            )
            count = len(data) if isinstance(data, list) else 0
        except Exception:
            count = 0
        return (opt, count)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        option_data_counts = list(executor.map(get_option_bar_count, options))

    top_options = sorted(option_data_counts, key=lambda x: x[1], reverse=True)[:20]
    if not top_options or top_options[0][1] == 0:
        print("No options with aggregate data found for this stock and date.")
        return

    print("\nTop 20 options by available data:")
    for idx, (opt, count) in enumerate(top_options):
        print(f"{idx}: {opt} (bars: {count})")

    print(f"\033[1;36mCurrent price of {symbol}: {last_price:.2f}\033[0m" if last_price is not None else "")

    # Separate calls and puts
    calls = []
    puts = []
    for opt, count in top_options:
        # Look for 'C' or 'P' followed by numbers (strike price)
        if 'C' in opt and opt.split('C')[1].isdigit():
            calls.append((opt, count))
        elif 'P' in opt and opt.split('P')[1].isdigit():
            puts.append((opt, count))

    print(f"\n📞 Available CALL options:")
    for idx, (opt, count) in enumerate(calls[:10]):  # Show top 10 calls
        # Extract strike price for display
        try:
            strike = float(opt.split('C')[-1]) / 1000
            print(f"  {idx}: {opt} (strike: ${strike}, bars: {count})")
        except:
            print(f"  {idx}: {opt} (bars: {count})")

    print(f"\n📞 Available PUT options:")
    for idx, (opt, count) in enumerate(puts[:10]):  # Show top 10 puts
        # Extract strike price for display
        try:
            strike = float(opt.split('P')[-1]) / 1000
            print(f"  {idx}: {opt} (strike: ${strike}, bars: {count})")
        except:
            print(f"  {idx}: {opt} (bars: {count})")

    # Get current stock price for ATM analysis
    current_stock_price = stock_df['close'].iloc[-1]
    
    print(f"\n🎯 STRADDLE SELECTION REMINDER:")
    print(f"   A straddle consists of ONE CALL + ONE PUT with the SAME:")
    print(f"   • Strike price (K)")
    print(f"   • Expiry date (T)")
    print(f"   • Underlying asset")
    print(f"   ")
    print(f"   For proper delta-neutral straddle strategy, select options with matching strikes!")
    print(f"   💡 TIP: Choose strikes close to current stock price (${current_stock_price:.2f}) for ATM straddle")
    print(f"   Example: Call at $325 + Put at $325 (same strike, close to current price)")
    print(f"   ")
    
    # Offer automatic straddle selection
    print(f"🔍 AUTOMATIC STRADDLE SELECTION:")
    print(f"   Would you like to automatically find the best straddle combinations?")
    print(f"   This will find options with same strike/expiry, close to ATM, and good liquidity.")
    
    auto_select = input("   Use automatic selection? (y/N): ").strip().lower()
    
    if auto_select == 'y':
        print(f"\n🔍 Finding best straddle combinations...")
        best_combinations = find_best_straddle_combinations(calls, puts, current_stock_price)
        
        if best_combinations:
            print(f"\n🏆 TOP STRADDLE COMBINATIONS (ranked by ATM distance + liquidity):")
            for i, (call_ticker, put_ticker, strike, expiry, atm_dist, liquidity, score) in enumerate(best_combinations):
                print(f"   {i+1}: Call {call_ticker} + Put {put_ticker}")
                print(f"       Strike: ${strike}, Expiry: {expiry.date()}")
                print(f"       ATM Distance: ${atm_dist:.2f}, Liquidity: {liquidity} bars")
                print(f"       Combined Score: {score:.2f}")
                print()
            
            choice = input(f"Select combination (1-{len(best_combinations)}) or press Enter for manual selection: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(best_combinations):
                selected_idx = int(choice) - 1
                call_ticker, put_ticker = best_combinations[selected_idx][0], best_combinations[selected_idx][1]
                print(f"✅ Selected: {call_ticker} + {put_ticker}")
            else:
                print("Proceeding with manual selection...")
                # Continue to manual selection
        else:
            print("❌ No suitable straddle combinations found. Proceeding with manual selection...")
            # Continue to manual selection
    
    # Manual selection (if auto-select was not used or failed)
    if auto_select != 'y' or 'choice' not in locals() or not choice.isdigit():
        # Select call option
        call_idx = int(input(f"\nSelect a CALL option (0-{len(calls)-1}): "))
        if call_idx < 0 or call_idx >= len(calls):
            print("Invalid call selection. Exiting.")
            return
        call_ticker = calls[call_idx][0]

        # Select put option
        put_idx = int(input(f"Select a PUT option (0-{len(puts)-1}): "))
        if put_idx < 0 or put_idx >= len(puts):
            print("Invalid put selection. Exiting.")
            return
        put_ticker = puts[put_idx][0]

    # Parse option details for validation
    try:
        call_underlying, call_expiry, call_type, call_strike = parse_option_ticker(call_ticker)
        put_underlying, put_expiry, put_type, put_strike = parse_option_ticker(put_ticker)
    except Exception as e:
        print(f"Could not parse option tickers: {e}. Exiting.")
        return
    
    # Validate that we have a proper straddle (same strike and expiry)
    if call_strike != put_strike:
        print(f"\n⚠️  WARNING: Asymmetric straddle detected!")
        print(f"   Call strike: ${call_strike}")
        print(f"   Put strike:  ${put_strike}")
        print(f"   This will result in non-zero straddle delta and require larger stock hedging.")
        print(f"   For proper delta-neutral straddle, select options with the same strike.")
        
        proceed = input("   Continue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("   Exiting. Please select options with the same strike for proper straddle strategy.")
            return
    
    if call_expiry != put_expiry:
        print(f"\n⚠️  WARNING: Different expiry dates detected!")
        print(f"   Call expiry: {call_expiry.date()}")
        print(f"   Put expiry:  {put_expiry.date()}")
        print(f"   This may cause issues with the straddle strategy.")
        
        proceed = input("   Continue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("   Exiting. Please select options with the same expiry for proper straddle strategy.")
            return
    
    # Get current stock price for ATM analysis
    current_stock_price = stock_df['close'].iloc[-1]
    atm_distance = abs(call_strike - current_stock_price)
    
    print(f"\nSelected straddle:")
    print(f"  Call: {call_ticker} (strike: ${call_strike})")
    print(f"  Put:  {put_ticker} (strike: ${put_strike})")
    print(f"  Expiry: {call_expiry.date()}")
    print(f"  Current stock price: ${current_stock_price:.2f}")
    print(f"  Distance from ATM: ${atm_distance:.2f}")
    
    if call_strike == put_strike:
        if atm_distance < 5.0:  # Within $5 of current price
            print(f"  ✅ Symmetric ATM straddle (close to current price)")
        else:
            print(f"  ⚠️  Symmetric straddle but not ATM (${atm_distance:.2f} from current price)")
    else:
        print(f"  ⚠️  Asymmetric straddle (different strikes)")
    
    # For now, we'll use the call ticker as the primary option for data fetching
    # In a full implementation, we'd fetch both call and put data
    option_ticker = call_ticker

    # Fetch and print option time series for both call and put
    call_df = data_handler.get_option_aggregates(call_ticker, start_date, end_date, timespan, multiplier)
    put_df = data_handler.get_option_aggregates(put_ticker, start_date, end_date, timespan, multiplier)
    
    print(f"\nCall option time series for {call_ticker} from {start_date} to {end_date} (freq={freq}):")
    print(f"Total bars: {len(call_df)}")
    print(f"Date range: {call_df.index[0]} to {call_df.index[-1]}")
    print(f"Price range: ${call_df['close'].min():.2f} - ${call_df['close'].max():.2f}")
    print(f"Trades range: {call_df['trades'].min()} - {call_df['trades'].max()}")
    print(f"Full time series:")
    print(call_df.head(5))
    print("...")
    print(call_df.tail(5))
    
    print(f"\nPut option time series for {put_ticker} from {start_date} to {end_date} (freq={freq}):")
    print(f"Total bars: {len(put_df)}")
    print(f"Date range: {put_df.index[0]} to {put_df.index[-1]}")
    print(f"Price range: ${put_df['close'].min():.2f} - ${put_df['close'].max():.2f}")
    print(f"Trades range: {put_df['trades'].min()} - {put_df['trades'].max()}")
    print(f"Full time series:")
    print(put_df.head(5))
    print("...")
    print(put_df.tail(5))

    # Align stock, call, and put data on timestamps
    combined_df = stock_df[['close']].copy()
    combined_df.columns = [f'{symbol}_close']
    
    call_close_df = call_df[['close']].copy()
    call_close_df.columns = [f'{call_ticker}_close']
    
    put_close_df = put_df[['close']].copy()
    put_close_df.columns = [f'{put_ticker}_close']
    
    combined_df = combined_df.join(call_close_df, how='inner')
    combined_df = combined_df.join(put_close_df, how='inner')
    combined_df = combined_df.dropna()
    
    if combined_df.empty:
        print("No overlapping data between stock and options. Exiting.")
        return

    # Initialize GreeksEngine
    print("Initializing GreeksEngine...")
    greeks_engine = GreeksEngine()
    
    # Check QuantLib availability
    helper_info = greeks_engine.helper()
    print(f"QuantLib available: {helper_info['quantlib_available']}")
    print(f"Pricing model: {helper_info['pricing_model']}")

    # Initialize Portfolio
    portfolio = Portfolio(initial_cash=100000)

    # Initial positions: straddle setup
    first_row = combined_df.iloc[0]
    S0 = first_row[f'{symbol}_close']
    
    # Parse call and put option details
    try:
        call_underlying, call_expiry, call_type, call_strike = parse_option_ticker(call_ticker)
        put_underlying, put_expiry, put_type, put_strike = parse_option_ticker(put_ticker)
        print(f"Parsed call: {call_underlying} {call_type} {call_strike} expiring {call_expiry}")
        print(f"Parsed put:  {put_underlying} {put_type} {put_strike} expiring {put_expiry}")
    except Exception as e:
        print(f"Could not parse option tickers for strike/expiry: {e}. Exiting.")
        return
    
    # Use the earlier expiry for time calculations
    expiry = min(call_expiry, put_expiry)
    T0 = max((expiry - combined_df.index[0]).days / 365.0, 1/365)
    
    # Get initial implied volatility from both call and put prices
    print("\n📊 Getting initial implied volatility...")
    initial_call_price = float(first_row[f'{call_ticker}_close'])
    initial_put_price = float(first_row[f'{put_ticker}_close'])
    sigma_implied = 0.20  # Default fallback
    
    try:
        from backtesting_module.quantlib import compute_iv_quantlib
        
        # Calculate days to maturity
        days_to_maturity = max((expiry - combined_df.index[0]).days, 1)
        
        # Calculate implied vol from call
        sigma_implied_call = compute_iv_quantlib(
            spot_price=float(S0),
            option_price=initial_call_price,
            strike_price=call_strike,
            days_to_maturity=days_to_maturity,
            risk_free_rate=r,
            option_type="call",
            exercise_style="american",
            tree="crr",
            steps=1000
        )
        
        # Calculate implied vol from put
        sigma_implied_put = compute_iv_quantlib(
            spot_price=float(S0),
            option_price=initial_put_price,
            strike_price=put_strike,
            days_to_maturity=days_to_maturity,
            risk_free_rate=r,
            option_type="put",
            exercise_style="american",
            tree="crr",
            steps=1000
        )
        
        # Use average of call and put implied vol
        sigma_implied = (sigma_implied_call + sigma_implied_put) / 2
        print(f"   Call implied volatility: {sigma_implied_call:.4f}")
        print(f"   Put implied volatility: {sigma_implied_put:.4f}")
        print(f"   Average implied volatility: {sigma_implied:.4f}")
    except Exception as e:
        print(f"Error calculating initial implied volatility: {e}")
        print("Using fallback volatility of 20%")
    
    # Get initial volatility forecast for comparison
    print("\n📊 Getting initial volatility forecast...")
    if isinstance(volatility_forecaster, LSTMVolatilityForecaster):
        # Detect data interval for initial forecast
        interval = volatility_forecaster.detect_time_interval(stock_df)
        print(f"   Data interval: {interval}")
        
        # Check if we have enough data for LSTM
        if interval == 'minute':
            min_data_points = volatility_forecaster.memory_window * 60
        elif interval.startswith('minute'):
            minute_val = int(interval.replace('minute', ''))
            min_data_points = volatility_forecaster.memory_window * 60 // minute_val
        elif interval == 'hour':
            min_data_points = volatility_forecaster.memory_window
        elif interval == 'day':
            min_data_points = volatility_forecaster.memory_window // 24
        else:
            min_data_points = volatility_forecaster.memory_window * 60
        
        if len(stock_df) < min_data_points:
            print(f"   ⚠️  Insufficient data for LSTM: Need {min_data_points} {interval}, have {len(stock_df)}")
            sigma_forecast = volatility_forecaster._fallback_forecast(stock_df, symbol)
        else:
            sigma_forecast = volatility_forecaster.forecast_volatility(stock_df, combined_df.index[0], symbol)
    elif use_batch_predictions:
        # Use pre-computed forecast
        sigma_forecast = volatility_forecasts.get(combined_df.index[0], 0.2)
    else:
        # Individual prediction
        initial_price_data = stock_df['close'].loc[:combined_df.index[0]]
        sigma_forecast = volatility_forecaster.forecast_volatility(initial_price_data, combined_df.index[0])
    print(f"   Initial volatility forecast: {sigma_forecast:.4f}")
    
    # Calculate initial vol_diff signal
    vol_diff = sigma_forecast - sigma_implied
    print(f"   Volatility difference (forecast - implied): {vol_diff:.4f}")
    
    # Determine initial position based on vol_diff
    threshold = 0.02  # 2% threshold to avoid noise trades
    if vol_diff > threshold:
        position_side = 1  # Long straddle
        print(f"   Signal: LONG straddle (vol_diff > {threshold:.3f})")
    elif vol_diff < -threshold:
        position_side = -1  # Short straddle
        print(f"   Signal: SHORT straddle (vol_diff < -{threshold:.3f})")
    else:
        position_side = 0  # No position
        print(f"   Signal: NO POSITION (|vol_diff| <= {threshold:.3f})")
    
    # Compute initial Greeks for both call and put using IMPLIED volatility
    try:
        print(f"Computing initial Greeks for straddle using IMPLIED volatility...")
        print(f"  S0={float(S0):.2f}, T={T0:.4f}, sigma_implied={sigma_implied:.2f}, r={r:.4f}")
        
        # Call Greeks
        call_greeks = greeks_engine.compute(
            symbol=call_ticker,
            underlying_price=float(S0),
            strike=call_strike,
            time_to_expiry=T0,
            volatility=sigma_implied,
            risk_free_rate=r,
            option_type="call"
        )
        
        # Put Greeks
        put_greeks = greeks_engine.compute(
            symbol=put_ticker,
            underlying_price=float(S0),
            strike=put_strike,
            time_to_expiry=T0,
            volatility=sigma_implied,
            risk_free_rate=r,
            option_type="put"
        )
        
        call_delta = call_greeks['delta']
        call_vega = call_greeks['vega']
        put_delta = put_greeks['delta']
        put_vega = put_greeks['vega']
        
        # Straddle Greeks
        straddle_delta = call_delta + put_delta  # For ATM straddle, this should be ≈ 0 (call_delta ≈ 0.5, put_delta ≈ -0.5)
        straddle_vega = call_vega + put_vega     # Straddle vega is sum of individual vegas
        
        print(f"  Call delta: {call_delta:.4f}, vega: {call_vega:.4f}")
        print(f"  Put delta: {put_delta:.4f}, vega: {put_vega:.4f}")
        print(f"  Straddle delta: {straddle_delta:.4f}, vega: {straddle_vega:.4f}")
        
        # Educational output about straddle delta and contract size
        if abs(straddle_delta) > 0.1:
            print(f"  ⚠️  Note: Straddle delta is {straddle_delta:.4f} (not close to 0)")
            if call_strike != put_strike:
                print(f"     This indicates an asymmetric straddle (different strikes)")
            else:
                print(f"     This indicates the strike is not ATM (current price: ${S0:.2f}, strike: ${call_strike})")
            print(f"     For ATM straddle, delta should be close to 0")
        else:
            print(f"  ✅ Straddle delta is close to 0 (symmetric ATM straddle)")
        
        print(f"  📊 Note: Each option contract controls 100 shares")
        print(f"     Stock hedge calculation: -straddle_delta × 100 × straddle_qty")
        
        if pd.isna(call_delta) or pd.isna(put_delta):
            print("Error: Initial delta is NaN. Exiting.")
            return
            
    except Exception as e:
        print(f"Error computing initial Greeks: {e}")
        print("This might be a QuantLib issue. Check the error details above.")
        return
    # Set up straddle position based on vol_diff signal
    if position_side == 0:
        print(f"\nNo initial position (vol_diff within threshold)")
        call_qty = 0
        put_qty = 0
        stock_qty = 0
    else:
        # Calculate position size based on vega risk budget
        V_vega = 1000  # Vega risk budget (can be made configurable)
        
        # Use actual straddle vega for position sizing
        straddle_qty = position_side * (V_vega / straddle_vega) if straddle_vega > 0 else 0
        
        # Round to reasonable position size
        straddle_qty = int(round(straddle_qty))
        if abs(straddle_qty) > 10:  # Cap position size
            straddle_qty = 10 if straddle_qty > 0 else -10
        
        # For ATM straddle: both call and put have same quantity
        call_qty = straddle_qty
        put_qty = straddle_qty
        
        # Calculate stock hedge for the straddle
        # Use actual straddle delta for hedging (should be close to 0 for ATM straddle)
        stock_qty = -straddle_delta * 100 * straddle_qty
        stock_qty = int(round(stock_qty))
        
        print(f"\nInitial position: {position_side} straddle")
        print(f"  Call quantity: {call_qty} {call_ticker}")
        print(f"  Put quantity: {put_qty} {put_ticker}")
        print(f"  Stock hedge: {stock_qty} {symbol}")
        print(f"  Vega exposure: {straddle_qty * straddle_vega:.2f}")
    
    # Initial fills
    t0 = pd.Timestamp(combined_df.index[0])
    if pd.isna(t0):
        print("Error: Invalid timestamp in data. Exiting.")
        return
    
    if call_qty != 0:
        portfolio.update_with_fill(Fill(symbol=call_ticker, qty=call_qty, price=float(first_row[f'{call_ticker}_close']), timestamp=t0))
    if put_qty != 0:
        portfolio.update_with_fill(Fill(symbol=put_ticker, qty=put_qty, price=float(first_row[f'{put_ticker}_close']), timestamp=t0))
    if stock_qty != 0:
        portfolio.update_with_fill(Fill(symbol=symbol, qty=stock_qty, price=float(first_row[f'{symbol}_close']), timestamp=t0))

    # Main backtest loop
    last_rebalance_date = None
    rebalance_count = 0
    position_changes = 0
    
    # LSTM performance tracking
    lstm_forecast_count = 0
    lstm_fallback_count = 0
    lstm_error_count = 0
    
    # Position tracking for hysteresis
    current_position_side = position_side  # Track current position
    last_vol_diff = vol_diff  # Track last vol_diff for hysteresis
    
    for i, (ts, row) in enumerate(combined_df.iterrows()):
        S = float(row[f'{symbol}_close'])
        call_price = float(row[f'{call_ticker}_close'])
        put_price = float(row[f'{put_ticker}_close'])
        # Convert ts to proper timestamp
        try:
            ts_timestamp = pd.Timestamp(ts)
            if pd.isna(ts_timestamp):
                print(f"Warning: Invalid timestamp {ts}. Skipping.")
                continue
        except Exception as e:
            print(f"Error converting timestamp {ts}: {e}. Skipping.")
            continue
        T = max((expiry - ts_timestamp).days / 365.0, 1/365)
        
        # First, get current positions to calculate portfolio delta
        portfolio_view = portfolio.portfolio_view()
        current_call_qty = portfolio_view['options'].get(call_ticker, None)
        current_put_qty = portfolio_view['options'].get(put_ticker, None)
        current_stock_qty = portfolio_view['stocks'].get(symbol, None)
        current_call_qty = current_call_qty.qty if current_call_qty is not None else 0
        current_put_qty = current_put_qty.qty if current_put_qty is not None else 0
        current_stock_qty = current_stock_qty.qty if current_stock_qty is not None else 0
        
        # Calculate current portfolio delta (we need Greeks for this)
        current_portfolio_delta = 0.0
        if current_call_qty != 0 or current_put_qty != 0:
            try:
                # Get current Greeks for delta calculation using fresh implied vol
                current_greeks = greeks_engine.compute(
                    symbol=call_ticker,
                    underlying_price=S,
                    strike=call_strike,
                    time_to_expiry=T,
                    volatility=sigma_implied,  # Use fresh implied vol
                    risk_free_rate=r,
                    option_type="call"
                )
                current_call_delta = current_greeks['delta']
                
                current_put_greeks = greeks_engine.compute(
                    symbol=put_ticker,
                    underlying_price=S,
                    strike=put_strike,
                    time_to_expiry=T,
                    volatility=sigma_implied,  # Use fresh implied vol
                    risk_free_rate=r,
                    option_type="put"
                )
                current_put_delta = current_put_greeks['delta']
                
                current_portfolio_delta = (100 * current_call_qty * current_call_delta + 
                                          100 * current_put_qty * current_put_delta + 
                                          current_stock_qty)
            except Exception as e:
                print(f"Warning: Could not calculate current portfolio delta at {ts}: {e}")
                current_portfolio_delta = 0.0
        
        # Check if we should rebalance based on frequency and delta deviation
        should_rebal = should_rebalance(
            ts_timestamp, last_rebalance_date, rebalancing_freq, 
            current_delta=current_portfolio_delta, target_delta=0.0, delta_threshold=delta_threshold
        )
        
        if should_rebal:
            # 1. Calculate current implied volatility from both call and put prices
            try:
                from backtesting_module.quantlib import compute_iv_quantlib
                
                # Calculate days to maturity
                days_to_maturity = max((expiry - ts_timestamp).days, 1)
                
                # Calculate implied vol from call
                sigma_implied_call = compute_iv_quantlib(
                    spot_price=S,
                    option_price=call_price,
                    strike_price=call_strike,
                    days_to_maturity=days_to_maturity,
                    risk_free_rate=r,
                    option_type="call",
                    exercise_style="american",
                    tree="crr",
                    steps=1000
                )
                
                # Calculate implied vol from put
                sigma_implied_put = compute_iv_quantlib(
                    spot_price=S,
                    option_price=put_price,
                    strike_price=put_strike,
                    days_to_maturity=days_to_maturity,
                    risk_free_rate=r,
                    option_type="put",
                    exercise_style="american",
                    tree="crr",
                    steps=1000
                )
                
                # Use average of call and put implied vol
                sigma_implied = (sigma_implied_call + sigma_implied_put) / 2
            except Exception as e:
                print(f"Warning: Could not calculate implied volatility at {ts}: {e}")
                sigma_implied = 0.20  # Fallback
            
            # 2. Get volatility forecast with enhanced error handling
            try:
                if isinstance(volatility_forecaster, LSTMVolatilityForecaster):
                    # LSTM forecaster needs the full stock data up to current timestamp
                    # Fix: Compute required rows based on detected interval
                    interval = volatility_forecaster.detect_time_interval(stock_df)
                    if interval == 'minute':
                        need = volatility_forecaster.memory_window * 60   # 60 hours * 60 minutes
                    elif interval.startswith('minute'):
                        m = int(interval.replace('minute',''))
                        need = volatility_forecaster.memory_window * (60 // m)
                    elif interval == 'hour':
                        need = volatility_forecaster.memory_window
                    elif interval == 'day':
                        need = max(1, volatility_forecaster.memory_window // 24)
                    else:
                        need = volatility_forecaster.memory_window * 60  # safe default
                    
                    buffer = int(0.25 * need)  # 25% buffer
                    max_data_points = need + buffer
                    
                    if len(stock_df) > max_data_points:
                        # Use only the most recent data to save memory
                        current_stock_data = stock_df.tail(max_data_points)
                    else:
                        current_stock_data = stock_df.loc[:ts_timestamp]
                    
                    # Check if we have enough data for LSTM
                    # Detect data interval and calculate minimum required data points
                    interval = volatility_forecaster.detect_time_interval(current_stock_data)
                    
                    if interval == 'minute':
                        min_data_points = volatility_forecaster.memory_window * 60  # 60 hours = 3600 minutes
                    elif interval.startswith('minute'):
                        # For multi-minute data, calculate accordingly
                        minute_val = int(interval.replace('minute', ''))
                        min_data_points = volatility_forecaster.memory_window * 60 // minute_val
                    elif interval == 'hour':
                        min_data_points = volatility_forecaster.memory_window  # 60 hours
                    elif interval == 'day':
                        min_data_points = volatility_forecaster.memory_window // 24  # Convert hours to days
                    else:
                        min_data_points = volatility_forecaster.memory_window * 60  # Default to minute assumption
                    
                    if len(current_stock_data) < min_data_points:
                        print(f"⚠️  Insufficient data for LSTM at {ts_timestamp.date()}: Need {min_data_points} {interval} ({volatility_forecaster.memory_window} hours), have {len(current_stock_data)}")
                        sigma_forecast = volatility_forecaster._fallback_forecast(current_stock_data, symbol)
                        lstm_fallback_count += 1
                    else:
                        sigma_forecast = volatility_forecaster.forecast_volatility(current_stock_data, ts_timestamp, symbol)
                        
                        # Validate LSTM forecast
                        if pd.isna(sigma_forecast) or sigma_forecast <= 0:
                            print(f"⚠️  Invalid LSTM forecast at {ts_timestamp.date()}: {sigma_forecast}")
                            sigma_forecast = volatility_forecaster._fallback_forecast(current_stock_data, symbol)
                            lstm_fallback_count += 1
                        else:
                            lstm_forecast_count += 1
                            
                elif use_batch_predictions:
                    sigma_forecast = volatility_forecasts.get(ts_timestamp, 0.2)
                else:
                    current_price_data = stock_df['close'].loc[:ts_timestamp]
                    sigma_forecast = volatility_forecaster.forecast_volatility(current_price_data, ts_timestamp)
                    
            except Exception as vol_error:
                print(f"⚠️  Volatility forecasting error at {ts_timestamp.date()}: {vol_error}")
                lstm_error_count += 1
                # Use fallback method
                if isinstance(volatility_forecaster, LSTMVolatilityForecaster):
                    current_stock_data = stock_df.loc[:ts_timestamp]
                    sigma_forecast = volatility_forecaster._fallback_forecast(current_stock_data, symbol)
                    lstm_fallback_count += 1
                else:
                    sigma_forecast = 0.2  # Default volatility
            
            # 3. Sanity check and clip volatility forecasts
            sigma_forecast = max(0.05, min(2.0, sigma_forecast))  # Clip to 5%-200% range
            sigma_implied = max(0.05, min(2.0, sigma_implied))    # Clip to 5%-200% range
            
            # 4. Compute vol_diff signal
            vol_diff = sigma_forecast - sigma_implied
            
            # 5. Determine new position based on vol_diff (with no-trade band and hysteresis)
            new_position_side = current_position_side  # Default to current position
            
            min_vol_diff = 0.03  # 3% minimum vol difference to trade
            hysteresis_threshold = 0.05  # Additional 5% to flip position
            
            # Only change position if vol_diff crosses threshold with hysteresis
            if current_position_side <= 0 and vol_diff > max(threshold, min_vol_diff) + hysteresis_threshold:
                new_position_side = 1  # Long straddle
            elif current_position_side >= 0 and vol_diff < -max(threshold, min_vol_diff) - hysteresis_threshold:
                new_position_side = -1  # Short straddle
            
            # 5. Calculate Greeks for both call and put using IMPLIED volatility
            try:
                # Call Greeks
                call_greeks = greeks_engine.compute(
                    symbol=call_ticker,
                    underlying_price=S,
                    strike=call_strike,
                    time_to_expiry=T,
                    volatility=sigma_implied,
                    risk_free_rate=r,
                    option_type="call"
                )
                
                # Put Greeks
                put_greeks = greeks_engine.compute(
                    symbol=put_ticker,
                    underlying_price=S,
                    strike=put_strike,
                    time_to_expiry=T,
                    volatility=sigma_implied,
                    risk_free_rate=r,
                    option_type="put"
                )
                
                call_delta = call_greeks['delta']
                call_vega = call_greeks['vega']
                put_delta = put_greeks['delta']
                put_vega = put_greeks['vega']
                
                # Straddle Greeks
                straddle_delta = call_delta + put_delta  # For ATM straddle, this should be ≈ 0
                straddle_vega = call_vega + put_vega
                
                if pd.isna(call_delta) or pd.isna(put_delta):
                    print(f"Warning: NaN Greeks at {ts}. Skipping rebalancing.")
                    continue
                    
            except Exception as e:
                print(f"Error calculating Greeks at {ts}: {e}. Skipping rebalancing.")
                continue
            
            # 6. Calculate new position sizes
            V_vega = 1000  # Vega risk budget
            new_call_qty = 0
            new_put_qty = 0
            new_stock_qty = 0
            
            if new_position_side != 0:
                # Calculate straddle position size using actual straddle vega
                new_straddle_qty = new_position_side * (V_vega / straddle_vega) if straddle_vega > 0 else 0
                new_straddle_qty = int(round(new_straddle_qty))
                if abs(new_straddle_qty) > 10:  # Cap position size
                    new_straddle_qty = 10 if new_straddle_qty > 0 else -10
                
                # For ATM straddle: both call and put have same quantity
                new_call_qty = new_straddle_qty
                new_put_qty = new_straddle_qty
                
                # Calculate stock hedge for the straddle using actual straddle delta
                new_stock_qty = -straddle_delta * 100 * new_straddle_qty
                new_stock_qty = int(round(new_stock_qty))
            
            # 7. Calculate new portfolio delta for monitoring
            new_portfolio_delta = (100 * new_call_qty * call_delta + 
                                 100 * new_put_qty * put_delta + 
                                 new_stock_qty)
            
            # Log delta deviation for monitoring
            delta_deviation = abs(new_portfolio_delta - 0.0)  # Target delta is 0
            if delta_deviation > delta_threshold:
                print(f"⚠️  Delta deviation at {ts_timestamp.date()}: {new_portfolio_delta:.4f} (threshold: {delta_threshold})")
            
            # 8. Execute trades if position changed
            call_trade = new_call_qty - current_call_qty
            put_trade = new_put_qty - current_put_qty
            stock_trade = new_stock_qty - current_stock_qty
            
            # Only execute trades if we're rebalancing AND there's a position change
            if should_rebal and (call_trade != 0 or put_trade != 0 or stock_trade != 0):
                if call_trade != 0:
                    portfolio.update_with_fill(Fill(symbol=call_ticker, qty=call_trade, price=call_price, timestamp=ts_timestamp))
                    position_changes += 1
                    print(f"📈 Call trade at {ts_timestamp.date()}: {current_call_qty} → {new_call_qty} {call_ticker}")
                
                if put_trade != 0:
                    portfolio.update_with_fill(Fill(symbol=put_ticker, qty=put_trade, price=put_price, timestamp=ts_timestamp))
                    position_changes += 1
                    print(f"📈 Put trade at {ts_timestamp.date()}: {current_put_qty} → {new_put_qty} {put_ticker}")
                
                if stock_trade != 0:
                    portfolio.update_with_fill(Fill(symbol=symbol, qty=stock_trade, price=S, timestamp=ts_timestamp))
                    print(f"📈 Stock hedge at {ts_timestamp.date()}: {current_stock_qty} → {new_stock_qty} {symbol}")
                
                # Log rebalancing info with LSTM status
                rebalance_count += 1
                lstm_status = ""
                if isinstance(volatility_forecaster, LSTMVolatilityForecaster):
                    if lstm_forecast_count > 0 and (lstm_forecast_count + lstm_fallback_count) > 0:
                        lstm_success_rate = (lstm_forecast_count / (lstm_forecast_count + lstm_fallback_count)) * 100
                        lstm_status = f" [LSTM: {lstm_success_rate:.0f}%]"
                    else:
                        lstm_status = " [LSTM: Loading...]"
                
                print(f"🔄 Rebalanced at {ts_timestamp.date()}: vol_diff={vol_diff:.4f}, signal={new_position_side}{lstm_status}")
                
                # Update quantities and position tracking
                current_call_qty = new_call_qty
                current_put_qty = new_put_qty
                current_stock_qty = new_stock_qty
                
                # Update position tracking for hysteresis
                if new_position_side != current_position_side:
                    current_position_side = new_position_side
                    last_vol_diff = vol_diff
            
            # Update last rebalance date
            last_rebalance_date = ts_timestamp
        
        # Update portfolio value (always)
        current_prices = {symbol: S, call_ticker: call_price, put_ticker: put_price}
        portfolio.update_portfolio_value(current_prices, ts_timestamp)
    print("\nBacktest complete.")
    metrics = portfolio.get_performance_metrics(risk_free_rate=r)
    print(f"Total return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Strategy summary
    print(f"\n📊 Volatility-Timing Strategy Summary:")
    print(f"   Volatility threshold: {threshold:.3f}")
    print(f"   Vega risk budget: $1000")
    print(f"   Total position changes: {position_changes}")
    print(f"   Rebalancing frequency: {rebalancing_freq}")
    print(f"   Total rebalances: {rebalance_count}")
    print(f"   Rebalance rate: {rebalance_count/len(combined_df)*100:.1f}% of bars")
    
    # Print out all trades executed during the backtest
    print(f"\n📈 Trade History:")
    if portfolio.fills_history:
        for i, fill in enumerate(portfolio.fills_history, 1):
            side = "BUY" if fill.qty > 0 else "SELL"
            print(f"   Trade {i}: {fill.symbol} {side} {abs(fill.qty)} @ ${fill.price:.2f} on {fill.timestamp}")
    else:
        print("   No trades executed during backtest")
    
    # Volatility forecasting performance
    if volatility_forecaster:
        if isinstance(volatility_forecaster, LSTMVolatilityForecaster):
            print(f"\n🤖 LSTM Volatility Forecasting Performance:")
            
            # Show runtime statistics
            total_forecasts = lstm_forecast_count + lstm_fallback_count + lstm_error_count
            if total_forecasts > 0:
                lstm_success_rate = (lstm_forecast_count / total_forecasts) * 100
                fallback_rate = (lstm_fallback_count / total_forecasts) * 100
                error_rate = (lstm_error_count / total_forecasts) * 100
                
                print(f"   Runtime Statistics:")
                print(f"   - LSTM forecasts: {lstm_forecast_count} ({lstm_success_rate:.1f}%)")
                print(f"   - Fallback forecasts: {lstm_fallback_count} ({fallback_rate:.1f}%)")
                print(f"   - Errors: {lstm_error_count} ({error_rate:.1f}%)")
                print(f"   - Total forecasts: {total_forecasts}")
            
            # Show model performance metrics
            vol_metrics = volatility_forecaster.get_performance_metrics()
            if vol_metrics:
                print(f"\n   Model Performance Metrics:")
                print(f"   - MAE: {vol_metrics['mae']*100:.2f}%")
                print(f"   - RMSE: {vol_metrics['rmse']*100:.2f}%")
                print(f"   - MAPE: {vol_metrics['mape']:.2f}%")
                print(f"   - Correlation: {vol_metrics['correlation']:.3f}")
                print(f"   - Forecasts made: {vol_metrics['n_forecasts']}")
            
            # Show model info
            model_info = volatility_forecaster.get_model_info()
            print(f"\n📊 LSTM Model Information:")
            print(f"   Model loaded: {model_info['model_loaded']}")
            print(f"   Memory window: {model_info['memory_window']} hours")
            print(f"   Total forecasts: {model_info['n_forecasts']}")
            
            # Memory optimization summary
            print(f"\n💾 Memory Optimization:")
            print(f"   Data window: {volatility_forecaster.memory_window + 24} hours")
            print(f"   Memory-efficient processing: ✅ Enabled")
            
        else:
            vol_metrics = volatility_forecaster.get_performance_metrics()
            if vol_metrics:
                print(f"\n📊 Volatility Forecasting Performance:")
                print(f"   MAE: {vol_metrics['mae']*100:.2f}%")
                print(f"   RMSE: {vol_metrics['rmse']*100:.2f}%")
                print(f"   MAPE: {vol_metrics['mape']:.2f}%")
                print(f"   Correlation: {vol_metrics['correlation']:.3f}")
                print(f"   Forecasts made: {vol_metrics['n_forecasts']}")

if __name__ == "__main__":
    main()