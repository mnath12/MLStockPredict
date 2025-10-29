#!/usr/bin/env python3
"""
Complete LSTM Integration Test Script

This script tests the full LSTM volatility forecaster integration
with the main backtesting system to ensure everything works properly.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the backtesting_module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backtesting_module'))

def test_environment():
    """Test the environment setup."""
    print("🧪 Testing Environment Setup...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        print(f"✅ Current environment: {conda_env}")
        
        return True
    except ImportError:
        print("❌ TensorFlow not available")
        return False

def test_lstm_model_files():
    """Test that LSTM model files exist."""
    print("\n🧪 Testing LSTM Model Files...")
    
    model_path = "volatility_models/volatility_lstm_model.h5"
    scaler_path = "volatility_models/volatility_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"✅ LSTM model files found")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        return True
    else:
        print(f"❌ LSTM model files not found")
        print(f"   Expected: {model_path}")
        print(f"   Expected: {scaler_path}")
        return False

def test_lstm_forecaster():
    """Test the LSTM forecaster directly."""
    print("\n🧪 Testing LSTM Forecaster...")
    
    try:
        from lstm_volatility_forecaster import create_lstm_volatility_forecaster
        
        forecaster = create_lstm_volatility_forecaster(
            model_path="volatility_models/volatility_lstm_model.h5",
            scaler_path="volatility_models/volatility_scaler.pkl",
            memory_window=60
        )
        
        if forecaster.is_model_loaded:
            print("✅ LSTM forecaster loaded successfully")
            
            # Test with sample data
            test_dates = pd.date_range(start='2024-01-01', periods=100, freq='h', tz='America/New_York')
            test_prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
            test_df = pd.DataFrame({'close': test_prices}, index=test_dates)
            
            forecast = forecaster.forecast_volatility(test_df, test_dates[-1], "TEST")
            print(f"✅ Test forecast: {forecast:.4f} ({forecast*100:.2f}%)")
            
            return True
        else:
            print("❌ LSTM forecaster failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LSTM forecaster: {e}")
        return False

def test_data_handler():
    """Test the data handler."""
    print("\n🧪 Testing Data Handler...")
    
    try:
        from data_handler import DataHandler
        
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
        data_handler = DataHandler(
            alpaca_api_key=ALPACA_API_KEY,
            alpaca_secret=ALPACA_SECRET_KEY,
            polygon_key=POLYGON_API_KEY,
        )
        
        # Test fetching stock data
        symbol = "AAPL"
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        stock_df = data_handler.get_stock_bars(symbol, start_date, end_date, "1Min")
        
        if not stock_df.empty:
            print(f"✅ Stock data fetched successfully")
            print(f"   Shape: {stock_df.shape}")
            print(f"   Date range: {stock_df.index[0]} to {stock_df.index[-1]}")
            return stock_df
        else:
            print("❌ No stock data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error testing data handler: {e}")
        return None

def test_main_integration():
    """Test the main integration by running a minimal backtest."""
    print("\n🧪 Testing Main Integration...")
    
    try:
        # Import main components
        from main import validate_environment, get_risk_free_rate_from_fred
        from data_handler import DataHandler
        from lstm_volatility_forecaster import create_lstm_volatility_forecaster
        
        # Test environment validation
        env_valid = validate_environment()
        print(f"Environment validation: {'✅ PASS' if env_valid else '❌ FAIL'}")
        
        # Test risk-free rate
        r = get_risk_free_rate_from_fred()
        print(f"Risk-free rate: {r*100:.2f}%")
        
        # Test LSTM forecaster creation
        forecaster = create_lstm_volatility_forecaster(
            model_path="volatility_models/volatility_lstm_model.h5",
            scaler_path="volatility_models/volatility_scaler.pkl",
            memory_window=60
        )
        
        if forecaster.is_model_loaded:
            print("✅ LSTM forecaster integration ready")
            return True
        else:
            print("❌ LSTM forecaster integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing main integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Complete LSTM Integration Test")
    print("=" * 60)
    
    # Test 1: Environment
    env_ok = test_environment()
    
    # Test 2: Model files
    files_ok = test_lstm_model_files()
    
    # Test 3: LSTM forecaster
    forecaster_ok = test_lstm_forecaster()
    
    # Test 4: Data handler
    stock_df = test_data_handler()
    data_ok = stock_df is not None
    
    # Test 5: Main integration
    integration_ok = test_main_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Summary:")
    print(f"   Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   Model files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"   LSTM forecaster: {'✅ PASS' if forecaster_ok else '❌ FAIL'}")
    print(f"   Data handler: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"   Main integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    all_tests_passed = all([env_ok, files_ok, forecaster_ok, data_ok, integration_ok])
    
    if all_tests_passed:
        print("\n🎉 All integration tests passed!")
        print("\n📋 Ready for backtesting:")
        print("   1. Run: python backtesting_module/main.py")
        print("   2. Choose option 1 for LSTM volatility forecasting")
        print("   3. Enter your parameters (stock symbol, dates, etc.)")
        print("   4. The system will use your LSTM model for volatility predictions")
        print("\n💡 Recommended settings:")
        print("   - Stock: TSLA (highly liquid options)")
        print("   - Date range: Last 30 days")
        print("   - Frequency: 1D (daily)")
        print("   - Rebalancing: Daily")
    else:
        print("\n❌ Some integration tests failed.")
        print("\n🔧 Troubleshooting:")
        if not env_ok:
            print("   - Install TensorFlow: pip install tensorflow>=2.13.0")
            print("   - Or activate tf-m1 conda environment")
        if not files_ok:
            print("   - Ensure volatility_lstm_model.h5 and volatility_scaler.pkl are in volatility_models/")
        if not forecaster_ok:
            print("   - Check TensorFlow compatibility")
            print("   - Verify model files are not corrupted")
        if not data_ok:
            print("   - Check API keys")
            print("   - Verify internet connection")
        if not integration_ok:
            print("   - Check all dependencies are installed")
            print("   - Verify file paths are correct")

if __name__ == "__main__":
    main()
