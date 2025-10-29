#!/usr/bin/env python3
"""
Test script for LSTM volatility model integration.

This script tests the LSTM volatility forecaster integration
without running the full backtesting system.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the backtesting_module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backtesting_module'))

from lstm_volatility_forecaster import LSTMVolatilityForecaster, create_lstm_volatility_forecaster
from data_handler import DataHandler

def test_lstm_model_loading():
    """Test loading the LSTM model and scaler."""
    print("🧪 Testing LSTM model loading...")
    
    model_path = "volatility_models/volatility_lstm_model.h5"
    scaler_path = "volatility_models/volatility_scaler.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure your Colab-trained model is in the volatility_models folder")
        return False
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        print("   Please ensure your Colab-trained scaler is in the volatility_models folder")
        return False
    
    try:
        # Create LSTM forecaster
        forecaster = create_lstm_volatility_forecaster(
            model_path=model_path,
            scaler_path=scaler_path,
            memory_window=60
        )
        
        if forecaster.is_model_loaded:
            print("✅ LSTM model loaded successfully")
            
            # Show model info
            model_info = forecaster.get_model_info()
            print(f"   Memory window: {model_info['memory_window']} hours")
            print(f"   Model path: {model_info['model_path']}")
            print(f"   Scaler path: {model_info['scaler_path']}")
            
            return True
        else:
            print("❌ LSTM model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        return False

def test_data_handler():
    """Test the data handler for fetching stock data."""
    print("\n🧪 Testing data handler...")
    
    try:
        # Initialize data handler with your API keys
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
        data_handler = DataHandler(
            alpaca_api_key=ALPACA_API_KEY,
            alpaca_secret=ALPACA_SECRET_KEY,
            polygon_key=POLYGON_API_KEY,
        )
        
        # Test fetching stock data
        symbol = "AAPL"
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        print(f"   Fetching {symbol} data from {start_date} to {end_date}")
        stock_df = data_handler.get_stock_bars(symbol, start_date, end_date, "1Min")
        
        if not stock_df.empty:
            print(f"✅ Stock data fetched successfully")
            print(f"   Shape: {stock_df.shape}")
            print(f"   Date range: {stock_df.index[0]} to {stock_df.index[-1]}")
            print(f"   Columns: {list(stock_df.columns)}")
            return stock_df
        else:
            print("❌ No stock data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None

def test_volatility_forecasting(stock_df):
    """Test volatility forecasting with the LSTM model."""
    print("\n🧪 Testing volatility forecasting...")
    
    if stock_df is None:
        print("❌ No stock data available for testing")
        return False
    
    try:
        # Create LSTM forecaster
        forecaster = create_lstm_volatility_forecaster(
            model_path="volatility_models/volatility_lstm_model.h5",
            scaler_path="volatility_models/volatility_scaler.pkl",
            memory_window=60
        )
        
        if not forecaster.is_model_loaded:
            print("❌ LSTM model not loaded")
            return False
        
        # Test forecasting
        symbol = "AAPL"
        test_timestamp = stock_df.index[-1]  # Use the last timestamp
        
        print(f"   Making volatility forecast for {test_timestamp}")
        forecast = forecaster.forecast_volatility(stock_df, test_timestamp, symbol)
        
        print(f"✅ Volatility forecast: {forecast:.4f} ({forecast*100:.2f}%)")
        
        # Test hourly realized volatility calculation
        print("\n🧪 Testing hourly realized volatility calculation...")
        rv_hourly = forecaster.prepare_hourly_realized_volatility(stock_df, symbol)
        
        if not rv_hourly.empty:
            print(f"✅ Hourly realized volatility calculated")
            print(f"   Shape: {rv_hourly.shape}")
            print(f"   Date range: {rv_hourly.index[0]} to {rv_hourly.index[-1]}")
            print(f"   Mean RV: {rv_hourly.mean():.4f}")
            print(f"   Std RV: {rv_hourly.std():.4f}")
        else:
            print("❌ No hourly realized volatility data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in volatility forecasting: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 LSTM Volatility Model Integration Test")
    print("=" * 50)
    
    # Test 1: Model loading
    model_loaded = test_lstm_model_loading()
    
    # Test 2: Data handler
    stock_df = test_data_handler()
    
    # Test 3: Volatility forecasting
    if model_loaded and stock_df is not None:
        forecasting_works = test_volatility_forecasting(stock_df)
    else:
        forecasting_works = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Model loading: {'✅ PASS' if model_loaded else '❌ FAIL'}")
    print(f"   Data handler: {'✅ PASS' if stock_df is not None else '❌ FAIL'}")
    print(f"   Forecasting: {'✅ PASS' if forecasting_works else '❌ FAIL'}")
    
    if all([model_loaded, stock_df is not None, forecasting_works]):
        print("\n🎉 All tests passed! LSTM integration is ready.")
        print("\n📋 Next steps:")
        print("   1. Run the main backtesting script: python backtesting_module/main.py")
        print("   2. Choose option 1 for LSTM volatility forecasting")
        print("   3. Enter your parameters (stock symbol, dates, etc.)")
        print("   4. The system will use your LSTM model for volatility predictions")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure volatility_lstm_model.h5 and volatility_scaler.pkl are in volatility_models/")
        print("   2. Check that TensorFlow is installed: pip install tensorflow")
        print("   3. Verify your API keys are correct")
        print("   4. Check your internet connection for data fetching")

if __name__ == "__main__":
    main()
