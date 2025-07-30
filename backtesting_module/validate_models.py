#!/usr/bin/env python3
"""
Simple script to validate volatility models.

This script provides an easy way to test volatility forecasting models
against realized volatility before using them in the backtesting system.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from volatility_model_validator import validate_all_models, VolatilityModelValidator
from data_handler import DataHandler

def main():
    """Main validation script."""
    print("🔍 VOLATILITY MODEL VALIDATOR")
    print("=" * 50)
    print("This script will validate your volatility forecasting models")
    print("against realized volatility to ensure they are working correctly.")
    print()
    
    # Get user input
    symbol = input("Enter stock symbol to test (e.g., TSLA): ").strip().upper()
    if not symbol:
        symbol = "TSLA"
    
    # Get validation period
    print("\nValidation Period Options:")
    print("1. Last 6 months (recommended for quick testing)")
    print("2. Last 1 year")
    print("3. Last 2 years")
    print("4. Custom period")
    
    period_choice = input("Choose validation period (1-4) [default: 1]: ").strip()
    if not period_choice:
        period_choice = "1"
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    if period_choice == "1":
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        print("Using last 6 months for validation")
    elif period_choice == "2":
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        print("Using last 1 year for validation")
    elif period_choice == "3":
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        print("Using last 2 years for validation")
    elif period_choice == "4":
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            print("Using default: last 6 months")
    else:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        print("Using last 6 months for validation")
    
    print(f"Validation period: {start_date} to {end_date}")
    
    # Add model configuration options
    print("\n🔧 Model Configuration Options:")
    print("1. Use default settings (memory=60, hourly RV)")
    print("2. Custom memory window")
    print("3. Use daily instead of hourly RV")
    
    config_choice = input("Choose configuration (1-3) [default: 1]: ").strip()
    if not config_choice:
        config_choice = "1"
    
    custom_memory = None
    use_hourly = True
    
    if config_choice == "2":
        memory_input = input("Enter memory window (lookback periods) [default: 60]: ").strip()
        custom_memory = int(memory_input) if memory_input.isdigit() else 60
        print(f"Using custom memory window: {custom_memory}")
    elif config_choice == "3":
        use_hourly = False
        print("Using daily realized volatility")
    else:
        print("Using default settings")
    
    # Check if we have API keys
    print("\nChecking API configuration...")
    alpaca_key = input("Enter Alpaca API key (or leave blank): ").strip()
    alpaca_secret = input("Enter Alpaca API secret (or leave blank): ").strip()
    polygon_key = input("Enter Polygon API key (or leave blank): ").strip()
    
    # Initialize data handler
    data_handler = DataHandler()
    if alpaca_key and alpaca_secret:
        data_handler.setup_alpaca(alpaca_key, alpaca_secret)
    if polygon_key:
        data_handler.setup_polygon(polygon_key)
    
    print("\n🔍 Starting model validation...")
    print("=" * 50)
    
    try:
        # Validate all models
        results = validate_all_models(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            models_dir="../volatility_models",
            custom_memory=custom_memory,
            use_hourly=use_hourly
        )
        
        print("\n✅ Validation complete!")
        print("Check the generated validation_results_* folders for detailed reports.")
        
        # Summary
        print("\n📊 VALIDATION SUMMARY:")
        print("-" * 30)
        for model_name, result in results.items():
            if 'error' in result:
                print(f"❌ {model_name}: ERROR - {result['error']}")
            else:
                metrics = result.get('metrics', {})
                r2 = metrics.get('r2', 0)
                correlation = metrics.get('correlation', 0)
                directional = metrics.get('directional_accuracy', 0)
                print(f"✅ {model_name}: R²={r2:.3f}, Corr={correlation:.3f}, Dir={directional:.1%}")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install pandas numpy matplotlib seaborn scikit-learn scipy")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 