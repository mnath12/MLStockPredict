# import unittest
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os
# import sys

# # Add parent directory to path to import modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from data_handler import DataHandler
# from greeks_engine import GreeksEngine


# class TestImpliedVolatility(unittest.TestCase):
#     """Test suite for implied volatility calculations."""
    
#     def setUp(self):
#         """Set up test fixtures."""
#         # Mock API keys for testing (replace with real keys for integration tests)
#         self.alpaca_key = "PKCLL4TXCDLRN76OGRAB"
#         self.alpaca_secret = "ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T" 
#         self.polygon_key = "ejp0y0ppSQJzIX1W8qSoTIvL5ja3ctO9"
   
        
#         # Initialize handlers
#         self.data_handler = DataHandler(
#             alpaca_api_key=self.alpaca_key,
#             alpaca_secret=self.alpaca_secret,
#             polygon_key=self.polygon_key
#         )
#         self.greeks_engine = GreeksEngine()
        
#         # Test data
#         self.test_option_ticker = "AAPL240322C00185000"  # AAPL call, March 22, 2024, $185 strike
#         self.test_underlying = "AAPL"
#         self.test_strike = 185.0
#         self.test_expiration = "2024-03-22"
        
#     def test_black_scholes_call_pricing(self):
#         """Test Black-Scholes call option pricing."""
#         # Test case 1: ATM call
#         S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
#         call_price = self.greeks_engine.black_scholes_call(S, K, T, r, sigma)
        
#         # Expected price should be around 10.45 for these parameters
#         self.assertGreater(call_price, 0)
#         self.assertLess(call_price, 20)
        
#         # Test case 2: ITM call
#         S, K, T, r, sigma = 110.0, 100.0, 0.5, 0.05, 0.2
#         call_price = self.greeks_engine.black_scholes_call(S, K, T, r, sigma)
        
#         # ITM call should be more expensive
#         self.assertGreater(call_price, 10)
        
#         # Test case 3: OTM call
#         S, K, T, r, sigma = 90.0, 100.0, 0.5, 0.05, 0.2
#         call_price = self.greeks_engine.black_scholes_call(S, K, T, r, sigma)
        
#         # OTM call should be cheaper
#         self.assertLess(call_price, 5)
        
#         # Test case 4: Expired option
#         S, K, T, r, sigma = 100.0, 100.0, 0.0, 0.05, 0.2
#         call_price = self.greeks_engine.black_scholes_call(S, K, T, r, sigma)
        
#         # Expired ATM call should be worth max(S-K, 0) = 0
#         self.assertEqual(call_price, 0)
        
#     def test_black_scholes_put_pricing(self):
#         """Test Black-Scholes put option pricing."""
#         # Test case 1: ATM put
#         S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
#         put_price = self.greeks_engine.black_scholes_put(S, K, T, r, sigma)
        
#         # Expected price should be around 5.57 for these parameters
#         self.assertGreater(put_price, 0)
#         self.assertLess(put_price, 15)
        
#         # Test case 2: ITM put
#         S, K, T, r, sigma = 90.0, 100.0, 0.5, 0.05, 0.2
#         put_price = self.greeks_engine.black_scholes_put(S, K, T, r, sigma)
        
#         # ITM put should be more expensive
#         self.assertGreater(put_price, 9)
#         self.assertLess(put_price, 10)
        
        
#         # Test case 3: OTM put
#         S, K, T, r, sigma = 110.0, 100.0, 0.5, 0.05, 0.2
#         put_price = self.greeks_engine.black_scholes_put(S, K, T, r, sigma)
        
#         # OTM put should be cheaper
#         self.assertLess(put_price, 5)
        
#     def test_implied_volatility_calculation(self):
#         """Test implied volatility calculation."""
#         # Test case 1: ATM call
#         S, K, T, r = 100.0, 100.0, 1.0, 0.05
#         true_sigma = 0.2
#         option_price = self.greeks_engine.black_scholes_call(S, K, T, r, true_sigma)
        
#         # Calculate implied volatility
#         calculated_iv = self.greeks_engine.calculate_implied_volatility(
#             option_price=option_price,
#             S=S, K=K, T=T, r=r,
#             option_type="call"
#         )
        
#         # Should be close to true volatility
#         self.assertAlmostEqual(calculated_iv, true_sigma, places=3)
        
#         # Test case 2: ITM call
#         S, K, T, r = 110.0, 100.0, 0.5, 0.05
#         true_sigma = 0.25
#         option_price = self.greeks_engine.black_scholes_call(S, K, T, r, true_sigma)
        
#         calculated_iv = self.greeks_engine.calculate_implied_volatility(
#             option_price=option_price,
#             S=S, K=K, T=T, r=r,
#             option_type="call"
#         )
        
#         self.assertAlmostEqual(calculated_iv, true_sigma, places=3)
        
#         # Test case 3: Put option
#         S, K, T, r = 90.0, 100.0, 0.5, 0.05
#         true_sigma = 0.3
#         option_price = self.greeks_engine.black_scholes_put(S, K, T, r, true_sigma)
        
#         calculated_iv = self.greeks_engine.calculate_implied_volatility(
#             option_price=option_price,
#             S=S, K=K, T=T, r=r,
#             option_type="put"
#         )
        
#         self.assertAlmostEqual(calculated_iv, true_sigma, places=3)
        
#     def test_implied_volatility_edge_cases(self):
#         """Test edge cases for implied volatility calculation."""
#         # Test with zero time to expiration
#         with self.assertRaises(ValueError):
#             self.greeks_engine.calculate_implied_volatility(
#                 option_price=5.0, S=100.0, K=100.0, T=0.0, r=0.05, option_type="call"
#             )
        
#         # Test with negative option price
#         with self.assertRaises(ValueError):
#             self.greeks_engine.calculate_implied_volatility(
#                 option_price=-1.0, S=100.0, K=100.0, T=1.0, r=0.05, option_type="call"
#             )
        
#         # Test with invalid option type
#         with self.assertRaises(ValueError):
#             self.greeks_engine.calculate_implied_volatility(
#                 option_price=5.0, S=100.0, K=100.0, T=1.0, r=0.05, option_type="invalid"
#             )
        
#     def test_implied_volatility_series(self):
#         """Test implied volatility series calculation."""
#         # Create mock data
#         dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        
#         # Mock underlying prices (slightly increasing)
#         underlying_prices = pd.Series(
#             [100 + i for i in range(len(dates))],
#             index=dates,
#             name='underlying_price'
#         )
        
#         # Mock option prices (decreasing due to time decay)
#         option_prices = pd.Series(
#             [10 - i*0.5 for i in range(len(dates))],
#             index=dates,
#             name='option_price'
#         )
        
#         # Calculate IV series
#         iv_series = self.greeks_engine.calculate_implied_volatility_series(
#             option_prices=option_prices,
#             underlying_prices=underlying_prices,
#             strike_price=100.0,
#             expiration_date="2024-03-22",
#             option_type="call"
#         )
        
#         # Check that we get a valid series
#         self.assertIsInstance(iv_series, pd.Series)
#         self.assertGreater(len(iv_series), 0)
#         self.assertTrue(all(iv_series > 0))  # All IVs should be positive
        
#         # Check that IVs are reasonable (between 0.01 and 5.0)
#         self.assertTrue(all((iv_series >= 0.01) & (iv_series <= 5.0)))
        
#     def test_option_price_series_retrieval(self):
#         """Test option price series retrieval from data handler."""
#         # This test would require real API keys and internet connection
#         # For now, we'll test the method signature and error handling
        
#         # Test with invalid option ticker
#         with self.assertRaises(Exception):
#             # This should fail without real API keys
#             self.data_handler.get_option_price_series(
#                 option_ticker="INVALID_TICKER",
#                 start_date="2024-01-01",
#                 end_date="2024-01-10"
#             )
    
#     def test_integration_workflow(self):
#         """Test the complete workflow from option ticker to implied volatility."""
#         # This is a demonstration of how the workflow would work
#         # In practice, you would need real API keys and data
        
#         # Step 1: Get option price series
#         # option_prices = self.data_handler.get_option_price_series(
#         #     option_ticker=self.test_option_ticker,
#         #     start_date="2024-01-01",
#         #     end_date="2024-01-31",
#         #     price_type="mid"
#         # )
        
#         # Step 2: Get underlying stock prices
#         # stock_prices = self.data_handler.get_stock_bars(
#         #     ticker=self.test_underlying,
#         #     start_date="2024-01-01",
#         #     end_date="2024-01-31"
#         # )['close']
        
#         # Step 3: Calculate implied volatility series
#         # iv_series = self.greeks_engine.calculate_implied_volatility_series(
#         #     option_prices=option_prices,
#         #     underlying_prices=stock_prices,
#         #     strike_price=self.test_strike,
#         #     expiration_date=self.test_expiration,
#         #     option_type="call"
#         # )
        
#         # For now, we'll just test that the method exists and has correct signature
#         self.assertTrue(hasattr(self.data_handler, 'get_option_price_series'))
#         self.assertTrue(hasattr(self.greeks_engine, 'calculate_implied_volatility_series'))
        
#     def test_volatility_smile_characteristics(self):
#         """Test that implied volatility follows expected smile characteristics."""
#         # Create test data for different strikes
#         S = 100.0  # Current stock price
#         T = 0.5    # Time to expiration
#         r = 0.05   # Risk-free rate
        
#         strikes = [80, 90, 100, 110, 120]  # Different strike prices
#         ivs = []
        
#         for K in strikes:
#             # Use a simple volatility smile model
#             moneyness = K / S
#             sigma = 0.2 + 0.1 * (moneyness - 1.0)**2  # U-shaped smile
            
#             # Calculate option price
#             option_price = self.greeks_engine.black_scholes_call(S, K, T, r, sigma)
            
#             # Calculate implied volatility
#             iv = self.greeks_engine.calculate_implied_volatility(
#                 option_price=option_price,
#                 S=S, K=K, T=T, r=r,
#                 option_type="call"
#             )
#             ivs.append(iv)
        
#         # Check that ATM (K=100) has lower IV than OTM/ITM options
#         atm_iv = ivs[2]  # K=100
#         otm_iv = ivs[4]  # K=120
#         itm_iv = ivs[0]  # K=80
        
#         # In a typical smile, ATM should have lower IV than OTM/ITM
#         # (This depends on the smile model used)
#         self.assertGreater(otm_iv, atm_iv * 0.8)  # Allow some tolerance
#         self.assertGreater(itm_iv, atm_iv * 0.8)  # Allow some tolerance


# def run_option_ticker_test():
#     """Interactive function to test with a real option ticker."""
#     print("=== Option Ticker Implied Volatility Test ===")
    
#     # Get user input
#     option_ticker = input("Enter option ticker (e.g., AAPL240322C00185000): ").strip()
#     if not option_ticker:
#         option_ticker = "AAPL240322C00185000"  # Default
    
#     start_date = input("Enter start date (YYYY-MM-DD) [2024-01-01]: ").strip()
#     if not start_date:
#         start_date = "2024-01-01"
    
#     end_date = input("Enter end date (YYYY-MM-DD) [2024-01-31]: ").strip()
#     if not end_date:
#         end_date = "2024-01-31"
    
#     # Parse option ticker to extract information
#     # Format: SYMBOL + YYMMDD + C/P + STRIKE
#     # Example: AAPL240322C00185000 = AAPL call, March 22, 2024, $185.000 strike
    
#     try:
#         # Extract components - handle variable length tickers
#         # Format: SYMBOL + YYMMDD + C/P + STRIKE
#         # Example: AAPL240322C00185000 = AAPL call, March 22, 2024, $185.000 strike
        
#         # Find the date part (YYMMDD) which is always 6 digits
#         import re
#         date_match = re.search(r'(\d{6})', option_ticker)
#         if not date_match:
#             raise ValueError("Could not find date pattern (YYMMDD) in ticker")
        
#         date_part = date_match.group(1)
#         date_start = date_match.start()
#         date_end = date_match.end()
        
#         # Extract components
#         underlying_with_prefix = option_ticker[:date_start]  # Everything before date
#         option_type = option_ticker[date_end]  # Character after date
#         strike_part = option_ticker[date_end + 1:]  # Everything after option type
        
#         # Remove "O:" prefix from underlying if present
#         if underlying_with_prefix.startswith("O:"):
#             underlying = underlying_with_prefix[2:]  # Remove "O:" prefix
#         else:
#             underlying = underlying_with_prefix
        
#         # Validate option type
#         if option_type not in ['C', 'P']:
#             raise ValueError(f"Invalid option type: {option_type}. Must be 'C' or 'P'")
        
#         # Validate strike part is numeric
#         if not strike_part.isdigit():
#             raise ValueError(f"Invalid strike price format: {strike_part}")
        
#         print(f"Debug - Parsed components:")
#         print(f"  Raw ticker: {option_ticker}")
#         print(f"  Underlying: '{underlying}'")
#         print(f"  Date part: '{date_part}'")
#         print(f"  Option type: '{option_type}'")
#         print(f"  Strike part: '{strike_part}'")
        
#         # Convert date
#         year = "20" + date_part[:2]
#         month = date_part[2:4]
#         day = date_part[4:6]
#         expiration_date = f"{year}-{month}-{day}"
        
#         # Convert strike
#         strike_price = float(strike_part) / 1000.0
        
#         print(f"\nParsed option details:")
#         print(f"  Underlying: {underlying}")
#         print(f"  Expiration: {expiration_date}")
#         print(f"  Type: {option_type}all" if option_type == 'C' else f"  Type: {option_type}ut")
#         print(f"  Strike: ${strike_price}")
        
#         # Initialize handlers (you'll need real API keys)
#         print("\nNote: This requires real API keys to fetch data.")
#         print("Please set your API keys in the DataHandler initialization.")
        
#         # Initialize handlers with real API keys
#         print("\nInitializing data handlers...")
        
#         data_handler = DataHandler(
#             alpaca_api_key="PKCLL4TXCDLRN76OGRAB",
#             alpaca_secret="ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T", 
#             polygon_key="ejp0y0ppSQJzIX1W8qSoTIvL5ja3ctO9"
#         )
#         greeks_engine = GreeksEngine()
        
#         # Comment out the backtest part - focus on ticker and IV testing only
#         """
#         try:
#             print(f"Fetching option prices for {option_ticker}...")
#             # Get option prices
#             option_prices = data_handler.get_option_price_series(
#                 option_ticker=option_ticker,
#                 start_date=start_date,
#                 end_date=end_date,
#                 price_type="mid"
#             )
#             print(f"Retrieved {len(option_prices)} option price points")
#             print(f"\nOption Prices:")
#             print(option_prices)
            
#             print(f"Fetching underlying stock prices for {underlying}...")
#             # Get underlying prices
#             stock_prices = data_handler.get_stock_bars(
#                 ticker=underlying,
#                 start_date=start_date,
#                 end_date=end_date
#             )['close']
#             print(f"Retrieved {len(stock_prices)} stock price points")
#             print(f"\nStock Prices:")
#             print(stock_prices)
            
#             print("Calculating implied volatility series...")
#             # Calculate IV series
#             iv_series = greeks_engine.calculate_implied_volatility_series(
#                 option_prices=option_prices,
#                 underlying_prices=stock_prices,
#                 strike_price=strike_price,
#                 expiration_date=expiration_date,
#                 option_type="call" if option_type == 'C' else "put"
#             )
            
#             print(f"\nImplied Volatility Series:")
#             print(iv_series)
            
#             # Summary statistics
#             print(f"\nIV Summary:")
#             print(f"  Mean: {iv_series.mean():.4f}")
#             print(f"  Std:  {iv_series.std():.4f}")
#             print(f"  Min:  {iv_series.min():.4f}")
#             print(f"  Max:  {iv_series.max():.4f}")
#         """
        
#         print("Backtest part commented out - focusing on ticker and IV testing only")
        
#         # Comment out the exception handling too
#         """
#         except Exception as e:
#             print(f"Error during data retrieval or calculation: {e}")
#             print("This might be due to:")
#             print("1. Invalid API keys")
#             print("2. No data available for the specified dates")
#             print("3. Option ticker not found")
#             print("4. Network connectivity issues")
#         """
        
#         # Test with last data point - commented out for now
#         """
#         print(f"\n" + "="*50)
#         print("TESTING LAST DATA POINT:")
        
#         try:
#             # Get the last option price
#             last_option_price = option_prices.iloc[-1]
#             last_option_date = option_prices.index[-1]
            
#             # Get stock price closest to the option date
#             stock_price_closest = stock_prices.loc[stock_prices.index <= last_option_date].iloc[-1]
#             stock_date_closest = stock_prices.loc[stock_prices.index <= last_option_date].index[-1]
            
#             print(f"Last option price: ${last_option_price:.2f} on {last_option_date}")
#             print(f"Closest stock price: ${stock_price_closest:.2f} on {stock_date_closest}")
            
#             # Calculate time to expiration
#             exp_dt = pd.to_datetime(expiration_date).tz_localize(last_option_date.tz)
#             ttm = (exp_dt - last_option_date).total_seconds() / (365.25 * 24 * 3600)
#             print(f"Time to expiration: {ttm:.4f} years")
            
#             # Calculate intrinsic value
#             intrinsic_value = max(0, stock_price_closest - strike_price) if option_type == 'C' else max(0, strike_price - stock_price_closest)
#             print(f"Intrinsic value: ${intrinsic_value:.2f}")
#             print(f"Time value: ${last_option_price - intrinsic_value:.2f}")
            
#             # Calculate implied volatility
#             iv = greeks_engine.calculate_implied_volatility(
#                 option_price=last_option_price,
#                 S=stock_price_closest,
#                 K=strike_price,
#                 T=ttm,
#                 r=0.02,
#                 option_type="call" if option_type == 'C' else "put"
#             )
#             print(f"Implied Volatility: {iv:.4f}")
            
#         except Exception as e:
#             print(f"Error in single point test: {e}")
#             import traceback
#             traceback.print_exc()
#         """
        
#         # Test multiple options from Polygon data
#         print(f"\n" + "="*50)
#         print("TESTING MULTIPLE OPTIONS FROM POLYGON DATA:")
#         print("Using Binomial Tree Model for IV calculation")
        
#         # Configurable tolerance for IV comparison
#         iv_tolerance = 0.15  # Adjust this value to change acceptable difference
#         print(f"IV Tolerance: ±{iv_tolerance}")
        
#         # Test cases from Polygon snapshot with actual option prices
#         test_cases = [
#             {
#                 "ticker": "O:AAPL250801C00110000",
#                 "strike": 110.0,
#                 "polygon_iv": None,  # No IV in data
#                 "option_price": 104.97,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00120000",
#                 "strike": 120.0,
#                 "polygon_iv": None,  # No IV in data
#                 "option_price": 95.0,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00125000",
#                 "strike": 125.0,
#                 "polygon_iv": None,  # No IV in data
#                 "option_price": 89.78,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00130000",
#                 "strike": 130.0,
#                 "polygon_iv": 1.6626,
#                 "option_price": 80.36,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00135000", 
#                 "strike": 135.0,
#                 "polygon_iv": 2.0220,
#                 "option_price": None,  # No price in data
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00140000",
#                 "strike": 140.0, 
#                 "polygon_iv": 1.6467,
#                 "option_price": 74.13,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00145000",
#                 "strike": 145.0,
#                 "polygon_iv": 1.7744,
#                 "option_price": 57.3,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00150000",
#                 "strike": 150.0,
#                 "polygon_iv": None,  # No IV in data
#                 "option_price": 64.0,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00155000",
#                 "strike": 155.0,
#                 "polygon_iv": None,  # No IV in data
#                 "option_price": 59.54,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             },
#             {
#                 "ticker": "O:AAPL250801C00160000",
#                 "strike": 160.0,
#                 "polygon_iv": 0.9744,
#                 "option_price": 54.15,
#                 "stock_price": 208.01,
#                 "ttm": 0.0461
#             }
#         ]
        
#         print(f"\n{'Ticker':<25} {'Strike':<8} {'Polygon IV':<12} {'Our IV':<12} {'Diff':<10} {'Status'}")
#         print("-" * 80)
        
#         successful_tests = 0
#         total_comparable_tests = 0
        
#         for case in test_cases:
#             try:
#                 # Skip if no option price available
#                 if case["option_price"] is None:
#                     print(f"{case['ticker']:<25} ${case['strike']:<7} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'SKIP'}")
#                     continue
                
#                 # Calculate our IV
#                 our_iv = greeks_engine.calculate_implied_volatility(
#                     option_price=case["option_price"],
#                     S=case["stock_price"],
#                     K=case["strike"],
#                     T=case["ttm"],
#                     r=0.0442,
#                     option_type="call"
#                 )
                
#                 # Check if we have Polygon IV to compare
#                 if case["polygon_iv"] is not None:
#                     # Calculate difference
#                     diff = abs(our_iv - case["polygon_iv"])
#                     status = "✅" if diff < iv_tolerance else "❌"
#                     if diff < iv_tolerance:
#                         successful_tests += 1
#                     total_comparable_tests += 1
                    
#                     print(f"{case['ticker']:<25} ${case['strike']:<7} {case['polygon_iv']:<12.4f} {our_iv:<12.4f} {diff:<10.4f} {status}")
#                 else:
#                     # No Polygon IV to compare, just show our calculation
#                     print(f"{case['ticker']:<25} ${case['strike']:<7} {'N/A':<12} {our_iv:<12.4f} {'N/A':<10} {'CALC'}")
                
#             except Exception as e:
#                 print(f"{case['ticker']:<25} ${case['strike']:<7} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} ❌")
#                 print(f"  Error: {e}")
        
#         # Summary of tolerance results
#         if total_comparable_tests > 0:
#             print(f"\nTolerance Summary: {successful_tests}/{total_comparable_tests} tests passed (±{iv_tolerance})")
#         else:
#             print(f"\nNo comparable tests - showing calculated IVs only")
        
#         if total_comparable_tests > 0:
#             if successful_tests == total_comparable_tests:
#                 print("🎉 All tests passed!")
#             elif successful_tests >= total_comparable_tests * 0.8:
#                 print("✅ Most tests passed - IV calculation is working well")
#             else:
#                 print("⚠️  Many tests failed - may need to adjust model parameters")
#         else:
#             print("📊 Showing calculated IVs for all available options")
        
#         print(f"\n" + "="*50)
#         print("VOLATILITY SMILE ANALYSIS:")
        
#         # Extract strikes and IVs for smile analysis (only those with IV data)
#         strikes_with_iv = [case["strike"] for case in test_cases if case["polygon_iv"] is not None]
#         polygon_ivs_with_data = [case["polygon_iv"] for case in test_cases if case["polygon_iv"] is not None]
        
#         print(f"Strikes with IV data: {strikes_with_iv}")
#         print(f"Polygon IVs: {[f'{iv:.4f}' for iv in polygon_ivs_with_data]}")
        
#         if strikes_with_iv:
#             # Find ATM strike (closest to stock price)
#             atm_strike = min(strikes_with_iv, key=lambda x: abs(x - 208.01))
#             atm_index = strikes_with_iv.index(atm_strike)
#             atm_iv = polygon_ivs_with_data[atm_index]
            
#             print(f"ATM Strike: ${atm_strike} (closest to $208.01)")
#             print(f"ATM IV: {atm_iv:.4f}")
            
#             # Check smile characteristics
#             print(f"\nSmile Analysis:")
#             for i, (strike, iv) in enumerate(zip(strikes_with_iv, polygon_ivs_with_data)):
#                 moneyness = strike / 208.01
#                 if moneyness < 0.9:
#                     region = "Deep ITM"
#                 elif moneyness < 1.0:
#                     region = "ITM"
#                 elif moneyness < 1.1:
#                     region = "ATM"
#                 elif moneyness < 1.2:
#                     region = "OTM"
#                 else:
#                     region = "Deep OTM"
                
#                 print(f"  ${strike}: {iv:.4f} ({region}, moneyness: {moneyness:.3f})")
#         else:
#             print("No IV data available for smile analysis")
        
        
#     except Exception as e:
#         print(f"Error parsing option ticker: {e}")
#         print("Expected format: SYMBOL + YYMMDD + C/P + STRIKE")
#         print("Example: AAPL240322C00185000")


# if __name__ == "__main__":
#     # Run unit tests
#     print("Running unit tests...")
#     unittest.main(argv=[''], exit=False, verbosity=2)
    
#     # Run interactive test
#     print("\n" + "="*50)
#     run_option_ticker_test()

# test_implied_volatility.py
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..greeks_engine import GreeksEngine
from ..data_handler import DataHandler

class TestImpliedVolatility(unittest.TestCase):
    """Test suite for implied volatility calculations."""
    def setUp(self):
        self.data_handler = DataHandler(
            alpaca_api_key=self.alpaca_key,
            alpaca_secret=self.alpaca_secret,
            polygon_key=self.polygon_key
        )
        self.engine = GreeksEngine()

    def test_black_scholes_call_pricing(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        price = self.engine.black_scholes_call(S, K, T, r, sigma)
        self.assertGreater(price, 0)
        self.assertLess(price, 20)

    def test_black_scholes_put_pricing(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        price = self.engine.black_scholes_put(S, K, T, r, sigma)
        self.assertGreater(price, 0)
        self.assertLess(price, 15)

    def test_implied_volatility_calculation(self):
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        true_sigma = 0.2
        option_price = self.engine.black_scholes_call(S, K, T, r, true_sigma)
        calculated = self.engine.calculate_implied_volatility(option_price, S, K, T, r, 0.0, 'call')
        self.assertAlmostEqual(calculated, true_sigma, places=3)

    def test_implied_volatility_edge_cases(self):
        with self.assertRaises(ValueError):
            self.engine.calculate_implied_volatility(5.0, 100.0, 100.0, 0.0)
        with self.assertRaises(ValueError):
            self.engine.calculate_implied_volatility(-1.0, 100.0, 100.0, 1.0)
        with self.assertRaises(ValueError):
            self.engine.calculate_implied_volatility(5.0, 100.0, 100.0, 1.0, option_type='invalid')

    def test_implied_volatility_series(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        underlying = pd.Series([100 + i for i in range(5)], index=dates)
        option = pd.Series([10 - i*0.5 for i in range(5)], index=dates)
        iv_series = self.engine.calculate_implied_volatility_series(
            option, underlying, 100.0, '2024-03-22', 0.05, 'call'
        )
        self.assertIsInstance(iv_series, pd.Series)
        self.assertTrue((iv_series > 0).all())

    def test_option_price_series_retrieval(self):
        with self.assertRaises(Exception):
            self.data_handler.get_option_price_series('INVALID', '2024-01-01', '2024-01-05')

if __name__ == '__main__':
    unittest.main()
