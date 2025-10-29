import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handler import DataHandler
from greeks_engine import GreeksEngine

# Check QuantLib availability
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
    print(f"✅ QuantLib is available - Version: {ql.__version__}")
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("❌ QuantLib is NOT available - Install with: pip install QuantLib")


class TestImpliedVolatilityQuantLib(unittest.TestCase):
    """Test suite for implied volatility calculations using QuantLib."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Import API keys from config
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
        self.alpaca_key = ALPACA_API_KEY
        self.alpaca_secret = ALPACA_SECRET_KEY
        self.polygon_key = POLYGON_API_KEY
   
        # Initialize handlers
        self.data_handler = DataHandler(
            alpaca_api_key=self.alpaca_key,
            alpaca_secret=self.alpaca_secret,
            polygon_key=self.polygon_key
        )
        self.greeks_engine = GreeksEngine()
        
        # Test data
        self.test_option_ticker = "AAPL240322C00185000"  # AAPL call, March 22, 2024, $185 strike
        self.test_underlying = "AAPL"
        self.test_strike = 185.0
        self.test_expiration = "2024-03-22"
        
    def test_quantlib_call_pricing(self):
        """Test QuantLib call option pricing."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Set up QuantLib date
        today = ql.Date().todaysDate()
        maturity = ql.Date(22, 3, 2024)  # March 22, 2024
        
        # Set up option parameters
        spot_price = 100.0
        strike_price = 100.0
        risk_free_rate = 0.05
        volatility = 0.2
        dividend_rate = 0.0
        
        # Create QuantLib objects
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), volatility, ql.Actual365Fixed()))
        
        # Create option
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
        exercise = ql.EuropeanExercise(maturity)
        european_option = ql.VanillaOption(payoff, exercise)
        
        # Create pricing engine
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        # Calculate price
        call_price = european_option.NPV()
        
        # Expected price should be around 10.45 for these parameters
        self.assertGreater(call_price, 0)
        self.assertLess(call_price, 20)
        
    def test_quantlib_put_pricing(self):
        """Test QuantLib put option pricing."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Set up QuantLib date
        today = ql.Date().todaysDate()
        maturity = ql.Date(22, 3, 2024)  # March 22, 2024
        
        # Set up option parameters
        spot_price = 100.0
        strike_price = 100.0
        risk_free_rate = 0.05
        volatility = 0.2
        dividend_rate = 0.0
        
        # Create QuantLib objects
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), volatility, ql.Actual365Fixed()))
        
        # Create option
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike_price)
        exercise = ql.EuropeanExercise(maturity)
        european_option = ql.VanillaOption(payoff, exercise)
        
        # Create pricing engine
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        # Calculate price
        put_price = european_option.NPV()
        
        # Expected price should be around 5.57 for these parameters
        self.assertGreater(put_price, 0)
        self.assertLess(put_price, 15)
        
    def test_quantlib_implied_volatility_calculation(self):
        """Test QuantLib implied volatility calculation."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Set up QuantLib date
        today = ql.Date().todaysDate()
        maturity = ql.Date(22, 3, 2024)  # March 22, 2024
        
        # Set up option parameters
        spot_price = 100.0
        strike_price = 100.0
        risk_free_rate = 0.05
        true_volatility = 0.2
        dividend_rate = 0.0
        
        # Create QuantLib objects
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), true_volatility, ql.Actual365Fixed()))
        
        # Create option
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
        exercise = ql.EuropeanExercise(maturity)
        european_option = ql.VanillaOption(payoff, exercise)
        
        # Create pricing engine
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        # Calculate option price
        option_price = european_option.NPV()
        
        # Calculate implied volatility using QuantLib
        calculated_iv = european_option.impliedVolatility(option_price, bsm_process)
        
        # Should be close to true volatility
        self.assertAlmostEqual(calculated_iv, true_volatility, places=3)
        
    def test_quantlib_implied_volatility_edge_cases(self):
        """Test edge cases for QuantLib implied volatility calculation."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Set up QuantLib date
        today = ql.Date().todaysDate()
        maturity = ql.Date(22, 3, 2024)  # March 22, 2024
        
        # Set up option parameters
        spot_price = 100.0
        strike_price = 100.0
        risk_free_rate = 0.05
        dividend_rate = 0.0
        
        # Create QuantLib objects
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        
        # Test with zero time to expiration
        today = maturity
        volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), 0.2, ql.Actual365Fixed()))
        
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
        exercise = ql.EuropeanExercise(maturity)
        european_option = ql.VanillaOption(payoff, exercise)
        
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        # This should raise an exception for zero time to expiration
        with self.assertRaises(Exception):
            european_option.impliedVolatility(5.0, bsm_process)
            
    def test_quantlib_implied_volatility_series(self):
        """Test QuantLib implied volatility series calculation."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Create mock data
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        
        # Mock underlying prices (slightly increasing)
        underlying_prices = pd.Series(
            [100 + i for i in range(len(dates))],
            index=dates,
            name='underlying_price'
        )
        
        # Mock option prices (decreasing due to time decay)
        option_prices = pd.Series(
            [10 - i*0.5 for i in range(len(dates))],
            index=dates,
            name='option_price'
        )
        
        # Calculate IV series using QuantLib
        iv_series = self._calculate_quantlib_iv_series(
            option_prices=option_prices,
            underlying_prices=underlying_prices,
            strike_price=100.0,
            expiration_date="2024-03-22",
            option_type="call"
        )
        
        # Check that we get a valid series
        self.assertIsInstance(iv_series, pd.Series)
        self.assertGreater(len(iv_series), 0)
        self.assertTrue(all(iv_series > 0))  # All IVs should be positive
        
        # Check that IVs are reasonable (between 0.01 and 5.0)
        self.assertTrue(all((iv_series >= 0.01) & (iv_series <= 5.0)))
        
    def _calculate_quantlib_iv_series(self, option_prices, underlying_prices, strike_price, expiration_date, option_type="call"):
        """Helper method to calculate IV series using QuantLib."""
        if not QUANTLIB_AVAILABLE:
            return pd.Series()
            
        # Parse expiration date
        exp_date = pd.to_datetime(expiration_date)
        maturity = ql.Date(exp_date.day, exp_date.month, exp_date.year)
        
        # Set up QuantLib parameters
        risk_free_rate = 0.05
        dividend_rate = 0.0
        
        # Create yield curves
        today = ql.Date().todaysDate()
        riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
        
        # Align data
        aligned_data = pd.DataFrame({
            'option_price': option_prices,
            'underlying_price': underlying_prices
        }).dropna()
        
        iv_values = []
        iv_dates = []
        
        for date, row in aligned_data.iterrows():
            try:
                # Set up option parameters for this date
                spot_price = row['underlying_price']
                option_price = row['option_price']
                
                # Create QuantLib objects
                spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
                volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), 0.2, ql.Actual365Fixed()))
                
                # Create option
                payoff_type = ql.Option.Call if option_type.lower() == "call" else ql.Option.Put
                payoff = ql.PlainVanillaPayoff(payoff_type, strike_price)
                exercise = ql.EuropeanExercise(maturity)
                european_option = ql.VanillaOption(payoff, exercise)
                
                # Create pricing engine
                bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
                european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
                
                # Calculate implied volatility
                iv = european_option.impliedVolatility(option_price, bsm_process)
                
                iv_values.append(iv)
                iv_dates.append(date)
                
            except Exception as e:
                # Skip this point if IV calculation fails
                continue
        
        return pd.Series(iv_values, index=iv_dates, name='implied_volatility')
        
    def test_quantlib_vs_binomial_tree_comparison(self):
        """Compare QuantLib IV calculation with our binomial tree implementation."""
        if not QUANTLIB_AVAILABLE:
            self.skipTest("QuantLib is not available")
            
        # Test parameters
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        true_sigma = 0.2
        
        # Calculate option price using QuantLib
        today = ql.Date().todaysDate()
        maturity = ql.Date().todaysDate() + ql.Period(int(T * 365), ql.Days)
        
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
        dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
        volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), true_sigma, ql.Actual365Fixed()))
        
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(maturity)
        european_option = ql.VanillaOption(payoff, exercise)
        
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
        
        option_price = european_option.NPV()
        
        # Calculate IV using QuantLib
        quantlib_iv = european_option.impliedVolatility(option_price, bsm_process)
        
        # Calculate IV using our binomial tree
        binomial_iv = self.greeks_engine.calculate_implied_volatility(
            option_price=option_price,
            S=S, K=K, T=T, r=r,
            option_type="call"
        )
        
        # Both should be close to true volatility
        self.assertAlmostEqual(quantlib_iv, true_sigma, places=3)
        self.assertAlmostEqual(binomial_iv, true_sigma, places=2)  # Binomial tree is less precise
        
        # QuantLib and binomial tree should be reasonably close
        self.assertAlmostEqual(quantlib_iv, binomial_iv, places=2)


def run_quantlib_option_ticker_test():
    """Interactive function to test with a real option ticker using QuantLib."""
    if not QUANTLIB_AVAILABLE:
        print("❌ QuantLib is not available. Install with: pip install QuantLib")
        return
        
    print("=== QuantLib Option Ticker Implied Volatility Test ===")
    
    # Get user input
    option_ticker = input("Enter option ticker (e.g., AAPL240322C00185000): ").strip()
    if not option_ticker:
        option_ticker = "AAPL240322C00185000"  # Default
    
    start_date = input("Enter start date (YYYY-MM-DD) [2024-01-01]: ").strip()
    if not start_date:
        start_date = "2024-01-01"
    
    end_date = input("Enter end date (YYYY-MM-DD) [2024-01-31]: ").strip()
    if not end_date:
        end_date = "2024-01-31"
    
    # Parse option ticker to extract information
    try:
        import re
        date_match = re.search(r'(\d{6})', option_ticker)
        if not date_match:
            raise ValueError("Could not find date pattern (YYMMDD) in ticker")
        
        date_part = date_match.group(1)
        date_start = date_match.start()
        date_end = date_match.end()
        
        # Extract components
        underlying_with_prefix = option_ticker[:date_start]
        option_type = option_ticker[date_end]
        strike_part = option_ticker[date_end + 1:]
        
        # Remove "O:" prefix from underlying if present
        if underlying_with_prefix.startswith("O:"):
            underlying = underlying_with_prefix[2:]
        else:
            underlying = underlying_with_prefix
        
        # Validate option type
        if option_type not in ['C', 'P']:
            raise ValueError(f"Invalid option type: {option_type}. Must be 'C' or 'P'")
        
        # Validate strike part is numeric
        if not strike_part.isdigit():
            raise ValueError(f"Invalid strike price format: {strike_part}")
        
        print(f"Debug - Parsed components:")
        print(f"  Raw ticker: {option_ticker}")
        print(f"  Underlying: '{underlying}'")
        print(f"  Date part: '{date_part}'")
        print(f"  Option type: '{option_type}'")
        print(f"  Strike part: '{strike_part}'")
        
        # Convert date
        year = "20" + date_part[:2]
        month = date_part[2:4]
        day = date_part[4:6]
        expiration_date = f"{year}-{month}-{day}"
        
        # Convert strike
        strike_price = float(strike_part) / 1000.0
        
        print(f"\nParsed option details:")
        print(f"  Underlying: {underlying}")
        print(f"  Expiration: {expiration_date}")
        print(f"  Type: {option_type}all" if option_type == 'C' else f"  Type: {option_type}ut")
        print(f"  Strike: ${strike_price}")
        
        # Initialize handlers
        print("\nInitializing data handlers...")
        
        data_handler = DataHandler(
            alpaca_api_key=self.alpaca_key,
            alpaca_secret=self.alpaca_secret, 
            polygon_key=self.polygon_key
        )
        greeks_engine = GreeksEngine()
        
        print("QuantLib-based IV calculation ready!")
        print("Note: This implementation uses QuantLib for more accurate IV calculations")
        
    except Exception as e:
        print(f"Error parsing option ticker: {e}")
        print("Expected format: SYMBOL + YYMMDD + C/P + STRIKE")
        print("Example: AAPL240322C00185000")


if __name__ == "__main__":
    # Check QuantLib availability
    print("="*50)
    print("CHECKING QUANTLIB AVAILABILITY:")
    if QUANTLIB_AVAILABLE:
        print(f"✅ QuantLib is available - Version: {ql.__version__}")
    else:
        print("❌ QuantLib is NOT available")
        print("Install with: pip install QuantLib")
    print("="*50)
    
    # Run unit tests
    print("Running QuantLib unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run interactive test
    if QUANTLIB_AVAILABLE:
        print("\n" + "="*50)
        run_quantlib_option_ticker_test()
    else:
        print("\nSkipping interactive test - QuantLib not available") 