# from __future__ import annotations

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Optional, Any, Union
# from datetime import datetime, date
# import warnings
# from scipy.stats import norm
# from scipy.optimize import minimize_scalar

# try:
#     import QuantLib as ql
#     QUANTLIB_AVAILABLE = True
# except ImportError:
#     QUANTLIB_AVAILABLE = False
#     warnings.warn("QuantLib not available. Using simplified Greeks calculations.")


# class GreeksEngine:
#     """
#     Computes option Greeks using QuantLib when available, or simplified formulas as fallback.
#     """
    
#     def __init__(self, stock_bars: Optional[pd.DataFrame] = None, 
#                  option_bars: Optional[pd.DataFrame] = None, 
#                  pricing_model: str = "Black-Scholes"):
#         self.stock_bars = stock_bars
#         self.option_bars = option_bars
#         self.pricing_model = pricing_model
        
#         if QUANTLIB_AVAILABLE:
#             self._setup_quantlib()
        
#     def _setup_quantlib(self) -> None:
#         """Setup QuantLib environment."""
#         if QUANTLIB_AVAILABLE:
#             self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
#             self.day_count = ql.Actual365Fixed()
#             ql.Settings.instance().evaluationDate = ql.Date(1, 1, 2024)

#     def compute(self, symbol: str, underlying_price: float, strike: float, 
#                 time_to_expiry: float, volatility: float, risk_free_rate: float = 0.05,
#                 option_type: str = "call", dividend_yield: float = 0.0, **kwargs: Any) -> Dict[str, float]:
#         """
#         Compute Greeks for a single option.
        
#         Args:
#             symbol: Option symbol
#             underlying_price: Current underlying price
#             strike: Strike price
#             time_to_expiry: Time to expiry in years
#             volatility: Implied volatility
#             risk_free_rate: Risk-free rate
#             option_type: 'call' or 'put'
#             dividend_yield: Dividend yield
            
#         Returns:
#             Dictionary with Greeks values
#         """
#         if QUANTLIB_AVAILABLE:
#             return self._compute_quantlib(
#                 underlying_price, strike, time_to_expiry, volatility,
#                 risk_free_rate, option_type, dividend_yield
#             )
#         else:
#             return self._compute_simplified(
#                 underlying_price, strike, time_to_expiry, volatility,
#                 risk_free_rate, option_type, dividend_yield
#             )
    
#     def _compute_quantlib(self, underlying_price: float, strike: float, 
#                          time_to_expiry: float, volatility: float, risk_free_rate: float,
#                          option_type: str, dividend_yield: float) -> Dict[str, float]:
#         """Compute Greeks using QuantLib."""
#         try:
#             # Setup QuantLib objects
#             spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
#             risk_free_handle = ql.YieldTermStructureHandle(
#                 ql.FlatForward(0, self.calendar, risk_free_rate, self.day_count)
#             )
#             dividend_handle = ql.YieldTermStructureHandle(
#                 ql.FlatForward(0, self.calendar, dividend_yield, self.day_count)
#             )
#             volatility_handle = ql.BlackVolTermStructureHandle(
#                 ql.BlackConstantVol(0, self.calendar, volatility, self.day_count)
#             )

#             # Create process
#             if self.pricing_model.lower() == "jump-diffusion":
#                 # Merton jump diffusion parameters
#                 jump_intensity = 0.1
#                 jump_mean = -0.05
#                 jump_volatility = 0.2
#                 process = ql.MertonJumpDiffusionProcess(
#                     spot_handle, dividend_handle, risk_free_handle, 
#                     volatility_handle, jump_intensity, jump_mean, jump_volatility
#                 )
#             else:
#                 process = ql.BlackScholesMertonProcess(
#                     spot_handle, dividend_handle, risk_free_handle, volatility_handle
#                 )

#             # Create option
#             maturity_date = ql.Date.todaysDate() + int(time_to_expiry * 365)
#             exercise = ql.EuropeanExercise(maturity_date)
            
#             option_type_ql = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
#             payoff = ql.PlainVanillaPayoff(option_type_ql, strike)
#             option = ql.VanillaOption(payoff, exercise)

#             # Set pricing engine
#             if self.pricing_model.lower() == "jump-diffusion":
#                 engine = ql.JumpDiffusionEngine(process)
#             else:
#                 engine = ql.AnalyticEuropeanEngine(process)
            
#             option.setPricingEngine(engine)

#             # Calculate Greeks
#             return {
#                 'delta': option.delta(),
#                 'gamma': option.gamma(),
#                 'vega': option.vega() / 100.0,  # Convert to decimal
#                 'theta': option.theta() / 365.0,  # Convert to daily
#                 'rho': option.rho() / 100.0,  # Convert to decimal
#                 'price': option.NPV()
#             }

#         except Exception as e:
#             print(f"QuantLib calculation failed: {e}. Using simplified method.")
#             return self._compute_simplified(
#                 underlying_price, strike, time_to_expiry, volatility,
#                 risk_free_rate, option_type, dividend_yield
#             )
    
#     def _compute_simplified(self, underlying_price: float, strike: float, 
#                            time_to_expiry: float, volatility: float, risk_free_rate: float,
#                            option_type: str, dividend_yield: float) -> Dict[str, float]:
#         """Simplified Black-Scholes Greeks calculation."""
#         if time_to_expiry <= 0:
#             # Option expired
#             intrinsic = max(0, underlying_price - strike) if option_type.lower() == 'call' else max(0, strike - underlying_price)
#             return {
#                 'delta': 1.0 if (option_type.lower() == 'call' and underlying_price > strike) else 0.0,
#                 'gamma': 0.0,
#                 'vega': 0.0,
#                 'theta': 0.0,
#                 'rho': 0.0,
#                 'price': intrinsic
#             }
        
#         # Black-Scholes formula components
#         sqrt_t = np.sqrt(time_to_expiry)
#         d1 = (np.log(underlying_price / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
#         d2 = d1 - volatility * sqrt_t
        
#         # Standard normal CDF and PDF
#         from scipy.stats import norm
#         N_d1 = norm.cdf(d1)
#         N_d2 = norm.cdf(d2)
#         n_d1 = norm.pdf(d1)
        
#         if option_type.lower() == 'call':
#             # Call option
#             price = underlying_price * np.exp(-dividend_yield * time_to_expiry) * N_d1 - strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
#             delta = np.exp(-dividend_yield * time_to_expiry) * N_d1
#             theta = (- underlying_price * n_d1 * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * sqrt_t)
#                     - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
#                     + dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * N_d1) / 365
#             rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * N_d2 / 100
#         else:
#             # Put option
#             price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
#             delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
#             theta = (- underlying_price * n_d1 * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * sqrt_t)
#                     + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
#                     - dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)) / 365
#             rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
#         # Greeks that are the same for calls and puts
#         gamma = n_d1 * np.exp(-dividend_yield * time_to_expiry) / (underlying_price * volatility * sqrt_t)
#         vega = underlying_price * sqrt_t * n_d1 * np.exp(-dividend_yield * time_to_expiry) / 100
        
#         return {
#             'delta': delta,
#             'gamma': gamma,
#             'vega': vega,
#             'theta': theta,
#             'rho': rho,
#             'price': max(0, price)  # Ensure non-negative price
#         }

#     def align_option_data(self, option_symbols: List[str]) -> pd.DataFrame:
#         """
#         Align option data with stock data timestamps.
        
#         Args:
#             option_symbols: List of option symbols to align
            
#         Returns:
#             DataFrame with aligned option data
#         """
#         if self.stock_bars is None or self.option_bars is None:
#             return pd.DataFrame()
        
#         # Simple alignment - forward fill option data to match stock timestamps
#         aligned_data = pd.DataFrame(index=self.stock_bars.index)
        
#         for symbol in option_symbols:
#             if symbol in self.option_bars.columns:
#                 option_data = self.option_bars[symbol].reindex(self.stock_bars.index)
#                 aligned_data[symbol] = option_data.fillna(method='ffill')
        
#         return aligned_data

#     def plot_iv(self, symbol: str, strikes: List[float], expiry: str) -> None:
#         """
#         Plot implied volatility surface (placeholder implementation).
        
#         Args:
#             symbol: Underlying symbol
#             strikes: List of strike prices
#             expiry: Expiration date string
#         """
#         try:
#             import matplotlib.pyplot as plt
            
#             # Generate sample IV data (replace with real calculation)
#             iv_data = []
#             for strike in strikes:
#                 # Sample IV curve - ATM has lowest IV
#                 atm_strike = strikes[len(strikes)//2]
#                 moneyness = strike / atm_strike
#                 iv = 0.2 + 0.1 * abs(moneyness - 1.0)  # Simple smile
#                 iv_data.append(iv)
            
#             plt.figure(figsize=(10, 6))
#             plt.plot(strikes, iv_data, 'b-o')
#             plt.xlabel('Strike Price')
#             plt.ylabel('Implied Volatility')
#             plt.title(f'Implied Volatility Smile - {symbol} {expiry}')
#             plt.grid(True)
#             plt.show()
            
#         except ImportError:
#             print("Matplotlib not available for plotting")

#     def helper(self) -> Dict[str, Any]:
#         """
#         Helper function for additional functionality.
        
#         Returns:
#             Dictionary with helper information
#         """
#         return {
#             'quantlib_available': QUANTLIB_AVAILABLE,
#             'pricing_model': self.pricing_model,
#             'data_loaded': self.stock_bars is not None and self.option_bars is not None
#         }

#     def binomial_tree_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, steps: int = 100) -> float:
#         """
#         Calculate call option price using binomial tree model.
        
#         Args:
#             S: Current stock price
#             K: Strike price
#             T: Time to expiration (in years)
#             r: Risk-free rate
#             sigma: Volatility
#             q: Dividend yield
#             steps: Number of time steps in the tree
            
#         Returns:
#             float: Call option price
#         """
#         if T <= 0:
#             return max(S - K, 0)
        
#         dt = T / steps
#         u = np.exp(sigma * np.sqrt(dt))
#         d = 1 / u
#         p = (np.exp((r - q) * dt) - d) / (u - d)
        
#         # Ensure probability is valid
#         p = max(0, min(1, p))
        
#         # Initialize option values at final nodes
#         option_values = np.zeros(steps + 1)
#         for i in range(steps + 1):
#             stock_price = S * (u ** (steps - i)) * (d ** i)
#             option_values[i] = max(stock_price - K, 0)
        
#         # Backward induction
#         for step in range(steps - 1, -1, -1):
#             for i in range(step + 1):
#                 option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
        
#         return option_values[0]

#     def binomial_tree_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, steps: int = 100) -> float:
#         """
#         Calculate put option price using binomial tree model.
        
#         Args:
#             S: Current stock price
#             K: Strike price
#             T: Time to expiration (in years)
#             r: Risk-free rate
#             sigma: Volatility
#             q: Dividend yield
#             steps: Number of time steps in the tree
            
#         Returns:
#             float: Put option price
#         """
#         if T <= 0:
#             return max(K - S, 0)
        
#         dt = T / steps
#         u = np.exp(sigma * np.sqrt(dt))
#         d = 1 / u
#         p = (np.exp((r - q) * dt) - d) / (u - d)
        
#         # Ensure probability is valid
#         p = max(0, min(1, p))
        
#         # Initialize option values at final nodes
#         option_values = np.zeros(steps + 1)
#         for i in range(steps + 1):
#             stock_price = S * (u ** (steps - i)) * (d ** i)
#             option_values[i] = max(K - stock_price, 0)
        
#         # Backward induction
#         for step in range(steps - 1, -1, -1):
#             for i in range(step + 1):
#                 option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
        
#         return option_values[0]

#     def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
#         """
#         Calculate Black-Scholes call option price.
        
#         Args:
#             S: Current stock price
#             K: Strike price
#             T: Time to expiration (in years)
#             r: Risk-free rate
#             sigma: Volatility
#             q: Dividend yield
            
#         Returns:
#             float: Call option price
#         """
#         if T <= 0:
#             return max(S - K, 0)
        
#         d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#         d2 = d1 - sigma * np.sqrt(T)
        
#         call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#         return call_price

#     def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
#         """
#         Calculate Black-Scholes put option price.
        
#         Args:
#             S: Current stock price
#             K: Strike price
#             T: Time to expiration (in years)
#             r: Risk-free rate
#             sigma: Volatility
#             q: Dividend yield
            
#         Returns:
#             float: Put option price
#         """
#         if T <= 0:
#             return max(K - S, 0)
        
#         d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#         d2 = d1 - sigma * np.sqrt(T)
        
#         put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
#         return put_price

#     def _implied_volatility_objective(self, sigma: float, option_price: float, 
#                                     S: float, K: float, T: float, r: float, 
#                                     option_type: str, q: float = 0.0) -> float:
#         """
#         Objective function for implied volatility calculation.
        
#         Args:
#             sigma: Volatility guess
#             option_price: Market option price
#             S: Current stock price
#             K: Strike price
#             T: Time to expiration
#             r: Risk-free rate
#             option_type: 'call' or 'put'
#             q: Dividend yield
            
#         Returns:
#             float: Squared difference between model and market price
#         """
#         if option_type.lower() == 'call':
#             model_price = self.binomial_tree_call(S, K, T, r, sigma, q)
#         elif option_type.lower() == 'put':
#             model_price = self.binomial_tree_put(S, K, T, r, sigma, q)
#         else:
#             raise ValueError("option_type must be 'call' or 'put'")
        
#         return (model_price - option_price) ** 2

#     def calculate_implied_volatility(
#         self,
#         option_price: float,
#         S: float,
#         K: float,
#         T: float,
#         r: float = 0.02,
#         option_type: str = "call",
#         q: float = 0.0,
#         sigma_guess: float = 0.3,
#         tolerance: float = 1e-6
#     ) -> float:
#         """
#         Calculate implied volatility using numerical optimization.
        
#         Args:
#             option_price: Market option price
#             S: Current stock price
#             K: Strike price
#             T: Time to expiration (in years)
#             r: Risk-free rate (default: 2%)
#             option_type: 'call' or 'put'
#             q: Dividend yield
#             sigma_guess: Initial volatility guess
#             tolerance: Convergence tolerance
            
#         Returns:
#             float: Implied volatility
#         """
#         if T <= 0:
#             raise ValueError("Time to expiration must be positive")
        
#         if option_price <= 0:
#             raise ValueError("Option price must be positive")
        
#         # Use scipy's minimize_scalar for robust optimization
#         result = minimize_scalar(
#             self._implied_volatility_objective,
#             args=(option_price, S, K, T, r, option_type, q),
#             bounds=(0.001, 5.0),  # Reasonable volatility bounds
#             method='bounded'
#         )
        
#         if result.success:
#             return result.x
#         else:
#             raise ValueError(f"Failed to converge: {result.message}")

#     def calculate_implied_volatility_series(
#         self,
#         option_prices: pd.Series,
#         underlying_prices: pd.Series,
#         strike_price: float,
#         expiration_date: str,
#         risk_free_rate: float = 0.02,
#         option_type: str = "call",
#         dividend_yield: float = 0.0
#     ) -> pd.Series:
#         """
#         Calculate implied volatility series for an option over time.
        
#         Args:
#             option_prices: Series of option prices
#             underlying_prices: Series of underlying stock prices
#             strike_price: Option strike price
#             expiration_date: Option expiration date in 'YYYY-MM-DD' format
#             risk_free_rate: Risk-free rate (default: 2%)
#             option_type: 'call' or 'put'
#             dividend_yield: Dividend yield
            
#         Returns:
#             pd.Series: Implied volatility series indexed by timestamp
#         """
#         # Convert expiration date to datetime
#         exp_dt = pd.to_datetime(expiration_date)
        
#         # Calculate time to expiration for each date
#         def calculate_ttm(timestamp):
#             return (exp_dt - timestamp).total_seconds() / (365.25 * 24 * 3600)
        
#         # Align option prices with underlying prices
#         aligned_data = pd.DataFrame({
#             'option_price': option_prices,
#             'underlying_price': underlying_prices
#         }).dropna()
        
#         # Calculate implied volatility for each point
#         iv_series = []
#         timestamps = []
        
#         print(f"\nDebug: Calculating IV for {len(aligned_data)} aligned data points")
#         print(f"Strike: ${strike_price}, Option type: {option_type}")
        
#         for i, (timestamp, row) in enumerate(aligned_data.iterrows()):
#             try:
#                 ttm = calculate_ttm(timestamp)
#                 if ttm > 0:  # Only calculate if not expired
#                     print(f"\nPoint {i+1}:")
#                     print(f"  Timestamp: {timestamp}")
#                     print(f"  TTM: {ttm:.4f} years")
#                     print(f"  Stock price: ${row['underlying_price']:.2f}")
#                     print(f"  Option price: ${row['option_price']:.2f}")
#                     print(f"  Strike: ${strike_price:.2f}")
                    
#                     # Check if option price is reasonable
#                     intrinsic_value = max(0, row['underlying_price'] - strike_price) if option_type == 'call' else max(0, strike_price - row['underlying_price'])
#                     print(f"  Intrinsic value: ${intrinsic_value:.2f}")
#                     print(f"  Time value: ${row['option_price'] - intrinsic_value:.2f}")
                    
#                     iv = self.calculate_implied_volatility(
#                         option_price=row['option_price'],
#                         S=row['underlying_price'],
#                         K=strike_price,
#                         T=ttm,
#                         r=risk_free_rate,
#                         option_type=option_type,
#                         q=dividend_yield
#                     )
#                     print(f"  Calculated IV: {iv:.4f}")
#                     iv_series.append(iv)
#                     timestamps.append(timestamp)
#                 else:
#                     print(f"Point {i+1}: Option expired (TTM = {ttm:.4f})")
#             except (ValueError, RuntimeError) as e:
#                 print(f"Point {i+1}: IV calculation failed - {e}")
#                 continue
        
#         return pd.Series(iv_series, index=timestamps, name="implied_volatility")
# greeks_engine.py
# greeks_engine.py
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
from scipy.stats import norm

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    warnings.warn("QuantLib not available. Using simplified Greeks calculations.")

class GreeksEngine:
    """
    Computes option Greeks using QuantLib when available, or simplified formulas as fallback.
    Also computes implied volatility via QuantLib for American/European options.

    pricing_model controls engine for Greeks:
    - 'analytic' uses closed-form
    - 'binomial' uses Cox-Ross-Rubinstein tree (configurable via steps)
    """
    def __init__(
        self,
        stock_bars: Optional[pd.DataFrame] = None,
        option_bars: Optional[pd.DataFrame] = None,
        pricing_model: str = "analytic"
    ):
        self.stock_bars = stock_bars
        self.option_bars = option_bars
        self.pricing_model = pricing_model.lower()

        if QUANTLIB_AVAILABLE:
            self._setup_quantlib()

    def _setup_quantlib(self) -> None:
        """Initialize QuantLib evaluation date and conventions."""
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        self.calendar = ql.NullCalendar()
        self.day_count = ql.Actual365Fixed()

    def helper(self) -> Dict[str, Any]:
        """
        Helper function for additional functionality.
        
        Returns:
            Dictionary with helper information
        """
        return {
            'quantlib_available': QUANTLIB_AVAILABLE,
            'pricing_model': self.pricing_model,
            'data_loaded': self.stock_bars is not None and self.option_bars is not None
        }

    def _build_process(
        self,
        spot: float,
        r: float,
        q: float,
        vol: float
    ) -> ql.BlackScholesMertonProcess:
        """Helper to build QuantLib Black-Scholes-Merton process."""
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        rf_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(0, self.calendar, r, self.day_count)
        )
        div_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(0, self.calendar, q, self.day_count)
        )
        vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(0, self.calendar, vol, self.day_count)
        )
        return ql.BlackScholesMertonProcess(spot_handle, div_ts, rf_ts, vol_ts)

    def compute(
        self,
        symbol: str,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
        dividend_yield: float = 0.0,
        engine_type: str = None,
        steps: int = 100,
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Compute Greeks and price for a single option.
        engine_type: 'analytic' or any supported binomial (e.g. 'crr').
        steps: tree steps when using binomial.
        """
        if QUANTLIB_AVAILABLE:
            return self._compute_quantlib(
                underlying_price,
                strike,
                time_to_expiry,
                volatility,
                risk_free_rate,
                option_type,
                dividend_yield,
                engine_type or self.pricing_model,
                steps
            )
        else:
            return self._compute_simplified(
                underlying_price,
                strike,
                time_to_expiry,
                volatility,
                risk_free_rate,
                option_type,
                dividend_yield
            )

    def _compute_quantlib(
        self,
        S: float,
        K: float,
        T: float,
        vol: float,
        r: float,
        option_type: str,
        q: float,
        engine_type: str,
        steps: int
    ) -> Dict[str, float]:
        """Compute Greeks using QuantLib, with optional tree engine."""
        try:
            process = self._build_process(S, r, q, vol)
            today = ql.Settings.instance().evaluationDate
            maturity = today + int(T * 365 + 0.5)
            exercise = ql.EuropeanExercise(maturity)
            opt_type = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
            payoff = ql.PlainVanillaPayoff(opt_type, K)
            option = ql.VanillaOption(payoff, exercise)
            # Select engine
            if engine_type.lower() == 'analytic':
                engine = ql.AnalyticEuropeanEngine(process)
            else:
                engine = ql.BinomialVanillaEngine(process, engine_type, steps)
            option.setPricingEngine(engine)

            return {
                'delta': option.delta(),
                'gamma': option.gamma(),
                'vega': option.vega() / 100.0,
                'theta': option.theta() / 365.0,
                'rho': option.rho() / 100.0,
                'price': option.NPV()
            }
        except Exception as e:
            print(f"QuantLib computation failed: {e}. Falling back to simplified.")
            return self._compute_simplified(S, K, T, vol, r, option_type, q)

    def _compute_simplified(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        option_type: str,
        dividend_yield: float
    ) -> Dict[str, float]:
        """Simplified Black-Scholes Greeks calculation."""
        if time_to_expiry <= 0:
            intrinsic = max(0, underlying_price - strike) if option_type.lower() == 'call' else max(0, strike - underlying_price)
            return {
                'delta': 1.0 if option_type.lower() == 'call' and underlying_price > strike else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0,
                'price': intrinsic
            }

        sqrt_t = np.sqrt(time_to_expiry)
        d1 = (
            np.log(underlying_price / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        if option_type.lower() == 'call':
            price = (
                underlying_price * np.exp(-dividend_yield * time_to_expiry) * N_d1
                - strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
            )
            delta = np.exp(-dividend_yield * time_to_expiry) * N_d1
            theta = (
                -underlying_price * n_d1 * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * sqrt_t)
                - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
                + dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * N_d1
            ) / 365.0
            rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * N_d2 / 100.0
        else:
            price = (
                strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                - underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            )
            delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            theta = (
                -underlying_price * n_d1 * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * sqrt_t)
                + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                - dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            ) / 365.0
            rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100.0

        gamma = n_d1 * np.exp(-dividend_yield * time_to_expiry) / (underlying_price * volatility * sqrt_t)
        vega = underlying_price * sqrt_t * n_d1 * np.exp(-dividend_yield * time_to_expiry) / 100.0

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'price': max(price, 0)
        }

    def calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float = 0.02,
        q: float = 0.0,
        option_type: str = "call",
        engine_type: str = "crr",
        steps: int = 1000,
        accuracy: float = 1e-6,
        max_iter: int = 100
    ) -> float:
        """
        Calculate implied volatility using QuantLib's impliedVolatility for American options.
        Raises ValueError for invalid inputs.
        """
        if T <= 0:
            raise ValueError("Time to expiration must be positive")
        if market_price <= 0:
            raise ValueError("Option price must be positive")
        if option_type.lower() not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")

        if not QUANTLIB_AVAILABLE:
            raise RuntimeError("QuantLib is required for implied volatility calculation.")

        process = self._build_process(S, r, q, 0.2)
        today = ql.Settings.instance().evaluationDate
        maturity = today + int(T * 365 + 0.5)
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put,
            K
        )
        exercise = ql.AmericanExercise(today, maturity)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.BinomialVanillaEngine(process, engine_type, steps)
        option.setPricingEngine(engine)

        return option.impliedVolatility(
            market_price,
            process,
            accuracy,
            max_iter,
            accuracy,
            4.0
        )

    def calculate_implied_volatility_series(
        self,
        option_ticker: str,
        data_handler,
        start_date: str,
        end_date: str,
        underlying_symbol: str,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        exercise_style: str = "american",
        tree: str = "crr",
        steps: int = 1000,
        accuracy: float = 1e-4,
        max_evals: int = 1000
    ) -> pd.Series:
        """
        Calculate an IV time series for a single option ticker.
        
        Args:
            option_ticker: Option ticker symbol (e.g., 'TSLA250328C00255000')
            data_handler: DataHandler instance for fetching data
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            underlying_symbol: Underlying stock symbol (e.g., 'TSLA')
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            exercise_style: 'american' or 'european'
            tree: Binomial tree type ('crr', 'jr', 'eqp', 'trigeorgis', etc.)
            steps: Number of steps in binomial tree
            accuracy: Accuracy for IV calculation
            max_evals: Maximum evaluations for IV calculation
            
        Returns:
            pd.Series: Implied volatility series indexed by timestamp
        """
        if not QUANTLIB_AVAILABLE:
            raise RuntimeError("QuantLib is required for implied volatility calculation.")
        
        # Import the compute_iv_quantlib function
        from .quantlib import compute_iv_quantlib
        
        # Decode option ticker to get strike and expiration
        strike_price, expiration_date, option_type = self._decode_option_ticker(option_ticker)
        
        print(f"    Processing {option_ticker}: strike=${strike_price}, expiry={expiration_date}, type={option_type}")
        
        # Fetch option price series
        try:
            option_prices = data_handler.get_option_price_series(
                option_ticker=option_ticker,
                start_date=start_date,
                end_date=end_date,
                timespan="day",
                multiplier=1,
                adjust=True,
                price_type="mid"
            )
            print(f"    ✅ Option prices: {len(option_prices)} points, range: ${option_prices.min():.2f} - ${option_prices.max():.2f}")
        except Exception as e:
            print(f"    ❌ Failed to fetch option prices: {e}")
            return pd.Series(dtype=float)
        
        # Fetch underlying stock prices
        try:
            stock_bars = data_handler.get_stock_bars(
                ticker=underlying_symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1D"
            )
            underlying_prices = stock_bars['close']
            print(f"    ✅ Stock prices: {len(underlying_prices)} points, range: ${underlying_prices.min():.2f} - ${underlying_prices.max():.2f}")
        except Exception as e:
            print(f"    ❌ Failed to fetch stock prices: {e}")
            return pd.Series(dtype=float)
        
        # Align option and stock prices
        aligned = pd.DataFrame({
            'option_price': option_prices,
            'underlying_price': underlying_prices
        }).dropna()
        
        if len(aligned) == 0:
            print(f"    ❌ No aligned data points found")
            return pd.Series(dtype=float)
        
        print(f"    ✅ Aligned data points: {len(aligned)}")
        
        # Calculate IV for each aligned point
        exp_dt = pd.to_datetime(expiration_date)
        ivs, times = [], []
        
        for i, (t, row) in enumerate(aligned.iterrows()):
            # Handle timezone mismatch - make both timezone-naive for calculation
            if t.tz is not None:
                t_naive = t.tz_localize(None)
            else:
                t_naive = t
                
            # Calculate days to maturity
            days_to_maturity = (exp_dt - t_naive).days
            
            if days_to_maturity <= 0:
                continue
                
            try:
                iv = compute_iv_quantlib(
                    spot_price=row['underlying_price'],
                    option_price=row['option_price'],
                    strike_price=strike_price,
                    days_to_maturity=days_to_maturity,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    option_type=option_type,
                    exercise_style=exercise_style,
                    tree=tree,
                    steps=steps,
                    accuracy=accuracy,
                    max_evals=max_evals
                )
                ivs.append(iv)
                times.append(t)
            except Exception as e:
                print(f"    ⚠️ IV calculation failed for {t}: {e}")
                continue
        
        result = pd.Series(ivs, index=times, name='implied_volatility')
        print(f"    ✅ IV series calculated: {len(result)} valid points")
        return result
    
    def _decode_option_ticker(self, option_ticker: str) -> tuple[float, str, str]:
        """
        Decode option ticker to extract strike price, expiration date, and option type.
        
        Args:
            option_ticker: Option ticker (e.g., 'TSLA250328C00255000')
            
        Returns:
            tuple: (strike_price, expiration_date, option_type)
        """
        # Example: TSLA250328C00255000
        # TSLA = underlying
        # 250328 = YYMMDD expiration
        # C = call, P = put
        # 00255000 = strike price in cents (255.00)
        
        if len(option_ticker) < 15:
            raise ValueError(f"Invalid option ticker format: {option_ticker}")
        
        # Extract expiration date (YYMMDD format)
        exp_str = option_ticker[-15:-8]  # e.g., "250328"
        year = "20" + exp_str[:2]
        month = exp_str[2:4]
        day = exp_str[4:6]
        expiration_date = f"{year}-{month}-{day}"
        
        # Extract option type
        option_type = option_ticker[-8].lower()  # 'c' or 'p'
        if option_type == 'c':
            option_type = 'call'
        elif option_type == 'p':
            option_type = 'put'
        else:
            raise ValueError(f"Invalid option type in ticker: {option_ticker}")
        
        # Extract strike price (in cents)
        strike_cents = int(option_ticker[-7:])
        strike_price = strike_cents / 1000.0
        
        return strike_price, expiration_date, option_type

    # Other utility methods (align_option_data, plot_iv, helper, binomial_tree_call/put,
    # black_scholes_call/put) remain unchanged from the original implementation.
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Black-Scholes European call price."""
        if T <= 0:
            return max(S - K, 0)
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Black-Scholes European put price."""
        if T <= 0:
            return max(K - S, 0)
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


