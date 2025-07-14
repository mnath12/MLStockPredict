from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import warnings

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    warnings.warn("QuantLib not available. Using simplified Greeks calculations.")


class GreeksEngine:
    """
    Computes option Greeks using QuantLib when available, or simplified formulas as fallback.
    """
    
    def __init__(self, stock_bars: Optional[pd.DataFrame] = None, 
                 option_bars: Optional[pd.DataFrame] = None, 
                 pricing_model: str = "Black-Scholes"):
        self.stock_bars = stock_bars
        self.option_bars = option_bars
        self.pricing_model = pricing_model
        
        if QUANTLIB_AVAILABLE:
            self._setup_quantlib()
        
    def _setup_quantlib(self) -> None:
        """Setup QuantLib environment."""
        if QUANTLIB_AVAILABLE:
            self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
            self.day_count = ql.Actual365Fixed()
            ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

    def compute(self, symbol: str, underlying_price: float, strike: float, 
                time_to_expiry: float, volatility: float, risk_free_rate: float = 0.05,
                option_type: str = "call", dividend_yield: float = 0.0, **kwargs: Any) -> Dict[str, float]:
        """
        Compute Greeks for a single option.
        
        Args:
            symbol: Option symbol
            underlying_price: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            
        Returns:
            Dictionary with Greeks values
        """
        if QUANTLIB_AVAILABLE:
            return self._compute_quantlib(
                underlying_price, strike, time_to_expiry, volatility,
                risk_free_rate, option_type, dividend_yield
            )
        else:
            return self._compute_simplified(
                underlying_price, strike, time_to_expiry, volatility,
                risk_free_rate, option_type, dividend_yield
            )
    
    def _compute_quantlib(self, underlying_price: float, strike: float, 
                         time_to_expiry: float, volatility: float, risk_free_rate: float,
                         option_type: str, dividend_yield: float) -> Dict[str, float]:
        """Compute Greeks using QuantLib."""
        try:
            # Setup QuantLib objects
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
            risk_free_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(0, self.calendar, risk_free_rate, self.day_count)
            )
            dividend_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(0, self.calendar, dividend_yield, self.day_count)
            )
            volatility_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(0, self.calendar, volatility, self.day_count)
            )

            # Create process
            if self.pricing_model.lower() == "jump-diffusion":
                # Merton jump diffusion parameters
                jump_intensity = 0.1
                jump_mean = -0.05
                jump_volatility = 0.2
                process = ql.MertonJumpDiffusionProcess(
                    spot_handle, dividend_handle, risk_free_handle, 
                    volatility_handle, jump_intensity, jump_mean, jump_volatility
                )
            else:
                process = ql.BlackScholesMertonProcess(
                    spot_handle, dividend_handle, risk_free_handle, volatility_handle
                )

            # Create option
            maturity_date = ql.Date.todaysDate() + int(time_to_expiry * 365)
            exercise = ql.EuropeanExercise(maturity_date)
            
            option_type_ql = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
            payoff = ql.PlainVanillaPayoff(option_type_ql, strike)
            option = ql.VanillaOption(payoff, exercise)

            # Set pricing engine
            if self.pricing_model.lower() == "jump-diffusion":
                engine = ql.JumpDiffusionEngine(process)
            else:
                engine = ql.AnalyticEuropeanEngine(process)
            
            option.setPricingEngine(engine)

            # Calculate Greeks
            return {
                'delta': option.delta(),
                'gamma': option.gamma(),
                'vega': option.vega() / 100.0,  # Convert to decimal
                'theta': option.theta() / 365.0,  # Convert to daily
                'rho': option.rho() / 100.0,  # Convert to decimal
                'price': option.NPV()
            }

        except Exception as e:
            print(f"QuantLib calculation failed: {e}. Using simplified method.")
            return self._compute_simplified(
                underlying_price, strike, time_to_expiry, volatility,
                risk_free_rate, option_type, dividend_yield
            )
    
    def _compute_simplified(self, underlying_price: float, strike: float, 
                           time_to_expiry: float, volatility: float, risk_free_rate: float,
                           option_type: str, dividend_yield: float) -> Dict[str, float]:
        """Simplified Black-Scholes Greeks calculation."""
        if time_to_expiry <= 0:
            # Option expired
            intrinsic = max(0, underlying_price - strike) if option_type.lower() == 'call' else max(0, strike - underlying_price)
            return {
                'delta': 1.0 if (option_type.lower() == 'call' and underlying_price > strike) else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0,
                'price': intrinsic
            }
        
        # Black-Scholes formula components
        sqrt_t = np.sqrt(time_to_expiry)
        d1 = (np.log(underlying_price / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        
        # Standard normal CDF and PDF
        from scipy.stats import norm
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        if option_type.lower() == 'call':
            # Call option
            price = underlying_price * np.exp(-dividend_yield * time_to_expiry) * N_d1 - strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
            delta = np.exp(-dividend_yield * time_to_expiry) * N_d1
            theta = (- underlying_price * n_d1 * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * sqrt_t)
                    - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
                    + dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * N_d1) / 365
            rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * N_d2 / 100
        else:
            # Put option
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            theta = (- underlying_price * n_d1 * volatility * np.exp(-dividend_yield * time_to_expiry) / (2 * sqrt_t)
                    + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
                    - dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)) / 365
            rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        # Greeks that are the same for calls and puts
        gamma = n_d1 * np.exp(-dividend_yield * time_to_expiry) / (underlying_price * volatility * sqrt_t)
        vega = underlying_price * sqrt_t * n_d1 * np.exp(-dividend_yield * time_to_expiry) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'price': max(0, price)  # Ensure non-negative price
        }

    def align_option_data(self, option_symbols: List[str]) -> pd.DataFrame:
        """
        Align option data with stock data timestamps.
        
        Args:
            option_symbols: List of option symbols to align
            
        Returns:
            DataFrame with aligned option data
        """
        if self.stock_bars is None or self.option_bars is None:
            return pd.DataFrame()
        
        # Simple alignment - forward fill option data to match stock timestamps
        aligned_data = pd.DataFrame(index=self.stock_bars.index)
        
        for symbol in option_symbols:
            if symbol in self.option_bars.columns:
                option_data = self.option_bars[symbol].reindex(self.stock_bars.index)
                aligned_data[symbol] = option_data.fillna(method='ffill')
        
        return aligned_data

    def plot_iv(self, symbol: str, strikes: List[float], expiry: str) -> None:
        """
        Plot implied volatility surface (placeholder implementation).
        
        Args:
            symbol: Underlying symbol
            strikes: List of strike prices
            expiry: Expiration date string
        """
        try:
            import matplotlib.pyplot as plt
            
            # Generate sample IV data (replace with real calculation)
            iv_data = []
            for strike in strikes:
                # Sample IV curve - ATM has lowest IV
                atm_strike = strikes[len(strikes)//2]
                moneyness = strike / atm_strike
                iv = 0.2 + 0.1 * abs(moneyness - 1.0)  # Simple smile
                iv_data.append(iv)
            
            plt.figure(figsize=(10, 6))
            plt.plot(strikes, iv_data, 'b-o')
            plt.xlabel('Strike Price')
            plt.ylabel('Implied Volatility')
            plt.title(f'Implied Volatility Smile - {symbol} {expiry}')
            plt.grid(True)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

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