# Requirements: See requirements.txt or run the provided Jupyter cell to install dependencies.
# This script is designed to be run as a module or imported in a Jupyter notebook.
#
# NOTE: You must provide a working DataHandler implementation for real data access.
# A minimal stub is provided below for development/testing purposes.

# Imports for Jupyter notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from scipy.optimize import brentq, minimize
from scipy.interpolate import interp1d, griddata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import datetime as dt
from typing import List, Optional, Tuple, Dict
import re

# Financial libraries
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes
import fredapi

# Your data handler (paste your DataHandler class here)
# from data_handler import DataHandlerfrom __future__ import annotations

import datetime as dt
import re
from typing import List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockBarsRequest
from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit
from polygon import RESTClient

# Add import for trading calendar
try:
    import pandas_market_calendars as mcal
    HAS_MCAL = True
except ImportError:
    HAS_MCAL = False

# Remove the DataHandler stub and import the real one
from data_handler import DataHandler

class SurfaceBuilder:
    """
    Builds implied volatility surfaces from option chain data.
    Uses SVI parameterization for no-arbitrage enforcement.
    """

    def __init__(self, k_bins: int = 11, j_bins: int = 6):
        self.k_bins = k_bins  # moneyness grid points
        self.j_bins = j_bins  # maturity grid points
        self.fred = fredapi.Fred(api_key='YOUR_FRED_API_KEY')  # Replace with your key

    def get_risk_free_rate(self, date: str) -> float:
        """Get risk-free rate from FRED (3-month Treasury)"""
        try:
            rate_data = self.fred.get_series('DGS3MO', start=date, end=date)
            if not rate_data.empty:
                return rate_data.iloc[-1] / 100.0
            else:
                # Fallback to recent data
                recent_data = self.fred.get_series('DGS3MO', limit=5)
                return recent_data.dropna().iloc[-1] / 100.0
        except:
            return 0.05  # 5% fallback

    def clean_option_chain(self, df_chain: pd.DataFrame,
                          bid_ask_spread_threshold: float = 0.15) -> pd.DataFrame:
        """Clean option chain data with bid-ask spread filter"""
        df = df_chain.copy()

        # Calculate mid price and spread
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = (df['ask'] - df['bid']) / df['mid']

        # Filter by spread threshold
        df = df[df['spread'] <= bid_ask_spread_threshold]

        # Remove options with zero bid or negative values
        df = df[(df['bid'] > 0) & (df['ask'] > df['bid']) & (df['mid'] > 0)]

        # Calculate time to expiry in years
        df['expiry'] = pd.to_datetime(df['expiry'])
        df['date'] = pd.to_datetime(df['date'])
        df['tau'] = (df['expiry'] - df['date']).dt.days / 365.25

        # Remove very short-dated options (< 7 days)
        df = df[df['tau'] >= 7/365.25]

        return df.reset_index(drop=True)

    def compute_implied_volatility(self, df_clean: pd.DataFrame,
                                 spot_price: float, risk_free_rate: float) -> pd.DataFrame:
        """Compute implied volatility using Black-Scholes"""
        df = df_clean.copy()
        df['moneyness'] = np.log(df['strike'] / spot_price)
        df['iv'] = np.nan

        for idx, row in df.iterrows():
            try:
                iv = implied_volatility(
                    price=row['mid'],
                    S=spot_price,
                    K=row['strike'],
                    t=row['tau'],
                    r=risk_free_rate,
                    flag='c' if row['option_type'].lower() == 'call' else 'p'
                )
                df.loc[idx, 'iv'] = iv
            except:
                continue

        # Remove failed IV calculations
        df = df.dropna(subset=['iv'])

        # Remove extreme IVs (likely data errors)
        df = df[(df['iv'] > 0.01) & (df['iv'] < 5.0)]

        return df

    def fit_svi_smile(self, strikes: np.ndarray, ivs: np.ndarray,
                     spot: float, tau: float) -> Dict:
        """Fit SVI parameterization to a single expiry smile"""
        log_moneyness = np.log(strikes / spot)
        total_var = ivs**2 * tau

        def svi_formula(k, params):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def svi_objective(params):
            a, b, rho, m, sigma = params

            # SVI no-arbitrage constraints
            if sigma <= 0 or abs(rho) >= 1:
                return 1e6
            if a + b * sigma * np.sqrt(1 - rho**2) <= 0:
                return 1e6
            if b * (1 + abs(rho)) >= 2:
                return 1e6

            predicted = svi_formula(log_moneyness, params)
            return np.sum((total_var - predicted)**2)

        # Initial guess
        initial_params = [
            np.mean(total_var),  # a
            0.1,                 # b
            0.0,                 # rho
            0.0,                 # m (ATM)
            0.1                  # sigma
        ]

        try:
            result = minimize(svi_objective, initial_params,
                            method='L-BFGS-B',
                            bounds=[(-np.inf, np.inf), (0, np.inf), (-0.999, 0.999),
                                   (-np.inf, np.inf), (1e-6, np.inf)])

            if result.success:
                return {
                    'params': result.x,
                    'success': True,
                    'formula': lambda k: svi_formula(k, result.x)
                }
        except:
            pass

        # Fallback to simple interpolation
        from scipy.interpolate import interp1d
        try:
            interp_func = interp1d(log_moneyness, total_var,
                                 kind='linear', fill_value='extrapolate')
            return {
                'params': None,
                'success': False,
                'formula': interp_func
            }
        except:
            return {
                'params': None,
                'success': False,
                'formula': lambda k: np.full_like(k, np.mean(total_var))
            }

    def fit_surface_grid(self, df_chain: pd.DataFrame, spot_price: float,
                        current_date: str) -> np.ndarray:
        """
        Main method to fit IV surface on K×J grid
        Returns total variance surface
        """
        # Get risk-free rate
        r = self.get_risk_free_rate(current_date)

        # Clean and compute IV
        df_clean = self.clean_option_chain(df_chain)
        df_iv = self.compute_implied_volatility(df_clean, spot_price, r)

        if df_iv.empty:
            raise ValueError("No valid options data after cleaning")

        # Create grid
        min_tau, max_tau = df_iv['tau'].min(), df_iv['tau'].max()
        tau_grid = np.linspace(min_tau, max_tau, self.j_bins)

        min_k = df_iv['moneyness'].quantile(0.05)
        max_k = df_iv['moneyness'].quantile(0.95)
        k_grid = np.linspace(min_k, max_k, self.k_bins)

        # Initialize surface
        surface = np.zeros((self.k_bins, self.j_bins))

        # Fit SVI for each maturity bucket
        for j, tau in enumerate(tau_grid):
            # Find options close to this maturity
            tau_mask = np.abs(df_iv['tau'] - tau) <= (max_tau - min_tau) / (2 * self.j_bins)
            df_slice = df_iv[tau_mask]

            if len(df_slice) < 3:
                # Not enough data, use nearby data
                sorted_by_tau = df_iv.iloc[(df_iv['tau'] - tau).abs().argsort()]
                df_slice = sorted_by_tau.head(10)

            if len(df_slice) >= 3:
                svi_fit = self.fit_svi_smile(
                    df_slice['strike'].values,
                    df_slice['iv'].values,
                    spot_price,
                    tau
                )

                # Evaluate on moneyness grid
                for i, k in enumerate(k_grid):
                    try:
                        total_var = svi_fit['formula'](k)
                        surface[i, j] = max(total_var, 1e-6)  # Floor at small positive
                    except:
                        surface[i, j] = 0.01  # Fallback
            else:
                # Fill with average IV
                avg_var = (df_iv['iv']**2 * df_iv['tau']).mean()
                surface[:, j] = avg_var

        # Store grid info for later use
        self.k_grid = k_grid
        self.tau_grid = tau_grid
        self.spot_price = spot_price

        return surface

# Example usage
def demo_surface_builder():
    """Demo function showing how to use SurfaceBuilder"""
    print("SurfaceBuilder created successfully!")
    print("Usage:")
    print("builder = SurfaceBuilder(k_bins=11, j_bins=6)")
    print("surface = builder.fit_surface_grid(df_chain, spot_price, '2025-01-15')")
    print("# Returns surface shape (11, 6) with total variance values")

class HARModel:
    """
    HAR-RV-J (Heterogeneous AutoRegressive - Realized Volatility - Jumps) model
    for forecasting implied volatility term structure.
    """

    def __init__(self, window_expanding: bool = True, refit_frequency: int = 22):
        self.window_expanding = window_expanding
        self.refit_frequency = refit_frequency  # days
        self.models = {}  # Store models for each maturity bucket
        self.feature_names = ['RV_daily', 'RV_week', 'RV_month', 'Jump_flag']

    def compute_realized_volatility(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute realized volatility features from intraday bars
        """
        # Ensure timestamp index
        if not isinstance(bars_df.index, pd.DatetimeIndex):
            bars_df.index = pd.to_datetime(bars_df.index)

        # Compute returns
        bars_df['returns'] = np.log(bars_df['close'] / bars_df['close'].shift(1))
        bars_df = bars_df.dropna()

        # Group by date for daily RV calculation
        daily_rv = []

        for date, group in bars_df.groupby(bars_df.index.date):
            if len(group) < 2:
                continue

            # Daily realized variance (sum of squared intraday returns)
            rv_daily = np.sum(group['returns']**2) * 252  # Annualized

            # Jump detection using MAD (Median Absolute Deviation)
            abs_returns = np.abs(group['returns'])
            mad = np.median(abs_returns - np.median(abs_returns))
            jump_threshold = 4 * mad
            jump_flag = int(np.any(abs_returns > jump_threshold))

            daily_rv.append({
                'date': pd.to_datetime(date),
                'RV_daily': np.sqrt(rv_daily),  # Convert to volatility
                'Jump_flag': jump_flag
            })

        rv_df = pd.DataFrame(daily_rv).set_index('date')

        # Compute weekly and monthly averages
        rv_df['RV_week'] = rv_df['RV_daily'].rolling(5, min_periods=3).mean()
        rv_df['RV_month'] = rv_df['RV_daily'].rolling(22, min_periods=10).mean()

        return rv_df.dropna()

    def prepare_har_features(self, rv_df: pd.DataFrame, surface_history: List[np.ndarray],
                           surface_dates: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Prepare feature matrix and targets for HAR regression

        Args:
            rv_df: DataFrame with RV features
            surface_history: List of historical surface grids
            surface_dates: Corresponding dates for surfaces

        Returns:
            X: Feature matrix (n_days, n_features)
            targets: Dict with target series for each maturity bucket
        """
        # Align dates
        surface_dates_dt = pd.to_datetime(surface_dates)
        common_dates = rv_df.index.intersection(surface_dates_dt)

        if len(common_dates) < 30:
            raise ValueError(f"Insufficient overlapping data: {len(common_dates)} days")

        # Build feature matrix
        rv_aligned = rv_df.loc[common_dates]
        X = rv_aligned[self.feature_names].values

        # Extract targets (ATM variance for each maturity)
        targets = {}
        j_bins = surface_history[0].shape[1]
        k_center = surface_history[0].shape[0] // 2  # ATM index

        for j in range(j_bins):
            target_series = []
            for date in common_dates:
                idx = list(surface_dates_dt).index(date)
                surface = surface_history[idx]
                target_series.append(surface[k_center, j])  # ATM variance

            targets[f'tau_{j}'] = np.array(target_series)

        return X, targets

    def fit_har_models(self, X: np.ndarray, targets: Dict) -> Dict:
        """
        Fit separate HAR model for each maturity bucket
        """
        models = {}

        for tau_key, y in targets.items():
            # Remove any invalid values
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]

            if len(y_valid) < 10:
                print(f"Warning: Insufficient data for {tau_key}, skipping")
                continue

            # Fit OLS regression
            model = LinearRegression(fit_intercept=True)
            model.fit(X_valid, y_valid)

            # Store model and diagnostics
            y_pred = model.predict(X_valid)
            r2 = model.score(X_valid, y_valid)
            mse = np.mean((y_valid - y_pred)**2)

            models[tau_key] = {
                'model': model,
                'r2': r2,
                'mse': mse,
                'n_obs': len(y_valid),
                'coefficients': model.coef_,
                'intercept': model.intercept_
            }

            print(f"{tau_key}: R² = {r2:.3f}, MSE = {mse:.6f}, N = {len(y_valid)}")

        return models

    def forecast_har(self, rv_features: np.ndarray) -> Dict:
        """
        Generate one-step-ahead forecasts using fitted HAR models

        Args:
            rv_features: Latest RV features [RV_daily, RV_week, RV_month, Jump_flag]

        Returns:
            Dictionary with forecasts for each maturity bucket
        """
        forecasts = {}

        for tau_key, model_dict in self.models.items():
            model = model_dict['model']
            forecast = model.predict(rv_features.reshape(1, -1))[0]
            forecasts[tau_key] = max(forecast, 1e-6)  # Floor at small positive

        return forecasts

    def train(self, bars_df: pd.DataFrame, surface_history: List[np.ndarray],
              surface_dates: List[str]) -> Dict:
        """
        Complete training pipeline for HAR-RV-J model
        """
        print("Computing realized volatility features...")
        rv_df = self.compute_realized_volatility(bars_df)
        print(f"Generated RV features for {len(rv_df)} days")

        print("Preparing HAR features and targets...")
        X, targets = self.prepare_har_features(rv_df, surface_history, surface_dates)
        print(f"Feature matrix shape: {X.shape}")

        print("Fitting HAR models...")
        self.models = self.fit_har_models(X, targets)

        print(f"Successfully fitted {len(self.models)} maturity buckets")
        return self.models

    def save_models(self, filepath: str):
        """Save fitted models to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'window_expanding': self.window_expanding
            }, f)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load fitted models from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.feature_names = data['feature_names']
            self.window_expanding = data['window_expanding']
        print(f"Models loaded from {filepath}")

# Utility function to generate sample data for testing
def generate_sample_har_data(n_days: int = 100, j_bins: int = 6) -> Tuple:
    """Generate sample data for testing HAR model"""

    # Sample RV time series with realistic properties
    np.random.seed(42)
    rv_daily = 0.2 + 0.1 * np.random.randn(n_days).cumsum() * 0.01
    rv_daily = np.abs(rv_daily)  # Ensure positive

    # Create bars DataFrame
    dates = pd.date_range('2024-01-01', periods=n_days*78, freq='5min')  # 78 5-min bars per day
    bars_data = []

    for i in range(n_days):
        daily_vol = rv_daily[i]
        daily_return = np.random.normal(0, daily_vol/np.sqrt(252))

        for j in range(78):
            price = 100 * np.exp(daily_return * (j+1)/78 + np.random.normal(0, daily_vol/np.sqrt(252*78)))
            bars_data.append({
                'timestamp': dates[i*78 + j],
                'close': price,
                'open': price * (1 + np.random.normal(0, 0.001)),
                'high': price * (1 + abs(np.random.normal(0, 0.002))),
                'low': price * (1 - abs(np.random.normal(0, 0.002))),
                'volume': np.random.randint(1000, 10000)
            })

    bars_df = pd.DataFrame(bars_data).set_index('timestamp')

    # Sample surface history
    k_bins = 11
    surface_history = []
    surface_dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

    for i in range(n_days):
        # Create realistic surface with term structure and smile
        tau_range = np.linspace(0.02, 1.0, j_bins)
        k_range = np.linspace(-0.3, 0.3, k_bins)

        surface = np.zeros((k_bins, j_bins))
        base_vol = rv_daily[i]

        for j, tau in enumerate(tau_range):
            for k, moneyness in enumerate(k_range):
                # Simple model: IV = base_vol + term_structure + smile
                term_premium = 0.05 * np.sqrt(tau)
                smile = 0.02 * moneyness**2  # U-shaped smile
                vol = base_vol + term_premium + smile
                surface[k, j] = vol**2 * tau  # Convert to total variance

        surface_history.append(surface)

    return bars_df, surface_history, [d.strftime('%Y-%m-%d') for d in surface_dates]

# Demo function
def demo_har_model():
    """Demonstrate HAR model usage"""
    print("Generating sample data...")
    bars_df, surface_history, surface_dates = generate_sample_har_data()

    print("Training HAR model...")
    har = HARModel()
    models = har.train(bars_df, surface_history, surface_dates)

    print("\nForecasting...")
    # Use latest RV features for forecast
    rv_df = har.compute_realized_volatility(bars_df)
    latest_features = rv_df.iloc[-1][har.feature_names].values
    forecasts = har.forecast_har(latest_features)

    print("Forecasts:")
    for tau_key, forecast in forecasts.items():
        print(f"{tau_key}: {forecast:.4f}")

    return har, forecasts

class IVPlotter:
    """
    Comprehensive plotting suite for implied volatility analysis
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')

    def plot_atm_term_structure(self, surface_builder: SurfaceBuilder,
                               har_forecasts: Dict, actual_surface: np.ndarray = None,
                               title: str = "ATM Term Structure") -> plt.Figure:
        """
        Plot ATM implied volatility term structure comparing actual vs forecast
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        tau_grid = surface_builder.tau_grid
        k_center = len(surface_builder.k_grid) // 2

        # Plot actual surface (if provided)
        if actual_surface is not None:
            atm_actual = np.sqrt(actual_surface[k_center, :] / tau_grid)
            ax.plot(tau_grid * 365, atm_actual, 'o-', color='blue',
                   linewidth=2, markersize=6, label='Actual IV')

        # Plot HAR forecast
        if har_forecasts:
            har_vols = []
            for j in range(len(tau_grid)):
                tau_key = f'tau_{j}'
                if tau_key in har_forecasts:
                    vol = np.sqrt(har_forecasts[tau_key] / tau_grid[j])
                    har_vols.append(vol)
                else:
                    har_vols.append(np.nan)

            ax.plot(tau_grid * 365, har_vols, 's--', color='red',
                   linewidth=2, markersize=8, label='HAR-RV-J Forecast')

        ax.set_xlabel('Days to Expiry', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        plt.tight_layout()
        return fig

    def plot_volatility_smile(self, surface_builder: SurfaceBuilder,
                             actual_surface: np.ndarray = None,
                             maturity_days: int = 30,
                             title: str = "30-Day Volatility Smile") -> plt.Figure:
        """
        Plot volatility smile for a specific maturity
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Find closest maturity to target
        target_tau = maturity_days / 365.25
        tau_idx = np.argmin(np.abs(surface_builder.tau_grid - target_tau))
        actual_tau = surface_builder.tau_grid[tau_idx]

        k_grid = surface_builder.k_grid
        spot = surface_builder.spot_price

        # Convert moneyness to strike/spot ratio for easier interpretation
        strike_ratios = np.exp(k_grid)

        if actual_surface is not None:
            smile_vars = actual_surface[:, tau_idx]
            smile_vols = np.sqrt(smile_vars / actual_tau)

            ax.plot(strike_ratios, smile_vols, 'o-', color='blue',
                   linewidth=2, markersize=6, label=f'Actual IV ({actual_tau*365:.0f}d)')

        ax.set_xlabel('Strike / Spot', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        ax.set_title(f'{title} ({actual_tau*365:.0f} days)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add ATM line
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='ATM')

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        plt.tight_layout()
        return fig

    def plot_3d_surface(self, surface_builder: SurfaceBuilder,
                       surface: np.ndarray, title: str = "IV Surface") -> go.Figure:
        """
        Interactive 3D volatility surface using Plotly
        """
        k_grid = surface_builder.k_grid
        tau_grid = surface_builder.tau_grid

        # Convert to volatility surface
        K_mesh, T_mesh = np.meshgrid(k_grid, tau_grid, indexing='ij')
        vol_surface = np.sqrt(surface / T_mesh)

        # Convert moneyness to strike ratios for display
        strike_ratios = np.exp(k_grid)
        days_to_expiry = tau_grid * 365

        fig = go.Figure(data=[go.Surface(
            x=days_to_expiry,
            y=strike_ratios,
            z=vol_surface,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Implied Vol")
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Strike / Spot',
                zaxis_title='Implied Volatility',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )

        return fig

    def plot_surface_heatmap(self, surface_builder: SurfaceBuilder,
                           surface: np.ndarray, title: str = "IV Heatmap") -> plt.Figure:
        """
        2D heatmap of volatility surface
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        k_grid = surface_builder.k_grid
        tau_grid = surface_builder.tau_grid

        # Convert to volatility
        vol_surface = np.zeros_like(surface)
        for i in range(surface.shape[0]):
            for j in range(surface.shape[1]):
                vol_surface[i, j] = np.sqrt(surface[i, j] / tau_grid[j])

        # Create heatmap
        im = ax.imshow(vol_surface, cmap='RdYlBu_r', aspect='auto',
                      extent=[tau_grid[0]*365, tau_grid[-1]*365,
                             np.exp(k_grid[-1]), np.exp(k_grid[0])])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Implied Volatility', fontsize=12)

        # Format colorbar as percentage
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        ax.set_xlabel('Days to Expiry', fontsize=12)
        ax.set_ylabel('Strike / Spot', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add contour lines
        X, Y = np.meshgrid(tau_grid * 365, np.exp(k_grid))
        contours = ax.contour(X, Y, vol_surface, colors='white', alpha=0.6, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f%%')

        plt.tight_layout()
        return fig

    def plot_forecast_comparison(self, surface_builder: SurfaceBuilder,
                               actual_surface: np.ndarray,
                               har_forecasts: Dict,
                               title: str = "Forecast vs Actual Comparison") -> plt.Figure:
        """
        Side-by-side comparison of actual and forecasted surfaces
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        k_grid = surface_builder.k_grid
        tau_grid = surface_builder.tau_grid

        # Convert to volatility surfaces
        actual_vol = np.zeros_like(actual_surface)
        forecast_vol = np.zeros_like(actual_surface)

        for i in range(actual_surface.shape[0]):
            for j in range(actual_surface.shape[1]):
                actual_vol[i, j] = np.sqrt(actual_surface[i, j] / tau_grid[j])

                # Fill forecast surface from HAR predictions
                tau_key = f'tau_{j}'
                if tau_key in har_forecasts:
                    forecast_vol[i, j] = np.sqrt(har_forecasts[tau_key] / tau_grid[j])
                else:
                    forecast_vol[i, j] = actual_vol[i, j]  # Fallback

        # Plot actual surface
        extent = [tau_grid[0]*365, tau_grid[-1]*365,
                 np.exp(k_grid[-1]), np.exp(k_grid[0])]

        im1 = ax1.imshow(actual_vol, cmap='RdYlBu_r', aspect='auto', extent=extent)
        ax1.set_title('Actual IV Surface', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Days to Expiry')
        ax1.set_ylabel('Strike / Spot')

        # Plot forecast surface
        im2 = ax2.imshow(forecast_vol, cmap='RdYlBu_r', aspect='auto', extent=extent)
        ax2.set_title('HAR-RV-J Forecast', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Days to Expiry')
        ax2.set_ylabel('Strike / Spot')

        # Shared colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Implied Volatility', fontsize=12)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        plt.suptitle(title, fontsize=14, fontweight='bold')
        return fig

    def plot_rv_features(self, rv_df: pd.DataFrame,
                        title: str = "Realized Volatility Features") -> plt.Figure:
        """
        Plot realized volatility time series and features
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Daily RV
        axes[0,0].plot(rv_df.index, rv_df['RV_daily'], color='blue', linewidth=1)
        axes[0,0].set_title('Daily Realized Volatility', fontweight='bold')
        axes[0,0].set_ylabel('RV')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Weekly and Monthly RV
        axes[0,1].plot(rv_df.index, rv_df['RV_daily'], color='blue', alpha=0.7, label='Daily')
        axes[0,1].plot(rv_df.index, rv_df['RV_week'], color='red', linewidth=2, label='Weekly MA')
        axes[0,1].plot(rv_df.index, rv_df['RV_month'], color='green', linewidth=2, label='Monthly MA')
        axes[0,1].set_title('RV Components', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Jump flags
        jump_dates = rv_df[rv_df['Jump_flag'] == 1].index
        axes[1,0].plot(rv_df.index, rv_df['RV_daily'], color='blue', alpha=0.7)
        if len(jump_dates) > 0:
            jump_values = rv_df.loc[jump_dates, 'RV_daily']
            axes[1,0].scatter(jump_dates, jump_values, color='red', s=50,
                            zorder=5, label=f'Jumps ({len(jump_dates)})')
        axes[1,0].set_title('Jump Detection', fontweight='bold')
        axes[1,0].set_ylabel('RV')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Distribution of RV
        axes[1,1].hist(rv_df['RV_daily'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1,1].set_title('RV Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Realized Volatility')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        plt.tight_layout()
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        return fig

    def plot_model_diagnostics(self, har_model: 'HARModel',
                             title: str = "HAR Model Diagnostics") -> plt.Figure:
        """
        Plot HAR model fit diagnostics
        """
        models = har_model.models
        n_models = len(models)

        if n_models == 0:
            raise ValueError("No fitted models found")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # R-squared by maturity
        tau_keys = list(models.keys())
        r2_values = [models[key]['r2'] for key in tau_keys]
        mse_values = [models[key]['mse'] for key in tau_keys]

        axes[0,0].bar(range(len(tau_keys)), r2_values, color='steelblue', alpha=0.7)
        axes[0,0].set_title('R² by Maturity Bucket', fontweight='bold')
        axes[0,0].set_xlabel('Maturity Bucket')
        axes[0,0].set_ylabel('R²')
        axes[0,0].set_xticks(range(len(tau_keys)))
        axes[0,0].set_xticklabels([f'τ_{i}' for i in range(len(tau_keys))])
        axes[0,0].grid(True, alpha=0.3)

        # MSE by maturity
        axes[0,1].bar(range(len(tau_keys)), mse_values, color='crimson', alpha=0.7)
        axes[0,1].set_title('MSE by Maturity Bucket', fontweight='bold')
        axes[0,1].set_xlabel('Maturity Bucket')
        axes[0,1].set_ylabel('MSE')
        axes[0,1].set_xticks(range(len(tau_keys)))
        axes[0,1].set_xticklabels([f'τ_{i}' for i in range(len(tau_keys))])
        axes[0,1].grid(True, alpha=0.3)

        # Coefficient heatmap
        feature_names = har_model.feature_names
        coef_matrix = np.zeros((len(tau_keys), len(feature_names)))

        for i, tau_key in enumerate(tau_keys):
            coef_matrix[i, :] = models[tau_key]['coefficients']

        im = axes[1,0].imshow(coef_matrix, cmap='RdBu_r', aspect='auto')
        axes[1,0].set_title('Coefficient Heatmap', fontweight='bold')
        axes[1,0].set_xticks(range(len(feature_names)))
        axes[1,0].set_xticklabels(feature_names, rotation=45)
        axes[1,0].set_yticks(range(len(tau_keys)))
        axes[1,0].set_yticklabels([f'τ_{i}' for i in range(len(tau_keys))])

        # Add colorbar for coefficients
        cbar = plt.colorbar(im, ax=axes[1,0])
        cbar.set_label('Coefficient Value')

        # Model summary statistics
        avg_r2 = np.mean(r2_values)
        avg_mse = np.mean(mse_values)
        total_obs = sum([models[key]['n_obs'] for key in tau_keys])

        summary_text = f"""Model Summary:

        Average R²: {avg_r2:.3f}
        Average MSE: {avg_mse:.6f}
        Total Observations: {total_obs:,}
        Maturity Buckets: {len(tau_keys)}
        Features: {len(feature_names)}
        """

        axes[1,1].text(0.1, 0.5, summary_text, transform=axes[1,1].transAxes,
                      fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Summary Statistics', fontweight='bold')

        plt.tight_layout()
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        return fig

# Demo plotting function
def demo_plotting():
    """Demonstrate plotting capabilities with sample data"""
    print("Creating sample data for plotting demo...")

    # Generate sample data
    from har_model import generate_sample_har_data, HARModel
    bars_df, surface_history, surface_dates = generate_sample_har_data()

    # Fit HAR model
    har = HARModel()
    har.train(bars_df, surface_history, surface_dates)

    # Generate forecast
    rv_df = har.compute_realized_volatility(bars_df)
    latest_features = rv_df.iloc[-1][har.feature_names].values
    forecasts = har.forecast_har(latest_features)

    # Create mock surface builder for plotting
    class MockSurfaceBuilder:
        def __init__(self):
            self.k_grid = np.linspace(-0.3, 0.3, 11)
            self.tau_grid = np.linspace(0.02, 1.0, 6)
            self.spot_price = 100.0

    builder = MockSurfaceBuilder()
    plotter = IVPlotter()

    # Generate plots
    print("Generating plots...")

    # 1. ATM term structure
    fig1 = plotter.plot_atm_term_structure(builder, forecasts, surface_history[-1])
    plt.show()

    # 2. Volatility smile
    fig2 = plotter.plot_volatility_smile(builder, surface_history[-1])
    plt.show()

    # 3. RV features
    fig3 = plotter.plot_rv_features(rv_df)
    plt.show()

    # 4. Model diagnostics
    fig4 = plotter.plot_model_diagnostics(har)
    plt.show()

    print("Plotting demo completed!")

    return plotter, builder, har

class IVForecastingPipeline:
    """
    Complete end-to-end pipeline for implied volatility forecasting
    Integrates data handling, surface building, HAR modeling, and visualization
    """

    def __init__(self, data_handler: 'DataHandler', fred_api_key: str,
                 k_bins: int = 11, j_bins: int = 6):
        self.data_handler = data_handler
        self.surface_builder = SurfaceBuilder(k_bins, j_bins)
        self.surface_builder.fred.api_key = fred_api_key
        self.har_model = HARModel()
        self.plotter = IVPlotter()

        # Storage for historical data
        self.surface_history = []
        self.surface_dates = []

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str,
                        timeframe: str = "5Min") -> pd.DataFrame:
        """Fetch stock bars for RV calculation"""
        print(f"Fetching stock data for {symbol}...")
        bars_df = self.data_handler.get_stock_bars(
            ticker=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        print(f"Retrieved {len(bars_df)} bars")
        return bars_df

    def fetch_option_chain(self, symbol: str, date: str,
                          dte_min: int = 7, dte_max: int = 365) -> pd.DataFrame:
        """Fetch option chain for a specific date"""
        print(f"Fetching option chain for {symbol} on {date}...")

        # Calculate date range for option search
        base_date = pd.to_datetime(date)
        exp_from = (base_date + pd.Timedelta(days=dte_min)).strftime('%Y-%m-%d')
        exp_to = (base_date + pd.Timedelta(days=dte_max)).strftime('%Y-%m-%d')

        # Get current stock price for strike filtering
        stock_data = self.fetch_stock_data(symbol, date, date, "1D")
        if stock_data.empty:
            raise ValueError(f"No stock data found for {symbol} on {date}")

        spot_price = stock_data['close'].iloc[-1]
        strike_min = spot_price * 0.7  # 30% OTM puts
        strike_max = spot_price * 1.3  # 30% OTM calls

        # Search for option contracts
        option_tickers = self.data_handler.options_search(
            underlying=symbol,
            exp_from=exp_from,
            exp_to=exp_to,
            strike_min=strike_min,
            strike_max=strike_max,
            as_of=date,
            limit=2000
        )

        print(f"Found {len(option_tickers)} option contracts")

        if not option_tickers:
            raise ValueError(f"No option contracts found for {symbol}")

        # Fetch option quotes (using aggregates as proxy for EOD quotes)
        chain_data = []
        for ticker in option_tickers:
            try:
                # Parse option ticker to extract details
                # Format: AAPL240315C00185000 (SYMBOL + YYMMDD + C/P + STRIKE*1000)
                parts = re.match(r'([A-Z]+)(\d{6})([CP])(\d{8})', ticker)
                if not parts:
                    continue

                underlying, exp_str, opt_type, strike_str = parts.groups()
                exp_date = pd.to_datetime(f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}")
                strike = float(strike_str) / 1000.0
                option_type = 'call' if opt_type == 'C' else 'put'

                # Get option price data
                option_data = self.data_handler.get_option_aggregates(
                    option_ticker=ticker,
                    start_date=date,
                    end_date=date,
                    timespan="day"
                )

                if not option_data.empty:
                    last_price = option_data['close'].iloc[-1]
                    volume = option_data['volume'].iloc[-1]

                    # Estimate bid-ask spread (simplified)
                    spread_est = max(0.01, last_price * 0.05)  # 5% spread estimate
                    bid = max(0.01, last_price - spread_est/2)
                    ask = last_price + spread_est/2

                    chain_data.append({
                        'date': date,
                        'expiry': exp_date,
                        'strike': strike,
                        'option_type': option_type,
                        'bid': bid,
                        'ask': ask,
                        'last': last_price,
                        'volume': volume,
                        'ticker': ticker
                    })

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        if not chain_data:
            raise ValueError("No valid option data retrieved")

        chain_df = pd.DataFrame(chain_data)
        print(f"Successfully processed {len(chain_df)} options")
        return chain_df

    def build_surface_for_date(self, symbol: str, date: str) -> np.ndarray:
        """Build IV surface for a specific date"""
        print(f"Building IV surface for {symbol} on {date}...")

        # Fetch data
        option_chain = self.fetch_option_chain(symbol, date)
        stock_data = self.fetch_stock_data(symbol, date, date, "1D")
        spot_price = stock_data['close'].iloc[-1]

        # Build surface
        surface = self.surface_builder.fit_surface_grid(
            option_chain, spot_price, date
        )

        # Store for history
        self.surface_history.append(surface)
        self.surface_dates.append(date)

        print(f"Surface built successfully: shape {surface.shape}")
        return surface

    def build_historical_surfaces(self, symbol: str, start_date: str,
                                end_date: str, frequency: str = "1W") -> List[np.ndarray]:
        """Build surfaces for multiple historical dates using only trading days"""
        print(f"Building historical surfaces for {symbol}...")

        # Use NYSE trading days if possible, else business days
        if HAS_MCAL:
            nyse = mcal.get_calendar('NYSE')
            schedule = nyse.schedule(start_date=start_date, end_date=end_date)
            trading_days = schedule.index
        else:
            trading_days = pd.date_range(start_date, end_date, freq="B")

        # Determine step for frequency
        if frequency.lower() in ["1w", "w", "weekly"]:
            step = 5  # Approximate 1 week as 5 trading days
        elif frequency.lower() in ["1d", "d", "daily"]:
            step = 1
        else:
            # Default to weekly if unknown
            step = 5

        dates = trading_days[::step]
        surfaces = []

        for date in dates:
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            try:
                surface = self.build_surface_for_date(symbol, date_str)
                surfaces.append(surface)
                print(f"✓ {date_str}")
            except Exception as e:
                print(f"✗ {date_str}: {e}")
                continue

        print(f"Built {len(surfaces)} surfaces successfully")
        return surfaces

    def train_har_model(self, symbol: str, bars_start_date: str,
                       bars_end_date: str) -> Dict:
        """Train HAR model using historical data"""
        print("Training HAR-RV-J model...")

        # Fetch stock bars for RV calculation
        bars_df = self.fetch_stock_data(symbol, bars_start_date, bars_end_date)

        if len(self.surface_history) == 0:
            raise ValueError("No historical surfaces available. Run build_historical_surfaces first.")

        # Train model
        models = self.har_model.train(bars_df, self.surface_history, self.surface_dates)

        print("HAR model training completed!")
        return models

    def generate_forecast(self, symbol: str, forecast_date: str) -> Dict:
        """Generate IV forecast for next day"""
        print(f"Generating forecast for {symbol}...")

        # Get recent bars for RV calculation
        end_date = pd.to_datetime(forecast_date)
        start_date = end_date - pd.Timedelta(days=30)

        recent_bars = self.fetch_stock_data(
            symbol, start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        # Compute latest RV features
        rv_df = self.har_model.compute_realized_volatility(recent_bars)
        if rv_df.empty:
            raise ValueError("Could not compute RV features")

        latest_features = rv_df.iloc[-1][self.har_model.feature_names].values

        # Generate forecast
        forecasts = self.har_model.forecast_har(latest_features)

        print(f"Forecast generated for {len(forecasts)} maturity buckets")
        return forecasts

    def create_comprehensive_report(self, symbol: str, forecast_date: str,
                                  actual_surface: np.ndarray = None) -> Dict:
        """Generate comprehensive analysis report with all plots"""
        print("Creating comprehensive IV forecasting report...")

        # Generate forecast
        forecasts = self.generate_forecast(symbol, forecast_date)

        # Create all plots
        plots = {}

        # 1. ATM Term Structure
        plots['atm_term'] = self.plotter.plot_atm_term_structure(
            self.surface_builder, forecasts, actual_surface,
            title=f"{symbol} ATM Term Structure - {forecast_date}"
        )

        # 2. Volatility Smile (if we have actual surface)
        if actual_surface is not None:
            plots['smile'] = self.plotter.plot_volatility_smile(
                self.surface_builder, actual_surface,
                title=f"{symbol} Volatility Smile - {forecast_date}"
            )

        # 3. Model Diagnostics
        plots['diagnostics'] = self.plotter.plot_model_diagnostics(
            self.har_model,
            title=f"{symbol} HAR Model Diagnostics"
        )

        # 4. RV Features (get recent data)
        end_date = pd.to_datetime(forecast_date)
        start_date = end_date - pd.Timedelta(days=60)
        recent_bars = self.fetch_stock_data(
            symbol, start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        rv_df = self.har_model.compute_realized_volatility(recent_bars)

        plots['rv_features'] = self.plotter.plot_rv_features(
            rv_df, title=f"{symbol} Realized Volatility Analysis"
        )

        # 5. 3D Surface (if actual surface available)
        if actual_surface is not None:
            plots['surface_3d'] = self.plotter.plot_3d_surface(
                self.surface_builder, actual_surface,
                title=f"{symbol} IV Surface - {forecast_date}"
            )

        # 6. Surface comparison (if actual surface available)
        if actual_surface is not None:
            plots['comparison'] = self.plotter.plot_forecast_comparison(
                self.surface_builder, actual_surface, forecasts,
                title=f"{symbol} Forecast vs Actual - {forecast_date}"
            )

        report = {
            'symbol': symbol,
            'forecast_date': forecast_date,
            'forecasts': forecasts,
            'plots': plots,
            'model_summary': {
                'n_maturity_buckets': len(forecasts),
                'surface_history_length': len(self.surface_history),
                'har_models': len(self.har_model.models)
            }
        }

        print("Comprehensive report created!")
        return report

    def save_pipeline_state(self, filepath: str):
        """Save the entire pipeline state"""
        import pickle

        state = {
            'surface_history': self.surface_history,
            'surface_dates': self.surface_dates,
            'surface_builder_config': {
                'k_bins': self.surface_builder.k_bins,
                'j_bins': self.surface_builder.j_bins
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        # Save HAR models separately
        har_path = filepath.replace('.pkl', '_har.pkl')
        self.har_model.save_models(har_path)

        print(f"Pipeline state saved to {filepath}")

    def load_pipeline_state(self, filepath: str):
        """Load pipeline state from file"""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.surface_history = state['surface_history']
        self.surface_dates = state['surface_dates']

        # Load HAR models
        har_path = filepath.replace('.pkl', '_har.pkl')
        self.har_model.load_models(har_path)

        print(f"Pipeline state loaded from {filepath}")

# Quick start function
def quick_start_demo():
    """Quick demonstration of the complete pipeline"""
    print("=== IV Forecasting Pipeline Demo ===")
    print("This demo shows how to use the complete system.")
    print("You'll need to replace the API keys and run with real data.")

    # Placeholder API keys - replace with your own
    demo_code = '''
    # Initialize data handler with your API keys
    data_handler = DataHandler(
        alpaca_api_key="YOUR_ALPACA_KEY",
        alpaca_secret="YOUR_ALPACA_SECRET",
        polygon_key="YOUR_POLYGON_KEY"
    )

    # Create pipeline
    pipeline = IVForecastingPipeline(
        data_handler=data_handler,
        fred_api_key="YOUR_FRED_KEY",
        k_bins=11,
        j_bins=6
    )

    # Example workflow for AAPL
    symbol = "AAPL"

    # 1. Build historical surfaces (weekly over past 3 months)
    surfaces = pipeline.build_historical_surfaces(
        symbol=symbol,
        start_date="2024-10-01",
        end_date="2025-01-15",
        frequency="1W"
    )

    # 2. Train HAR model
    models = pipeline.train_har_model(
        symbol=symbol,
        bars_start_date="2024-07-01",
        bars_end_date="2025-01-15"
    )

    # 3. Generate forecast and create report
    report = pipeline.create_comprehensive_report(
        symbol=symbol,
        forecast_date="2025-01-16"
    )

    # 4. Save pipeline for later use
    pipeline.save_pipeline_state("aapl_pipeline.pkl")

    # Display results
    print("Forecasts:")
    for tau, forecast in report['forecasts'].items():
        print(f"{tau}: {forecast:.4f}")

    # Show plots
    for plot_name, fig in report['plots'].items():
        if hasattr(fig, 'show'):  # Plotly figure
            fig.show()
        else:  # Matplotlib figure
            plt.show()
    '''

    print(demo_code)
    print("\n=== Next Steps ===")
    print("1. Replace API keys with your actual keys")
    print("2. Choose a symbol to analyze")
    print("3. Run the pipeline step by step")
    print("4. Examine the generated plots and forecasts")

# Print usage instructions
print("IV Forecasting System Ready!")
print("To use, instantiate your own DataHandler with real data access and pass it to IVForecastingPipeline.")
print("See the quick_start_demo() function for example usage.")