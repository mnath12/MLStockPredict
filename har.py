#!/usr/bin/env python
"""
HAR-RV (Heterogeneous Autoregressive Realized Volatility) Forecasting System    v1.0  (June-2025)

Implementation of the HAR-RV model from Corsi (2009) for volatility forecasting.
The HAR-RV model captures volatility at daily, weekly, and monthly frequencies.

Model: RV_{t+1} = β₀ + β₁·RV_t + β₂·RV_t^(w) + β₃·RV_t^(m) + ε_{t+1}

Where:
- RV_t = daily realized volatility
- RV_t^(w) = weekly average realized volatility  
- RV_t^(m) = monthly average realized volatility

Installation:
pip install yfinance pandas numpy matplotlib statsmodels scikit-learn alpaca-py

Usage
-----
python har_rv_forecaster.py [TICKER] [START_DATE] [END_DATE]
  TICKER     : stock symbol (default AAPL)
  START_DATE : training start date YYYY-MM-DD (default 1 year ago)
  END_DATE   : training end date / forecast start YYYY-MM-DD (default 30 days ago)

Examples:
python har_rv_forecaster.py AAPL                           # Default dates
python har_rv_forecaster.py AAPL 2023-01-01 2024-06-01    # Train 2023-2024, forecast Jun-today
python har_rv_forecaster.py TSLA 2024-01-01 2024-11-01    # Train 2024, forecast Nov-today
"""

from __future__ import annotations
import os, sys, time, warnings
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

# Alpaca imports (optional)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

warnings.filterwarnings("ignore")
np.random.seed(42)

# Suppress Intel MKL warnings
os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# ═══════════════════════════ DATA ACQUISITION ═══════════════════════════════

def download_prices_yfinance(ticker: str, start: str, end: str, tries: int = 3) -> pd.Series:
    """Download daily prices using yfinance with retry logic."""
    for attempt in range(tries):
        try:
            print(f"  📡 yfinance attempt {attempt + 1}/{tries}...")
            
            if attempt > 0:
                time.sleep(2 ** attempt)
            
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            prices = data['Close']
            prices.name = 'close'
            
            print(f"  ✅ Downloaded {len(prices)} days of data")
            return prices.dropna()
            
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} failed: {e}")
            if attempt == tries - 1:
                raise RuntimeError(f"Failed to download data for {ticker}: {e}")

def download_prices_alpaca(ticker: str, start: str, end: str, tries: int = 3) -> pd.Series:
    """Download prices using Alpaca API with fallback to yfinance."""
    if not ALPACA_AVAILABLE:
        print("📡 Alpaca SDK not available, using yfinance...")
        return download_prices_yfinance(ticker, start, end, tries)
    
    key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    
    if not (key and secret):
        print("📡 No Alpaca credentials, using yfinance...")
        return download_prices_yfinance(ticker, start, end, tries)
    
    print("📡 Attempting Alpaca download...")
    
    try:
        alpaca = StockHistoricalDataClient(key, secret)
        
        for attempt in range(tries):
            try:
                print(f"  🔄 Alpaca attempt {attempt + 1}/{tries}...")
                
                req = StockBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=TimeFrame.Day,
                    start=pd.Timestamp(start),
                    end=pd.Timestamp(end),
                    adjustment=Adjustment.ALL
                )
                
                bars = alpaca.get_stock_bars(req).df
                
                if bars.empty:
                    raise ValueError(f"No data returned from Alpaca")
                
                prices = bars.droplevel(0)["close"]
                prices.name = 'close'
                
                print(f"  ✅ Alpaca: Downloaded {len(prices)} days")
                return prices.dropna()
                
            except Exception as e:
                print(f"  ❌ Alpaca attempt {attempt + 1} failed: {e}")
                if attempt < tries - 1:
                    time.sleep(2 ** attempt)
                
        print("⚠️  Alpaca failed, falling back to yfinance...")
        return download_prices_yfinance(ticker, start, end, tries)
        
    except Exception as e:
        print(f"⚠️  Alpaca error ({e}), using yfinance...")
        return download_prices_yfinance(ticker, start, end, tries)

# ═══════════════════════════ REALIZED VOLATILITY CALCULATION ═══════════════════════════════

def calculate_realized_volatility(prices: pd.Series, method: str = 'log_returns') -> pd.Series:
    """
    Calculate daily realized volatility from price data.
    
    For daily data, we approximate realized volatility using daily returns.
    In practice, this would use high-frequency intraday data.
    """
    if method == 'log_returns':
        # Standard approach: daily log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        rv = log_returns.abs() * np.sqrt(np.pi / 2) * 100  # Convert to percentage, adjust for bias
        
    elif method == 'squared_returns':
        # Alternative: squared log returns (closer to theoretical RV)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        rv = (log_returns ** 2) * 100  # Convert to percentage
        
    elif method == 'parkinson':
        # Parkinson estimator (requires high/low data)
        if hasattr(prices, 'index') and 'High' in str(type(prices)):
            # This would work with OHLC data
            pass
        else:
            # Fallback to log_returns
            return calculate_realized_volatility(prices, 'log_returns')
            
    else:
        raise ValueError(f"Unknown method: {method}")
    
    rv.name = 'realized_vol'
    return rv

def calculate_har_features(rv_series: pd.Series, max_lag: int = 22) -> pd.DataFrame:
    """
    Calculate HAR features: daily, weekly (5-day), and monthly (22-day) averages.
    
    Args:
        rv_series: Daily realized volatility series
        max_lag: Maximum lag for monthly average (default 22 trading days ≈ 1 month)
    
    Returns:
        DataFrame with HAR features
    """
    data = pd.DataFrame(index=rv_series.index)
    
    # Current realized volatility (RV_t)
    data['rv_daily'] = rv_series
    
    # Weekly average (RV_t^(w)) - average of last 5 days including today
    data['rv_weekly'] = rv_series.rolling(window=5, min_periods=1).mean()
    
    # Monthly average (RV_t^(m)) - average of last 22 days including today  
    data['rv_monthly'] = rv_series.rolling(window=22, min_periods=1).mean()
    
    # Additional features for enhanced HAR models
    
    # Lag terms
    data['rv_lag1'] = rv_series.shift(1)
    data['rv_lag2'] = rv_series.shift(2)
    
    # Weekly lag
    data['rv_weekly_lag1'] = data['rv_weekly'].shift(1)
    
    # Monthly lag  
    data['rv_monthly_lag1'] = data['rv_monthly'].shift(1)
    
    # Volatility of volatility (RV momentum)
    data['rv_vol'] = rv_series.rolling(window=5).std()
    
    # Jump component (difference between current and average)
    data['rv_jump'] = rv_series - data['rv_weekly']
    
    # Persistence measures
    data['rv_mean_reversion'] = rv_series - data['rv_monthly']
    
    # Regime indicators
    data['rv_regime_high'] = (rv_series > rv_series.rolling(window=60).quantile(0.75)).astype(int)
    data['rv_regime_low'] = (rv_series < rv_series.rolling(window=60).quantile(0.25)).astype(int)
    
    # Seasonal effects (day of week, if needed)
    data['day_of_week'] = data.index.dayofweek
    data['monday'] = (data['day_of_week'] == 0).astype(int)
    data['friday'] = (data['day_of_week'] == 4).astype(int)
    
    return data

# ═══════════════════════════ HAR-RV MODEL IMPLEMENTATIONS ═══════════════════════════════

class BaseHARModel:
    """Base class for HAR-RV models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None

class HARLinear(BaseHARModel):
    """Standard HAR-RV model using OLS regression."""
    
    def __init__(self):
        super().__init__("HAR-Linear")
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Basic HAR features: daily, weekly, monthly
        features = ['rv_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1']
        X_har = X[features].dropna()
        y_har = y.reindex(X_har.index).dropna()
        
        # Add constant
        X_har = sm.add_constant(X_har)
        
        # Fit OLS model
        self.model = sm.OLS(y_har, X_har).fit()
        self.fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        features = ['rv_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1']
        X_pred = X[features].dropna()
        X_pred = sm.add_constant(X_pred)
        
        return self.model.predict(X_pred)
    
    def get_coefficients(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        return dict(zip(self.model.params.index, self.model.params.values))

class HARRidge(BaseHARModel):
    """HAR-RV with Ridge regularization for extended feature set."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("HAR-Ridge")
        self.alpha = alpha
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Extended feature set
        features = [
            'rv_lag1', 'rv_lag2', 'rv_weekly_lag1', 'rv_monthly_lag1',
            'rv_vol', 'rv_jump', 'rv_mean_reversion',
            'rv_regime_high', 'rv_regime_low', 'monday', 'friday'
        ]
        
        # Use available features
        available_features = [f for f in features if f in X.columns]
        X_har = X[available_features].dropna()
        y_har = y.reindex(X_har.index).dropna()
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_har, y_har)
        self.features = available_features
        self.fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_pred = X[self.features].dropna()
        return self.model.predict(X_pred)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        return dict(zip(self.features, self.model.coef_))

class HARRandomForest(BaseHARModel):
    """HAR-RV using Random Forest for non-linear relationships."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("HAR-RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Use all available features
        X_clean = X.dropna()
        y_clean = y.reindex(X_clean.index).dropna()
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X_clean, y_clean)
        self.features = X_clean.columns.tolist()
        self.fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_pred = X[self.features].dropna()
        return self.model.predict(X_pred)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        return dict(zip(self.features, self.model.feature_importances_))

class HAREnsemble(BaseHARModel):
    """Ensemble of multiple HAR models."""
    
    def __init__(self, models: Optional[list] = None, weights: Optional[list] = None):
        super().__init__("HAR-Ensemble")
        
        if models is None:
            self.models = [
                HARLinear(),
                HARRidge(alpha=0.1),
                HARRidge(alpha=1.0),
                HARRandomForest(n_estimators=50, max_depth=8)
            ]
        else:
            self.models = models
            
        self.weights = weights or [0.3, 0.3, 0.2, 0.2]  # Equal-ish weighting
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        print(f"    🔧 Fitting {len(self.models)} ensemble models...")
        
        for i, model in enumerate(self.models):
            try:
                model.fit(X, y)
                print(f"      ✅ {model.name} fitted successfully")
            except Exception as e:
                print(f"      ❌ {model.name} failed: {e}")
                self.weights[i] = 0  # Zero weight for failed models
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        weights = []
        
        for model, weight in zip(self.models, self.weights):
            if weight > 0 and model.fitted:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(weight)
                except:
                    continue
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += pred * weight
            
        return weighted_pred

# ═══════════════════════════ HAR-RV FORECASTING SYSTEM ═══════════════════════════════

class HARForecaster:
    """HAR-RV based volatility forecasting system."""
    
    def __init__(
        self, 
        model_type: str = 'ensemble',
        refit_frequency: int = 5,
        min_train_size: int = 100
    ):
        self.model_type = model_type
        self.refit_frequency = refit_frequency
        self.min_train_size = min_train_size
        
        # Initialize model
        if model_type == 'linear':
            self.base_model = HARLinear()
        elif model_type == 'ridge':
            self.base_model = HARRidge()
        elif model_type == 'rf':
            self.base_model = HARRandomForest()
        elif model_type == 'ensemble':
            self.base_model = HAREnsemble()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forecast(
        self, 
        prices: pd.Series, 
        train_start: str, 
        train_end: str
    ) -> Dict[str, Any]:
        """
        Perform HAR-RV volatility forecasting.
        
        Args:
            prices: Price series
            train_start: Training start date
            train_end: Training end date (forecast start)
            
        Returns:
            Dictionary with forecasts and performance metrics
        """
        print(f"🎯 HAR-RV Training period: {train_start} to {train_end}")
        
        # Calculate realized volatility
        rv_series = calculate_realized_volatility(prices, method='log_returns')
        print(f"📊 Realized volatility calculated: {len(rv_series)} observations")
        
        # Calculate HAR features
        har_features = calculate_har_features(rv_series)
        print(f"🔧 HAR features calculated: {list(har_features.columns)}")
        
        # Get training data
        train_mask = (har_features.index >= train_start) & (har_features.index <= train_end)
        train_data = har_features[train_mask].dropna()
        
        if len(train_data) < self.min_train_size:
            raise ValueError(f"Insufficient training data: {len(train_data)} < {self.min_train_size}")
        
        print(f"📈 Training data: {len(train_data)} observations")
        
        # Get forecast dates
        forecast_dates = har_features.index[har_features.index > train_end]
        
        if len(forecast_dates) == 0:
            raise ValueError("No forecast dates available")
        
        print(f"🔮 Forecasting {len(forecast_dates)} days: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
        
        # Rolling forecast
        results = {
            'dates': [], 'vol_forecasts': [], 'actual_vols': [],
            'model_performance': [], 'feature_importance': []
        }
        
        days_since_refit = 0
        current_model = None
        
        for i, forecast_date in enumerate(forecast_dates):
            try:
                # Expanding window for training
                current_train_end_idx = har_features.index.get_loc(forecast_date) - 1
                if current_train_end_idx < 0:
                    continue
                
                current_train_end = har_features.index[current_train_end_idx]
                expanded_train = har_features[
                    (har_features.index >= train_start) & 
                    (har_features.index <= current_train_end)
                ].dropna()
                
                if len(expanded_train) < self.min_train_size:
                    continue
                
                # Refit model periodically
                if current_model is None or days_since_refit >= self.refit_frequency:
                    print(f"  🔄 Refitting HAR model for {forecast_date.date()}...")
                    
                    # Prepare training data
                    X_train = expanded_train.drop('rv_daily', axis=1)
                    y_train = expanded_train['rv_daily']
                    
                    # Fit model
                    current_model = type(self.base_model)() if hasattr(self.base_model, '__class__') else self.base_model
                    if self.model_type == 'ensemble':
                        current_model = HAREnsemble()
                        
                    current_model.fit(X_train, y_train)
                    days_since_refit = 0
                    
                    # Get feature importance
                    importance = current_model.get_feature_importance()
                    if importance:
                        top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                        print(f"    📋 Top features: {top_features}")
                else:
                    days_since_refit += 1
                
                # Make forecast
                if current_model and current_model.fitted:
                    # Prepare forecast features
                    forecast_features = har_features.loc[current_train_end:current_train_end].drop('rv_daily', axis=1)
                    
                    if not forecast_features.empty:
                        vol_forecast = current_model.predict(forecast_features)[0]
                        
                        # Get actual volatility
                        actual_vol = har_features.loc[forecast_date, 'rv_daily']
                        
                        # Store results
                        results['dates'].append(forecast_date)
                        results['vol_forecasts'].append(max(0.1, vol_forecast))  # Ensure positive
                        results['actual_vols'].append(actual_vol)
                        
                        # Store model info
                        importance = current_model.get_feature_importance() or {}
                        results['feature_importance'].append(importance)
                        
                        # Periodic progress
                        if i % 10 == 0:
                            error = abs(vol_forecast - actual_vol)
                            print(f"    📊 {forecast_date.date()}: Forecast={vol_forecast:.2f}%, Actual={actual_vol:.2f}%, Error={error:.2f}%")
                
            except Exception as e:
                print(f"  ❌ Error forecasting {forecast_date.date()}: {e}")
                continue
        
        # Process results
        if len(results['dates']) == 0:
            return {'forecasts': pd.DataFrame(), 'summary': {}, 'metadata': {}}
        
        df = pd.DataFrame({
            'vol_forecast': results['vol_forecasts'],
            'actual_vol': results['actual_vols']
        }, index=results['dates'])
        
        # Calculate errors
        df['vol_error'] = df['actual_vol'] - df['vol_forecast']
        df['vol_error_pct'] = (df['vol_error'] / df['actual_vol']) * 100
        df['vol_error_abs'] = np.abs(df['vol_error'])
        
        # Summary statistics
        summary = self._calculate_summary(df)
        
        metadata = {
            'train_start': train_start,
            'train_end': train_end,
            'forecast_start': forecast_dates[0].strftime('%Y-%m-%d'),
            'forecast_end': forecast_dates[-1].strftime('%Y-%m-%d'),
            'n_forecasts': len(df),
            'model_type': self.model_type,
            'refit_frequency': self.refit_frequency,
            'feature_importance': results['feature_importance'][-1] if results['feature_importance'] else {}
        }
        
        return {'forecasts': df, 'summary': summary, 'metadata': metadata}
    
    def _calculate_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for HAR-RV forecasts."""
        if len(df) == 0:
            return {}
        
        # Basic metrics
        mae = mean_absolute_error(df['actual_vol'], df['vol_forecast'])
        mse = mean_squared_error(df['actual_vol'], df['vol_forecast'])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(df['vol_error_pct']))
        
        # R-squared
        r2 = r2_score(df['actual_vol'], df['vol_forecast'])
        
        # Hit rate (within 20% tolerance)
        hit_rate = np.mean(np.abs(df['vol_error_pct']) <= 20) * 100
        
        # Persistence (forecast vs random walk)
        random_walk_mse = mean_squared_error(df['actual_vol'], df['actual_vol'].shift(1).fillna(df['actual_vol'].mean()))
        persistence_ratio = mse / random_walk_mse if random_walk_mse > 0 else 1
        
        # VOLATILITY DIRECTIONAL ACCURACY
        if len(df) > 1:
            # Calculate changes in actual and forecasted volatility
            actual_vol_changes = df['actual_vol'].diff().dropna()
            forecast_vol_changes = df['vol_forecast'].diff().dropna()
            
            # Align the series (in case of different lengths)
            min_len = min(len(actual_vol_changes), len(forecast_vol_changes))
            actual_vol_changes = actual_vol_changes.iloc[:min_len]
            forecast_vol_changes = forecast_vol_changes.iloc[:min_len]
            
            if len(actual_vol_changes) > 0:
                # Direction: 1 for increase, -1 for decrease, 0 for no change
                actual_vol_direction = np.sign(actual_vol_changes.values)
                forecast_vol_direction = np.sign(forecast_vol_changes.values)
                
                # Overall directional accuracy
                vol_directional_accuracy = np.mean(actual_vol_direction == forecast_vol_direction) * 100
                
                # Count specific directions
                increase_mask = actual_vol_direction > 0
                decrease_mask = actual_vol_direction < 0
                nochange_mask = actual_vol_direction == 0
                
                # Accuracy for increases
                if np.sum(increase_mask) > 0:
                    vol_increase_accuracy = np.mean(
                        actual_vol_direction[increase_mask] == forecast_vol_direction[increase_mask]
                    ) * 100
                    vol_increase_days = np.sum(increase_mask)
                    vol_increase_correct = np.sum(
                        actual_vol_direction[increase_mask] == forecast_vol_direction[increase_mask]
                    )
                else:
                    vol_increase_accuracy = 0
                    vol_increase_days = 0
                    vol_increase_correct = 0
                
                # Accuracy for decreases
                if np.sum(decrease_mask) > 0:
                    vol_decrease_accuracy = np.mean(
                        actual_vol_direction[decrease_mask] == forecast_vol_direction[decrease_mask]
                    ) * 100
                    vol_decrease_days = np.sum(decrease_mask)
                    vol_decrease_correct = np.sum(
                        actual_vol_direction[decrease_mask] == forecast_vol_direction[decrease_mask]
                    )
                else:
                    vol_decrease_accuracy = 0
                    vol_decrease_days = 0
                    vol_decrease_correct = 0
                
                # Total correct predictions
                vol_total_correct = np.sum(actual_vol_direction == forecast_vol_direction)
                vol_total_days = len(actual_vol_direction)
                
            else:
                vol_directional_accuracy = 0
                vol_increase_accuracy = 0
                vol_decrease_accuracy = 0
                vol_increase_days = 0
                vol_decrease_days = 0
                vol_increase_correct = 0
                vol_decrease_correct = 0
                vol_total_correct = 0
                vol_total_days = 0
        else:
            vol_directional_accuracy = 0
            vol_increase_accuracy = 0
            vol_decrease_accuracy = 0
            vol_increase_days = 0
            vol_decrease_days = 0
            vol_increase_correct = 0
            vol_decrease_correct = 0
            vol_total_correct = 0
            vol_total_days = 0
        
        return {
            'vol_mae': mae,
            'vol_mse': mse,
            'vol_rmse': rmse,
            'vol_mape': mape,
            'vol_r2': r2,
            'vol_hit_rate': hit_rate,
            'persistence_ratio': persistence_ratio,
            # Volatility directional metrics
            'vol_directional_accuracy': vol_directional_accuracy,
            'vol_increase_accuracy': vol_increase_accuracy,
            'vol_decrease_accuracy': vol_decrease_accuracy,
            'vol_increase_days': vol_increase_days,
            'vol_decrease_days': vol_decrease_days,
            'vol_increase_correct': vol_increase_correct,
            'vol_decrease_correct': vol_decrease_correct,
            'vol_total_correct': vol_total_correct,
            'vol_total_days': vol_total_days
        }

# ═══════════════════════════ VISUALIZATION ═══════════════════════════════

def plot_har_results(results: Dict[str, Any], ticker: str):
    """Plot HAR-RV forecast results."""
    df = results['forecasts']
    metadata = results['metadata']
    
    if df.empty:
        print("❌ No data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Volatility forecasts vs actual
    axes[0].plot(df.index, df['actual_vol'], 'b-', label='Actual Volatility', linewidth=2)
    axes[0].plot(df.index, df['vol_forecast'], 'r--', label='HAR-RV Forecast', linewidth=2, alpha=0.8)
    axes[0].set_title(f'{ticker} - HAR-RV Volatility Forecasts ({metadata["forecast_start"]} to {metadata["forecast_end"]})')
    axes[0].set_ylabel('Realized Volatility (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Forecast errors
    axes[1].plot(df.index, df['vol_error_pct'], 'g-', alpha=0.7, label='Forecast Error (%)')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.3, label='±20% tolerance')
    axes[1].axhline(y=-20, color='red', linestyle='--', alpha=0.3)
    axes[1].set_title(f'{ticker} - HAR-RV Forecast Errors')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    summary = results['summary']
    print(f"\n📊 {ticker} HAR-RV Forecast Results ({metadata['model_type'].upper()})")
    print("=" * 60)
    print(f"Training: {metadata['train_start']} to {metadata['train_end']}")
    print(f"Forecasting: {metadata['forecast_start']} to {metadata['forecast_end']} ({metadata['n_forecasts']} days)")
    
    if summary:
        print(f"\nVolatility Forecast Performance:")
        print(f"  MAE: {summary['vol_mae']:.3f}%")
        print(f"  RMSE: {summary['vol_rmse']:.3f}%") 
        print(f"  MAPE: {summary['vol_mape']:.2f}%")
        print(f"  R²: {summary['vol_r2']:.3f}")
        print(f"  Hit Rate (±20%): {summary['vol_hit_rate']:.1f}%")
        print(f"  Persistence Ratio: {summary['persistence_ratio']:.3f} {'✅' if summary['persistence_ratio'] < 1 else '⚠️'}")
        
        # Volatility directional accuracy
        print(f"\nVolatility Direction Prediction:")
        print(f"  Overall Accuracy: {summary['vol_directional_accuracy']:.1f}% ({summary['vol_total_correct']:.0f}/{summary['vol_total_days']:.0f} days)")
        print(f"  Volatility Increases: {summary['vol_increase_accuracy']:.1f}% ({summary['vol_increase_correct']:.0f}/{summary['vol_increase_days']:.0f} days)")
        print(f"  Volatility Decreases: {summary['vol_decrease_accuracy']:.1f}% ({summary['vol_decrease_correct']:.0f}/{summary['vol_decrease_days']:.0f} days)")
        
        # Trading strategy implications
        if summary['vol_directional_accuracy'] > 60:
            print(f"  📈 Strong directional signal - suitable for volatility trading strategies")
        elif summary['vol_directional_accuracy'] > 50:
            print(f"  📊 Moderate directional signal - can be combined with other indicators")
        else:
            print(f"  ⚠️  Weak directional signal - focus on magnitude forecasts")
    
    # Feature importance
    if 'feature_importance' in metadata and metadata['feature_importance']:
        print(f"\nTop Feature Importance ({metadata['model_type']}):")
        importance = metadata['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, importance_val in sorted_features[:8]:
            print(f"  {feature}: {importance_val:.4f}")

# ═══════════════════════════ MAIN EXECUTION ═══════════════════════════════

def main():
    """Main execution for HAR-RV forecasting."""
    
    # Parse command line arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    if len(sys.argv) > 2:
        train_start = sys.argv[2]
    else:
        # Default: 1 year ago
        train_start = (datetime.now() - timedelta(days=8*365)).strftime('%Y-%m-%d')
    
    if len(sys.argv) > 3:
        train_end = sys.argv[3]
    else:
        # Default: 30 days ago (gives us 30 days to forecast)
        train_end = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
    
    print(f"🚀 HAR-RV Volatility Forecasting for {ticker}")
    print(f"📅 Training: {train_start} to {train_end}")
    print(f"🎯 Will forecast volatility from {train_end} to today")
    
    try:
        # Download data
        data_start = (datetime.strptime(train_start, '%Y-%m-%d') - timedelta(days=200)).strftime('%Y-%m-%d')
        data_end = datetime.now().strftime('%Y-%m-%d')
        
        prices = download_prices_alpaca(ticker, data_start, data_end)
        print(f"📈 Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Test different HAR models
        models_to_test = ['linear', 'ridge', 'rf', 'ensemble']
        
        print(f"\n🔬 Testing {len(models_to_test)} HAR-RV model variants...")
        
        all_results = {}
        
        for model_type in models_to_test:
            print(f"\n{'='*50}")
            print(f"🧪 Testing HAR-{model_type.upper()} Model")
            print(f"{'='*50}")
            
            try:
                # Initialize forecaster
                forecaster = HARForecaster(
                    model_type=model_type,
                    refit_frequency=5,
                    min_train_size=60
                )
                
                # Run forecast
                results = forecaster.forecast(prices, train_start, train_end)
                
                if results['forecasts'].empty:
                    print(f"❌ No forecasts generated for {model_type}")
                    continue
                
                all_results[model_type] = results
                
                # Quick performance summary
                summary = results['summary']
                if summary:
                    print(f"\n📊 {model_type.upper()} Performance Summary:")
                    print(f"  RMSE: {summary['vol_rmse']:.3f}%")
                    print(f"  MAPE: {summary['vol_mape']:.2f}%")
                    print(f"  R²: {summary['vol_r2']:.3f}")
                    print(f"  Hit Rate: {summary['vol_hit_rate']:.1f}%")
                    print(f"  Persistence: {summary['persistence_ratio']:.3f}")
                    print(f"  Vol Direction: {summary['vol_directional_accuracy']:.1f}% ({summary['vol_total_correct']:.0f}/{summary['vol_total_days']:.0f})")
                    print(f"    ↗️ Increases: {summary['vol_increase_accuracy']:.1f}% ({summary['vol_increase_correct']:.0f}/{summary['vol_increase_days']:.0f})")
                    print(f"    ↘️ Decreases: {summary['vol_decrease_accuracy']:.1f}% ({summary['vol_decrease_correct']:.0f}/{summary['vol_decrease_days']:.0f})")
                
            except Exception as e:
                print(f"❌ Error with {model_type} model: {e}")
                continue
        
        # Compare models and select best
        if all_results:
            print(f"\n{'='*60}")
            print("🏆 MODEL COMPARISON RESULTS")
            print(f"{'='*60}")
            
            comparison_df = pd.DataFrame()
            for model_name, results in all_results.items():
                if results['summary']:
                    comparison_df[model_name] = pd.Series(results['summary'])
            
            if not comparison_df.empty:
                print("\nPerformance Metrics Comparison:")
                print(comparison_df.round(3))
                
                # Rank models by multiple criteria
                ranking_metrics = ['vol_rmse', 'vol_mape', 'persistence_ratio']  # Lower is better
                ranking_scores = {}
                
                for metric in ranking_metrics:
                    if metric in comparison_df.index:
                        ranks = comparison_df.loc[metric].rank(ascending=True)
                        for model in ranks.index:
                            ranking_scores[model] = ranking_scores.get(model, 0) + ranks[model]
                
                # Add R² ranking (higher is better)
                if 'vol_r2' in comparison_df.index:
                    r2_ranks = comparison_df.loc['vol_r2'].rank(ascending=False)
                    for model in r2_ranks.index:
                        ranking_scores[model] = ranking_scores.get(model, 0) + r2_ranks[model]
                
                # Add volatility directional accuracy ranking (higher is better)
                if 'vol_directional_accuracy' in comparison_df.index:
                    dir_ranks = comparison_df.loc['vol_directional_accuracy'].rank(ascending=False)
                    for model in dir_ranks.index:
                        ranking_scores[model] = ranking_scores.get(model, 0) + dir_ranks[model]
                
                # Best model has lowest total rank
                best_model = min(ranking_scores.items(), key=lambda x: x[1])[0]
                print(f"\n🥇 Best Model: HAR-{best_model.upper()}")
                print(f"   Overall Ranking Score: {ranking_scores[best_model]:.1f}")
                
                # Plot results for best model
                print(f"\n📊 Plotting results for best model: {best_model}")
                plot_har_results(all_results[best_model], ticker)
                
                # Save results
                best_results = all_results[best_model]
                filename = f"{ticker}_har_rv_{best_model}_{train_start}_{train_end}.csv"
                best_results['forecasts'].to_csv(filename)
                print(f"💾 Best model results saved to {filename}")
                
                # Additional analysis: Compare with simple benchmark
                print(f"\n📈 Additional Analysis:")
                benchmark_comparison(best_results, ticker)
                
            else:
                print("❌ No valid results to compare")
        else:
            print("❌ No successful model runs")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def benchmark_comparison(results: Dict[str, Any], ticker: str):
    """Compare HAR-RV results with simple benchmarks."""
    df = results['forecasts']
    
    if df.empty:
        return
    
    print("Benchmark Comparison:")
    print("-" * 40)
    
    actual = df['actual_vol'].values
    har_forecast = df['vol_forecast'].values
    
    # Benchmark 1: Random walk (previous day's volatility)
    rw_forecast = np.concatenate([[actual[0]], actual[:-1]])
    rw_rmse = np.sqrt(mean_squared_error(actual, rw_forecast))
    
    # Benchmark 2: Historical average
    hist_avg = np.full_like(actual, np.mean(actual))
    hist_rmse = np.sqrt(mean_squared_error(actual, hist_avg))
    
    # Benchmark 3: Exponential smoothing
    alpha = 0.3
    exp_smooth = [actual[0]]
    for i in range(1, len(actual)):
        exp_smooth.append(alpha * actual[i-1] + (1-alpha) * exp_smooth[-1])
    exp_rmse = np.sqrt(mean_squared_error(actual, exp_smooth))
    
    # HAR-RV performance
    har_rmse = np.sqrt(mean_squared_error(actual, har_forecast))
    
    print(f"RMSE Comparison:")
    print(f"  HAR-RV RMSE:           {har_rmse:.3f}%")
    print(f"  Random Walk RMSE:      {rw_rmse:.3f}% ({'✅' if har_rmse < rw_rmse else '❌'})")
    print(f"  Historical Avg RMSE:   {hist_rmse:.3f}% ({'✅' if har_rmse < hist_rmse else '❌'})")
    print(f"  Exp. Smoothing RMSE:   {exp_rmse:.3f}% ({'✅' if har_rmse < exp_rmse else '❌'})")
    
    # Directional accuracy comparison
    if len(actual) > 1:
        actual_changes = np.diff(actual)
        har_changes = np.diff(har_forecast)
        rw_changes = np.diff(rw_forecast)
        exp_changes = np.diff(exp_smooth)
        
        # Calculate directional accuracy for each method
        har_dir_acc = np.mean(np.sign(actual_changes) == np.sign(har_changes)) * 100
        rw_dir_acc = np.mean(np.sign(actual_changes) == np.sign(rw_changes)) * 100
        exp_dir_acc = np.mean(np.sign(actual_changes) == np.sign(exp_changes)) * 100
        
        print(f"\nDirectional Accuracy Comparison:")
        print(f"  HAR-RV Direction:      {har_dir_acc:.1f}%")
        print(f"  Random Walk Direction: {rw_dir_acc:.1f}% ({'✅' if har_dir_acc > rw_dir_acc else '❌'})")
        print(f"  Exp. Smooth Direction: {exp_dir_acc:.1f}% ({'✅' if har_dir_acc > exp_dir_acc else '❌'})")
        
        # Random baseline (should be ~50%)
        print(f"  Random Baseline:       ~50.0%")
    
    # Improvement percentages
    if rw_rmse > 0:
        rw_improvement = ((rw_rmse - har_rmse) / rw_rmse) * 100
        print(f"\nRMSE Improvement over Random Walk: {rw_improvement:.1f}%")
    
    if exp_rmse > 0:
        exp_improvement = ((exp_rmse - har_rmse) / exp_rmse) * 100
        print(f"RMSE Improvement over Exp. Smoothing: {exp_improvement:.1f}%")

if __name__ == "__main__":
    main()