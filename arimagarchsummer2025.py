#!/usr/bin/env python
"""
Simplified ARIMA–GARCH Rolling Forecast System           v3.0  (June-2025)

Simplified Logic:
- start_date to end_date: Training data period
- end_date to today: Forecast period (rolling daily forecasts)
- Automatically handles weekends and holidays
- Rolling window approach with expanding training data

Installation:
pip install yfinance pandas numpy matplotlib statsmodels arch pmdarima alpaca-py

Alpaca Setup (Optional):
export APCA_API_KEY_ID="your_key_here"
export APCA_API_SECRET_KEY="your_secret_here"

Usage
-----
python improved_arima_garch.py [TICKER] [START_DATE] [END_DATE]
  TICKER     : stock symbol (default AAPL)
  START_DATE : training start date YYYY-MM-DD (default 1 year ago)
  END_DATE   : training end date / forecast start YYYY-MM-DD (default 30 days ago)

Examples:
python improved_arima_garch.py AAPL                           # Default dates
python improved_arima_garch.py AAPL 2023-01-01 2024-06-01    # Train 2023-2024, forecast Jun-today
python improved_arima_garch.py TSLA 2024-01-01 2024-11-01    # Train 2024, forecast Nov-today
"""

from __future__ import annotations
import os, sys, time, warnings
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Alpaca imports (optional)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from arch import arch_model

warnings.filterwarnings("ignore")
np.random.seed(42)

# Suppress Intel MKL warnings
os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# ═══════════════════════════ DATA ACQUISITION ═══════════════════════════════

def download_prices_yfinance(
    ticker: str,
    start: str,
    end: str,
    tries: int = 3
) -> pd.Series:
    """Download daily prices using yfinance with retry logic."""
    for attempt in range(tries):
        try:
            print(f"  📡 yfinance attempt {attempt + 1}/{tries}...")
            
            if attempt > 0:
                time.sleep(2 ** attempt)  # Exponential backoff
            
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                auto_adjust=True,
                threads=False
            )
            
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

def download_prices_alpaca(
    ticker: str,
    start: str,
    end: str,
    tries: int = 3
) -> pd.Series:
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

# ═══════════════════════════ ENHANCED MODEL SELECTION ═══════════════════════════════

def advanced_arima_selection(returns: pd.Series, max_order: int = 4):
    """Enhanced ARIMA selection with multiple criteria and diagnostics."""
    best_model = None
    best_score = np.inf
    
    # Test stationarity
    adf_result = adfuller(returns.dropna())
    is_stationary = adf_result[1] < 0.05
    
    # Determine differencing order
    d_max = 0 if is_stationary else 2
    
    candidates = []
    
    # Grid search with multiple information criteria
    for p in range(max_order + 1):
        for d in range(d_max + 1):
            for q in range(max_order + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                    
                try:
                    model = ARIMA(returns, order=(p, d, q))
                    fitted = model.fit()
                    
                    # Multiple criteria for model selection
                    aic = fitted.aic
                    bic = fitted.bic
                    hqic = fitted.hqic
                    
                    # Residual diagnostics
                    residuals = fitted.resid
                    lb_stat = acorr_ljungbox(residuals, lags=10, return_df=True)['lb_pvalue'].iloc[-1]
                    
                    # Combined score (lower is better)
                    # Penalize models that fail residual tests
                    penalty = 100 if lb_stat < 0.05 else 0
                    combined_score = 0.5 * aic + 0.3 * bic + 0.2 * hqic + penalty
                    
                    candidates.append({
                        'order': (p, d, q),
                        'model': fitted,
                        'aic': aic,
                        'bic': bic,
                        'score': combined_score,
                        'lb_pvalue': lb_stat
                    })
                    
                except:
                    continue
    
    if candidates:
        # Select best model based on combined score
        best_candidate = min(candidates, key=lambda x: x['score'])
        return best_candidate['model']
    else:
        # Fallback to pmdarima
        try:
            auto_model = pm.auto_arima(
                returns, max_p=max_order, max_q=max_order, max_d=d_max,
                stepwise=False, approximation=False, 
                suppress_warnings=True, error_action='ignore'
            )
            return ARIMA(returns, order=auto_model.order).fit()
        except:
            return ARIMA(returns, order=(1, 0, 1)).fit()

def enhanced_garch_selection(residuals: pd.Series):
    """Enhanced GARCH model selection with multiple specifications."""
    models_to_try = [
        # Standard GARCH models
        {'vol': 'GARCH', 'p': 1, 'q': 1},
        {'vol': 'GARCH', 'p': 1, 'q': 2},
        {'vol': 'GARCH', 'p': 2, 'q': 1},
        {'vol': 'GARCH', 'p': 2, 'q': 2},
        
        # EGARCH for asymmetric effects
        {'vol': 'EGARCH', 'p': 1, 'q': 1},
        {'vol': 'EGARCH', 'p': 1, 'q': 2},
        
        # GJR-GARCH for leverage effects
        {'vol': 'GARCH', 'p': 1, 'q': 1, 'o': 1},
        {'vol': 'GARCH', 'p': 1, 'q': 2, 'o': 1},
        
        # ARCH models
        {'vol': 'ARCH', 'p': 1},
        {'vol': 'ARCH', 'p': 2},
        {'vol': 'ARCH', 'p': 3},
    ]
    
    best_model = None
    best_aic = np.inf
    
    for spec in models_to_try:
        try:
            # Build model specification
            if 'o' in spec:  # GJR-GARCH
                model = arch_model(
                    residuals, 
                    vol=spec['vol'], 
                    p=spec['p'], 
                    q=spec['q'], 
                    o=spec['o'],
                    mean='Zero', 
                    dist='normal'
                )
            else:
                model = arch_model(
                    residuals, 
                    vol=spec['vol'], 
                    p=spec['p'], 
                    q=spec.get('q', 0),
                    mean='Zero', 
                    dist='normal'
                )
            
            fitted = model.fit(disp='off', show_warning=False)
            
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                
        except:
            continue
    
    return best_model

# ═══════════════════════════ VOLATILITY FORECASTING ENHANCEMENTS ═══════════════════════════════

def calculate_volatility_features(returns: pd.Series, window: int = 20):
    """Calculate additional volatility features for forecasting."""
    features = {}
    
    if len(returns) < window:
        return features
    
    # Rolling statistics
    rolling_vol = returns.rolling(window).std()
    rolling_mean = returns.rolling(window).mean()
    
    # Volatility clustering measure
    vol_changes = rolling_vol.diff().abs()
    features['vol_clustering'] = vol_changes.rolling(5).mean().iloc[-1] if len(vol_changes) > 5 else 0
    
    # Skewness and kurtosis (for tail risk)
    features['skewness'] = returns.rolling(window).skew().iloc[-1]
    features['kurtosis'] = returns.rolling(window).kurt().iloc[-1]
    
    # Volatility momentum
    short_vol = returns.rolling(5).std().iloc[-1]
    long_vol = returns.rolling(window).std().iloc[-1]
    features['vol_momentum'] = short_vol / long_vol if long_vol > 0 else 1
    
    # Return magnitude clustering
    abs_returns = returns.abs()
    features['magnitude_clustering'] = abs_returns.rolling(5).mean().iloc[-1] / abs_returns.rolling(window).mean().iloc[-1]
    
    return features

def regime_aware_volatility_forecast(garch_fit, recent_returns: pd.Series, horizon: int = 1):
    """Advanced volatility forecasting with regime awareness and feature engineering."""
    forecasts = []
    weights = []
    
    # Get volatility features
    features = calculate_volatility_features(recent_returns, window=min(20, len(recent_returns)))
    
    # Method 1: Enhanced GARCH forecast
    if garch_fit is not None:
        try:
            garch_forecast = garch_fit.forecast(horizon=horizon, reindex=False)
            base_garch_vol = np.sqrt(garch_forecast.variance.iloc[0, 0])
            
            # Adjust GARCH forecast based on features
            adjustment = 1.0
            
            # Volatility clustering adjustment
            if 'vol_clustering' in features and features['vol_clustering'] > 0:
                clustering_factor = min(1.5, 1 + features['vol_clustering'])
                adjustment *= clustering_factor
            
            # Momentum adjustment
            if 'vol_momentum' in features:
                momentum_factor = min(1.3, max(0.8, features['vol_momentum']))
                adjustment *= momentum_factor
            
            adjusted_garch_vol = base_garch_vol * adjustment
            forecasts.append(adjusted_garch_vol)
            weights.append(0.35)
            
        except:
            pass
    
    # Method 2: Exponential smoothing with adaptive alpha
    if len(recent_returns) >= 5:
        # Adaptive smoothing parameter based on volatility regime
        regime, regime_mult = detect_volatility_regime(recent_returns)
        alpha = 0.3 if regime == 'high' else 0.1  # More responsive in high vol
        
        ewm_vol = recent_returns.ewm(alpha=alpha).std().iloc[-1]
        forecasts.append(ewm_vol)
        weights.append(0.25)
    
    # Method 3: LSTM-inspired recurrent volatility
    if len(recent_returns) >= 10:
        # Simple recurrent-style volatility forecast
        vol_series = recent_returns.rolling(3).std().dropna()
        if len(vol_series) >= 5:
            # Weighted average of recent volatilities with decay
            decay_weights = np.exp(-0.1 * np.arange(5))
            decay_weights = decay_weights / decay_weights.sum()
            
            recurrent_vol = np.sum(vol_series[-5:].values * decay_weights[:len(vol_series[-5:])])
            forecasts.append(recurrent_vol)
            weights.append(0.2)
    
    # Method 4: Regime-switching adjustment
    if len(recent_returns) >= 15:
        regime, regime_mult = detect_volatility_regime(recent_returns)
        
        # Historical volatility in same regime
        rolling_vol = recent_returns.rolling(5).std()
        
        if regime == 'high':
            # In high vol regime, look at recent high vol periods
            high_vol_periods = rolling_vol[rolling_vol > rolling_vol.quantile(0.75)]
            if len(high_vol_periods) > 0:
                regime_vol = high_vol_periods.mean() * 1.1  # Slight upward bias
                forecasts.append(regime_vol)
                weights.append(0.15)
        elif regime == 'low':
            # In low vol regime, look at recent low vol periods
            low_vol_periods = rolling_vol[rolling_vol < rolling_vol.quantile(0.25)]
            if len(low_vol_periods) > 0:
                regime_vol = low_vol_periods.mean() * 0.9  # Slight downward bias
                forecasts.append(regime_vol)
                weights.append(0.15)
    
    # Method 5: Volatility of volatility adjustment
    if len(recent_returns) >= 10:
        vol_series = recent_returns.rolling(3).std().dropna()
        if len(vol_series) >= 5:
            vol_of_vol = vol_series.rolling(5).std().iloc[-1]
            base_vol = vol_series.iloc[-1]
            
            # If volatility is very unstable, increase forecast
            vvol_adjusted = base_vol * (1 + vol_of_vol * 0.5)
            forecasts.append(vvol_adjusted)
            weights.append(0.05)
    
    # Combine forecasts
    if forecasts and weights:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_vol = np.sum(np.array(forecasts) * weights)
        
        # Final bounds and regime adjustment
        regime, regime_mult = detect_volatility_regime(recent_returns)
        ensemble_vol *= regime_mult
        
        # Ensure reasonable bounds
        ensemble_vol = max(0.1, min(15.0, ensemble_vol))
        
        return ensemble_vol
    else:
        # Ultimate fallback
        return max(0.1, recent_returns.std()) if len(recent_returns) > 1 else 1.0

def calculate_realized_volatility(returns: pd.Series, window: int = 5) -> float:
    """Calculate realized volatility using multiple methods."""
    if len(returns) < window:
        return returns.std()
    
    # Method 1: Simple rolling standard deviation
    rolling_std = returns.rolling(window=window, min_periods=1).std().iloc[-1]
    
    # Method 2: Exponentially weighted volatility (more responsive)
    ewm_std = returns.ewm(span=window).std().iloc[-1]
    
    # Method 3: High-frequency proxy (use absolute returns for daily data)
    recent_abs_returns = returns.abs().rolling(window=window, min_periods=1).mean().iloc[-1]
    
    # Combine methods (weighted average)
    realized_vol = 0.4 * rolling_std + 0.4 * ewm_std + 0.2 * recent_abs_returns * np.sqrt(np.pi/2)
    
    return realized_vol

def fit_garch_adaptive(residuals: pd.Series, max_lag: int = 2):
    """Fit GARCH model with adaptive lag selection."""
    best_model = None
    best_aic = np.inf
    
    # Try different GARCH specifications
    for p in range(1, max_lag + 1):
        for q in range(1, max_lag + 1):
            try:
                model = arch_model(
                    residuals, 
                    vol='GARCH', 
                    p=p, q=q,
                    mean='Zero', 
                    dist='normal'
                )
                fitted = model.fit(disp='off')
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    
            except:
                continue
    
    # If GARCH fails, try EGARCH (captures asymmetric volatility)
    if best_model is None:
        try:
            model = arch_model(
                residuals,
                vol='EGARCH',
                p=1, q=1,
                mean='Zero',
                dist='normal'
            )
            best_model = model.fit(disp='off')
        except:
            # Final fallback to simple ARCH
            try:
                model = arch_model(
                    residuals,
                    vol='ARCH',
                    p=1,
                    mean='Zero',
                    dist='normal'
                )
                best_model = model.fit(disp='off')
            except:
                best_model = None
    
    return best_model

def detect_volatility_regime(returns: pd.Series, lookback: int = 30):
    """Detect current volatility regime (low/normal/high)."""
    if len(returns) < lookback:
        return 'normal', 1.0
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=5).std()
    recent_vol = rolling_vol.iloc[-5:].mean()
    
    # Historical percentiles
    hist_vol = rolling_vol.dropna()
    if len(hist_vol) < 10:
        return 'normal', 1.0
    
    low_threshold = hist_vol.quantile(0.25)
    high_threshold = hist_vol.quantile(0.75)
    
    # Determine regime
    if recent_vol > high_threshold:
        regime = 'high'
        multiplier = 1.5  # Increase forecast in high vol regime
    elif recent_vol < low_threshold:
        regime = 'low'
        multiplier = 0.8  # Decrease forecast in low vol regime
    else:
        regime = 'normal'
        multiplier = 1.0
    
    return regime, multiplier

def forecast_volatility_ensemble(garch_fit, recent_returns: pd.Series, horizon: int = 1):
    """Ensemble volatility forecasting combining multiple approaches."""
    forecasts = []
    
    # Detect volatility regime
    regime, regime_multiplier = detect_volatility_regime(recent_returns)
    
    # Method 1: GARCH forecast
    if garch_fit is not None:
        try:
            garch_forecast = garch_fit.forecast(horizon=horizon, reindex=False)
            garch_vol = np.sqrt(garch_forecast.variance.iloc[0, 0])
            forecasts.append(('garch', garch_vol, 0.4))
        except:
            pass
    
    # Method 2: Exponential smoothing of recent volatility
    if len(recent_returns) >= 5:
        recent_vol = calculate_realized_volatility(recent_returns, window=5)
        ewm_vol = recent_returns.ewm(span=10).std().iloc[-1]
        forecasts.append(('ewm', ewm_vol, 0.3))
        forecasts.append(('recent', recent_vol, 0.2))
    
    # Method 3: Volatility persistence model
    if len(recent_returns) >= 10:
        vol_series = recent_returns.rolling(5).std().dropna()
        if len(vol_series) >= 3:
            # Auto-regressive volatility model (simple)
            recent_vols = vol_series[-3:].values
            persistence = np.corrcoef(recent_vols[:-1], recent_vols[1:])[0,1] if len(recent_vols) > 2 else 0.5
            persistence = max(0.1, min(0.9, persistence))  # Bound persistence
            
            ar_vol = persistence * recent_vols[-1] + (1 - persistence) * vol_series.mean()
            forecasts.append(('persistence', ar_vol, 0.1))
    
    # Combine forecasts using weights
    if forecasts:
        total_weight = sum(weight for _, _, weight in forecasts)
        ensemble_vol = sum(vol * weight for _, vol, weight in forecasts) / total_weight
        
        # Apply regime adjustment
        ensemble_vol *= regime_multiplier
        
        # Ensure reasonable bounds
        ensemble_vol = max(0.1, min(10.0, ensemble_vol))
        
        return ensemble_vol
    else:
        # Fallback to simple standard deviation
        fallback_vol = recent_returns.std() if len(recent_returns) > 1 else 1.0
        return max(0.1, fallback_vol * regime_multiplier)

# ═══════════════════════════ MODEL FITTING ═══════════════════════════════

def fit_arima_auto(returns: pd.Series) -> ARIMA:
    """Fit ARIMA model with automatic order selection."""
    try:
        auto_model = pm.auto_arima(
            returns,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=None,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        return ARIMA(returns, order=auto_model.order).fit()
    except:
        # Fallback to simple AR(1)
        try:
            return ARIMA(returns, order=(1, 0, 0)).fit()
        except:
            # Last resort: mean model
            return ARIMA(returns, order=(0, 0, 0)).fit()

# ═══════════════════════════ SIMPLIFIED FORECASTER ═══════════════════════════

class EnhancedForecaster:
    """Enhanced ARIMA-GARCH forecaster with advanced model selection."""
    
    def __init__(self, refit_frequency: int = 3, use_enhanced_selection: bool = True):
        self.refit_frequency = refit_frequency  # Refit more frequently
        self.use_enhanced_selection = use_enhanced_selection
        self.model_cache = {}  # Cache fitted models
    
    def forecast(
        self, 
        prices: pd.Series, 
        train_start: str, 
        train_end: str
    ) -> Dict[str, Any]:
        """Enhanced forecasting with better model selection."""
        print(f"🎯 Training period: {train_start} to {train_end}")
        
        # Convert to log returns (percentage for GARCH)
        log_prices = np.log(prices)
        returns_pct = 100 * log_prices.diff().dropna()
        
        # Get training data
        train_mask = (returns_pct.index >= train_start) & (returns_pct.index <= train_end)
        train_returns = returns_pct[train_mask]
        
        if len(train_returns) < 50:
            raise ValueError(f"Insufficient training data: {len(train_returns)} days")
        
        print(f"📊 Training data: {len(train_returns)} days")
        
        # Get forecast dates
        all_dates_after_train = returns_pct.index[returns_pct.index > train_end]
        forecast_dates = all_dates_after_train
        
        if len(forecast_dates) == 0:
            raise ValueError("No forecast dates available")
        
        print(f"🔮 Forecasting {len(forecast_dates)} days: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
        
        # Initialize storage
        results = {
            'dates': [], 'price_forecasts': [], 'vol_forecasts': [],
            'actual_prices': [], 'actual_returns': [], 'realized_vol': [],
            'model_info': []  # Track which models were used
        }
        
        # Enhanced rolling forecast loop
        arima_fit = None
        garch_fit = None
        days_since_refit = 0
        
        for i, forecast_date in enumerate(forecast_dates):
            try:
                # Expanding window
                current_train_end_idx = returns_pct.index.get_loc(forecast_date) - 1
                if current_train_end_idx < 0:
                    continue
                    
                current_train_end = returns_pct.index[current_train_end_idx]
                expanded_train = returns_pct[
                    (returns_pct.index >= train_start) & 
                    (returns_pct.index <= current_train_end)
                ]
                
                if len(expanded_train) < 50:
                    continue
                
                # Enhanced model refitting
                if arima_fit is None or days_since_refit >= self.refit_frequency:
                    print(f"  🔄 Enhanced refitting for {forecast_date.date()}...")
                    
                    if self.use_enhanced_selection:
                        arima_fit = advanced_arima_selection(expanded_train)
                        garch_fit = enhanced_garch_selection(arima_fit.resid)
                    else:
                        arima_fit = fit_arima_auto(expanded_train)
                        garch_fit = fit_garch_adaptive(arima_fit.resid)
                    
                    # Store model info
                    arima_order = getattr(arima_fit, 'model', {}).get('order', 'unknown')
                    garch_type = type(garch_fit.model).__name__ if garch_fit else 'none'
                    
                    days_since_refit = 0
                    print(f"    📋 ARIMA{arima_order}, GARCH: {garch_type}")
                else:
                    days_since_refit += 1
                
                # Enhanced forecasting
                if arima_fit is not None:
                    # Return forecast with confidence intervals
                    forecast_result = arima_fit.get_forecast(steps=1)
                    return_forecast = forecast_result.predicted_mean.iloc[0]
                    return_std = forecast_result.se_mean.iloc[0]
                    
                    # Enhanced volatility forecast
                    recent_returns_for_vol = expanded_train[-30:]  # More data for vol
                    vol_forecast = regime_aware_volatility_forecast(
                        garch_fit, 
                        recent_returns_for_vol,
                        horizon=1
                    )
                    
                    # Price forecast with uncertainty
                    prev_price = prices.loc[current_train_end]
                    price_forecast = prev_price * np.exp(return_forecast / 100.0)
                    
                    # Actual values
                    actual_price = prices.loc[forecast_date]
                    actual_return = returns_pct.loc[forecast_date]
                    
                    # Enhanced realized volatility
                    recent_actual_returns = returns_pct.loc[
                        (returns_pct.index <= forecast_date) & 
                        (returns_pct.index > forecast_date - timedelta(days=15))
                    ]
                    realized_vol = calculate_realized_volatility(recent_actual_returns, window=5)
                    
                    # Store results
                    results['dates'].append(forecast_date)
                    results['price_forecasts'].append(price_forecast)
                    results['vol_forecasts'].append(vol_forecast)
                    results['actual_prices'].append(actual_price)
                    results['actual_returns'].append(actual_return)
                    results['realized_vol'].append(realized_vol)
                    results['model_info'].append({
                        'arima_order': arima_order,
                        'garch_type': garch_type,
                        'return_std': return_std,
                        'vol_features': calculate_volatility_features(recent_returns_for_vol)
                    })
                    
                    # Periodic progress with enhanced metrics
                    if i % 10 == 0:
                        regime, _ = detect_volatility_regime(recent_returns_for_vol)
                        vol_features = calculate_volatility_features(recent_returns_for_vol)
                        clustering = vol_features.get('vol_clustering', 0)
                        momentum = vol_features.get('vol_momentum', 1)
                        
                        print(f"    📊 Date: {forecast_date.date()}")
                        print(f"        Vol regime: {regime}, Clustering: {clustering:.3f}, Momentum: {momentum:.3f}")
                        print(f"        Forecast vol: {vol_forecast:.2f}%, Realized: {realized_vol:.2f}%")
                
            except Exception as e:
                print(f"  ❌ Error forecasting {forecast_date.date()}: {e}")
                continue
        
        # Enhanced results processing
        if len(results['dates']) == 0:
            return {'forecasts': pd.DataFrame(), 'summary': {}, 'metadata': {}}
        
        df = pd.DataFrame({
            'price_forecast': results['price_forecasts'],
            'vol_forecast': results['vol_forecasts'],
            'actual_price': results['actual_prices'],
            'actual_return': results['actual_returns'],
            'realized_vol': results['realized_vol']
        }, index=results['dates'])
        
        # Calculate enhanced errors
        df['price_error'] = df['actual_price'] - df['price_forecast']
        df['price_error_pct'] = (df['price_error'] / df['actual_price']) * 100
        df['vol_error'] = df['realized_vol'] - df['vol_forecast']
        df['vol_error_pct'] = (df['vol_error'] / df['realized_vol']) * 100
        
        # Enhanced summary statistics
        summary = self._calculate_enhanced_summary(df, results['model_info'])
        
        metadata = {
            'train_start': train_start,
            'train_end': train_end,
            'forecast_start': forecast_dates[0].strftime('%Y-%m-%d'),
            'forecast_end': forecast_dates[-1].strftime('%Y-%m-%d'),
            'n_forecasts': len(df),
            'train_days': len(train_returns),
            'enhanced_selection': self.use_enhanced_selection,
            'refit_frequency': self.refit_frequency
        }
        
        return {'forecasts': df, 'summary': summary, 'metadata': metadata}
    
    def _calculate_enhanced_summary(self, df: pd.DataFrame, model_info: list) -> Dict[str, float]:
        """Calculate enhanced performance metrics."""
        if len(df) == 0:
            return {}
        
        # Basic metrics
        price_mae = np.mean(np.abs(df['price_error']))
        price_mse = np.mean(df['price_error'] ** 2)
        price_mape = np.mean(np.abs(df['price_error_pct']))
        
        vol_mae = np.mean(np.abs(df['vol_error']))
        vol_mse = np.mean(df['vol_error'] ** 2)
        vol_mape = np.mean(np.abs(df['vol_error_pct']))
        
        # Enhanced directional accuracy
        if len(df) > 1:
            actual_returns = df['actual_return'].values
            price_forecasts = df['price_forecast'].values
            
            actual_direction = np.sign(np.diff(actual_returns))
            forecast_direction = np.sign(np.diff(price_forecasts) / price_forecasts[:-1])
            
            if len(actual_direction) > 0 and len(forecast_direction) > 0:
                min_len = min(len(actual_direction), len(forecast_direction))
                directional_accuracy = np.mean(
                    actual_direction[:min_len] == forecast_direction[:min_len]
                ) * 100
            else:
                directional_accuracy = 0
        else:
            directional_accuracy = 0
        
        # Volatility-specific metrics
        vol_hit_rate = np.mean(
            (df['vol_forecast'] * 0.8 <= df['realized_vol']) & 
            (df['realized_vol'] <= df['vol_forecast'] * 1.2)
        ) * 100  # Within 20% tolerance
        
        # Risk-adjusted metrics
        price_errors = df['price_error_pct'].values
        sharpe_like = np.mean(price_errors) / np.std(price_errors) if np.std(price_errors) > 0 else 0
        
        return {
            'price_mae': price_mae,
            'price_mse': price_mse,
            'price_mape': price_mape,
            'vol_mae': vol_mae,
            'vol_mse': vol_mse,
            'vol_mape': vol_mape,
            'vol_hit_rate': vol_hit_rate,
            'directional_accuracy': directional_accuracy,
            'forecast_sharpe': -sharpe_like,  # Negative because we want small errors
            'model_changes': len([m for m in model_info if 'arima_order' in m])
        }
    
    def forecast(
        self, 
        prices: pd.Series, 
        train_start: str, 
        train_end: str
    ) -> Dict[str, Any]:
        """
        Train on [train_start, train_end], then forecast from train_end to latest data.
        
        Args:
            prices: Full price series
            train_start: Start of training period  
            train_end: End of training period (start of forecast period)
            
        Returns:
            Dictionary with forecasts and actuals
        """
        print(f"🎯 Training period: {train_start} to {train_end}")
        
        # Convert to log returns (percentage for GARCH)
        log_prices = np.log(prices)
        returns_pct = 100 * log_prices.diff().dropna()
        
        # Get training data
        train_mask = (returns_pct.index >= train_start) & (returns_pct.index <= train_end)
        train_returns = returns_pct[train_mask]
        
        if len(train_returns) < 50:
            raise ValueError(f"Insufficient training data: {len(train_returns)} days")
        
        print(f"📊 Training data: {len(train_returns)} days")
        
        # Get forecast dates (trading days after train_end)
        all_dates_after_train = returns_pct.index[returns_pct.index > train_end]
        forecast_dates = all_dates_after_train  # Only trading days in our data
        
        if len(forecast_dates) == 0:
            raise ValueError("No forecast dates available (train_end is too recent)")
        
        print(f"🔮 Forecasting {len(forecast_dates)} days: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
        
        # Initialize storage
        results = {
            'dates': [],
            'price_forecasts': [],
            'vol_forecasts': [],
            'actual_prices': [],
            'actual_returns': [],
            'realized_vol': []
        }
        
        # Rolling forecast loop
        arima_fit = None
        garch_fit = None
        days_since_refit = 0
        
        for i, forecast_date in enumerate(forecast_dates):
            try:
                # Expanding window: train data + previous forecasts
                current_train_end_idx = returns_pct.index.get_loc(forecast_date) - 1
                if current_train_end_idx < 0:
                    continue
                    
                current_train_end = returns_pct.index[current_train_end_idx]
                expanded_train = returns_pct[
                    (returns_pct.index >= train_start) & 
                    (returns_pct.index <= current_train_end)
                ]
                
                if len(expanded_train) < 50:  # Need minimum data
                    continue
                
                # Refit models periodically
                if arima_fit is None or days_since_refit >= self.refit_frequency:
                    print(f"  🔄 Refitting models for {forecast_date.date()}...")
                    
                    arima_fit = fit_arima_auto(expanded_train)
                    garch_fit = fit_garch_adaptive(arima_fit.resid)  # Use improved GARCH
                    
                    days_since_refit = 0
                else:
                    days_since_refit += 1
                
                # Generate 1-day ahead forecasts
                if arima_fit is not None:
                    # Return forecast
                    return_forecast = arima_fit.forecast(steps=1).iloc[0]
                    
                    # Enhanced volatility forecast using ensemble method
                    recent_returns_for_vol = expanded_train[-20:]  # Last 20 days for vol estimation
                    vol_forecast = forecast_volatility_ensemble(
                        garch_fit, 
                        recent_returns_for_vol,
                        horizon=1
                    )
                    
                    # Price forecast
                    prev_price = prices.loc[current_train_end]
                    price_forecast = prev_price * np.exp(return_forecast / 100.0)
                    
                    # Actual values
                    actual_price = prices.loc[forecast_date]
                    actual_return = returns_pct.loc[forecast_date]
                    
                    # Enhanced realized volatility calculation
                    recent_actual_returns = returns_pct.loc[
                        (returns_pct.index <= forecast_date) & 
                        (returns_pct.index > forecast_date - timedelta(days=10))
                    ]
                    realized_vol = calculate_realized_volatility(recent_actual_returns, window=5)
                    
                    # Debug output for volatility (every 5 days)
                    if days_since_refit == 0 or i % 5 == 0:
                        regime, _ = detect_volatility_regime(recent_returns_for_vol)
                        print(f"    📊 Vol regime: {regime}, Forecast: {vol_forecast:.2f}%, Realized: {realized_vol:.2f}%")
                    
                    # Store results
                    results['dates'].append(forecast_date)
                    results['price_forecasts'].append(price_forecast)
                    results['vol_forecasts'].append(vol_forecast)
                    results['actual_prices'].append(actual_price)
                    results['actual_returns'].append(actual_return)
                    results['realized_vol'].append(realized_vol)
                
            except Exception as e:
                print(f"  ❌ Error forecasting {forecast_date.date()}: {e}")
                continue
        
        # Convert to DataFrame
        if len(results['dates']) == 0:
            return {'forecasts': pd.DataFrame(), 'summary': {}, 'metadata': {}}
        
        df = pd.DataFrame({
            'price_forecast': results['price_forecasts'],
            'vol_forecast': results['vol_forecasts'],
            'actual_price': results['actual_prices'],
            'actual_return': results['actual_returns'],
            'realized_vol': results['realized_vol']
        }, index=results['dates'])
        
        # Calculate errors
        df['price_error'] = df['actual_price'] - df['price_forecast']
        df['price_error_pct'] = (df['price_error'] / df['actual_price']) * 100
        df['vol_error'] = df['realized_vol'] - df['vol_forecast']
        
        # Summary statistics
        summary = self._calculate_summary(df)
        
        metadata = {
            'train_start': train_start,
            'train_end': train_end,
            'forecast_start': forecast_dates[0].strftime('%Y-%m-%d'),
            'forecast_end': forecast_dates[-1].strftime('%Y-%m-%d'),
            'n_forecasts': len(df),
            'train_days': len(train_returns)
        }
        
        return {'forecasts': df, 'summary': summary, 'metadata': metadata}
    
    def _calculate_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(df) == 0:
            return {}
        
        # Price metrics
        price_mae = np.mean(np.abs(df['price_error']))
        price_mse = np.mean(df['price_error'] ** 2)
        price_mape = np.mean(np.abs(df['price_error_pct']))
        
        # Volatility metrics
        vol_mae = np.mean(np.abs(df['vol_error']))
        vol_mse = np.mean(df['vol_error'] ** 2)
        
        # Directional accuracy - fix index alignment
        if len(df) > 1:
            # Convert to numpy arrays to avoid index issues
            actual_returns = df['actual_return'].values
            price_forecasts = df['price_forecast'].values
            
            # Calculate directions as numpy arrays
            actual_direction = np.sign(np.diff(actual_returns))
            forecast_direction = np.sign(np.diff(price_forecasts) / price_forecasts[:-1])
            
            # Calculate directional accuracy
            if len(actual_direction) > 0 and len(forecast_direction) > 0:
                min_len = min(len(actual_direction), len(forecast_direction))
                directional_accuracy = np.mean(
                    actual_direction[:min_len] == forecast_direction[:min_len]
                ) * 100
            else:
                directional_accuracy = 0
        else:
            directional_accuracy = 0
        
        return {
            'price_mae': price_mae,
            'price_mse': price_mse,
            'price_mape': price_mape,
            'vol_mae': vol_mae,
            'vol_mse': vol_mse,
            'directional_accuracy': directional_accuracy
        }

# ═══════════════════════════ VISUALIZATION ═══════════════════════════════

def plot_results(results: Dict[str, Any], ticker: str):
    """Plot forecast results."""
    df = results['forecasts']
    metadata = results['metadata']
    
    if df.empty:
        print("❌ No data to plot")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 1. Prices
    axes[0].plot(df.index, df['actual_price'], 'b-', label='Actual', linewidth=2)
    axes[0].plot(df.index, df['price_forecast'], 'r--', label='Forecast', linewidth=2, alpha=0.8)
    axes[0].set_title(f'{ticker} - Price Forecasts ({metadata["forecast_start"]} to {metadata["forecast_end"]})')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Volatility  
    axes[1].plot(df.index, df['realized_vol'], 'b-', label='Realized Vol', linewidth=2)
    axes[1].plot(df.index, df['vol_forecast'], 'r--', label='Forecast Vol', linewidth=2, alpha=0.8)
    axes[1].set_title(f'{ticker} - Volatility Forecasts')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Errors
    axes[2].plot(df.index, df['price_error_pct'], 'g-', alpha=0.7, label='Price Error (%)')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_title(f'{ticker} - Forecast Errors')
    axes[2].set_ylabel('Error (%)')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    summary = results['summary']
    print(f"\n📊 {ticker} Forecast Results")
    print("=" * 50)
    print(f"Training: {metadata['train_start']} to {metadata['train_end']} ({metadata['train_days']} days)")
    print(f"Forecasting: {metadata['forecast_start']} to {metadata['forecast_end']} ({metadata['n_forecasts']} days)")
    
    if summary:
        print(f"\nPerformance Metrics:")
        print(f"  Price MAE: ${summary['price_mae']:.2f}")
        print(f"  Price MAPE: {summary['price_mape']:.2f}%")
        print(f"  Price RMSE: ${np.sqrt(summary['price_mse']):.2f}")
        print(f"  Vol MAE: {summary['vol_mae']:.2f}%")
        print(f"  Vol MAPE: {summary['vol_mape']:.2f}%") 
        print(f"  Vol Hit Rate: {summary['vol_hit_rate']:.1f}% (within 20% tolerance)")
        print(f"  Vol RMSE: {np.sqrt(summary['vol_mse']):.2f}%")
        print(f"  Directional Accuracy: {summary['directional_accuracy']:.1f}%")
        print(f"  Forecast Sharpe: {summary['forecast_sharpe']:.3f}")
        print(f"  Model Changes: {summary['model_changes']}")

# ═══════════════════════════ MAIN EXECUTION ═══════════════════════════════

def main():
    """Main execution with simplified date logic."""
    
    # Parse arguments
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
        train_end = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    
    print(f"🚀 ARIMA-GARCH Forecast for {ticker}")
    print(f"📅 Training: {train_start} to {train_end}")
    print(f"🎯 Will forecast from {train_end} to today")
    
    try:
        # Download data (get extra history for training)
        data_start = (datetime.strptime(train_start, '%Y-%m-%d') - timedelta(days=250)).strftime('%Y-%m-%d')
        data_end = datetime.now().strftime('%Y-%m-%d')
        
        prices = download_prices_alpaca(ticker, data_start, data_end)
        
        print(f"📈 Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
        
        # Run enhanced forecast
        forecaster = EnhancedForecaster(
            refit_frequency=3,  # More frequent refitting
            use_enhanced_selection=True
        )
        results = forecaster.forecast(prices, train_start, train_end)
        
        if results['forecasts'].empty:
            print("❌ No forecasts generated. Check your dates.")
            print(f"   Try: python {sys.argv[0]} {ticker} 2023-01-01 2024-06-01")
            return
        
        # Plot and save results
        plot_results(results, ticker)
        
        # Save to CSV
        filename = f"{ticker}_forecast_{train_start}_{train_end}.csv"
        results['forecasts'].to_csv(filename)
        print(f"💾 Results saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()