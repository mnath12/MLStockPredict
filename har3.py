#!/usr/bin/env python
"""
Enhanced HAR-RV (Heterogeneous Autoregressive Realized Volatility) Forecasting System v2.0

Key Improvements:
1. True intraday realized volatility calculation
2. Jump-robust volatility estimators  
3. Enhanced HAR variants (HAR-RV-CJ, HAR-Q)
4. Regime-aware modeling
5. Volatility surface modeling
6. Better validation framework
7. Real-time trading signal generation
8. Risk management integration
"""

from __future__ import annotations
import os, sys, time, warnings
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════ ENHANCED REALIZED VOLATILITY CALCULATION ═══════════════════════════════

class VolatilityEstimator:
    """Enhanced volatility estimation with multiple robust estimators."""
    
    @staticmethod
    def realized_volatility_simple(returns: pd.Series, annualize: bool = True) -> float:
        """Simple realized volatility from returns."""
        rv = np.sum(returns**2)
        return np.sqrt(rv * 252) if annualize else np.sqrt(rv)
    
    @staticmethod
    def realized_volatility_5min(prices: pd.Series, freq: str = '5T') -> pd.Series:
        """Calculate RV from 5-minute intraday data (when available)."""
        # Resample to desired frequency
        price_freq = prices.resample(freq).last().dropna()
        
        # Calculate returns
        returns = np.log(price_freq / price_freq.shift(1)).dropna()
        
        # Daily RV (sum of squared intraday returns)
        daily_rv = returns.groupby(returns.index.date).apply(
            lambda x: np.sqrt(np.sum(x**2) * 252)  # Annualized
        )
        
        return daily_rv
    
    @staticmethod
    def bipower_variation(returns: pd.Series, annualize: bool = True) -> float:
        """Bipower variation - jump-robust volatility estimator."""
        abs_returns = np.abs(returns)
        n = len(abs_returns)
        
        if n < 2:
            return np.nan
            
        # Bipower variation
        bv = (np.pi/2) * np.sum(abs_returns[1:] * abs_returns[:-1])
        
        return np.sqrt(bv * 252) if annualize else np.sqrt(bv)
    
    @staticmethod
    def tripower_quarticity(returns: pd.Series) -> float:
        """Tripower quarticity for jump testing."""
        abs_returns = np.abs(returns)
        n = len(abs_returns)
        
        if n < 3:
            return np.nan
            
        # Tripower quarticity
        tq = n * (4/3) * np.sum(
            abs_returns[2:]**(4/3) * abs_returns[1:-1]**(4/3) * abs_returns[:-2]**(4/3)
        )
        
        return tq
    
    @staticmethod
    def jump_component(returns: pd.Series, significance_level: float = 0.05) -> Tuple[float, bool]:
        """Extract jump component using BNS test."""
        rv = VolatilityEstimator.realized_volatility_simple(returns, annualize=False)
        bv = VolatilityEstimator.bipower_variation(returns, annualize=False)
        tq = VolatilityEstimator.tripower_quarticity(returns)
        
        if np.isnan(bv) or np.isnan(tq) or tq <= 0:
            return 0.0, False
        
        # Jump test statistic (Barndorff-Nielsen & Shephard, 2006)
        n = len(returns)
        jump_stat = np.sqrt(n) * (rv - bv) / np.sqrt(tq)
        
        # Test for jumps
        critical_value = stats.norm.ppf(1 - significance_level/2)
        has_jumps = np.abs(jump_stat) > critical_value
        
        # Jump component (ensure non-negative)
        jump_component = max(0, rv - bv) if has_jumps else 0.0
        
        return jump_component, has_jumps

def calculate_enhanced_realized_volatility(prices: pd.Series, method: str = 'robust') -> pd.DataFrame:
    """
    Calculate enhanced realized volatility measures.
    
    Returns DataFrame with multiple volatility measures:
    - rv_simple: Standard realized volatility
    - rv_bv: Bipower variation (jump-robust)
    - rv_jump: Jump component
    - rv_continuous: Continuous component (BV)
    """
    
    # Calculate daily returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Group by date to handle potential intraday data
    daily_groups = log_returns.groupby(log_returns.index.date)
    
    results = []
    
    for date, day_returns in daily_groups:
        if len(day_returns) < 2:
            continue
            
        # Standard RV
        rv_simple = VolatilityEstimator.realized_volatility_simple(day_returns)
        
        # Jump-robust measures
        rv_bv = VolatilityEstimator.bipower_variation(day_returns)
        jump_comp, has_jumps = VolatilityEstimator.jump_component(day_returns)
        
        results.append({
            'date': pd.Timestamp(date),
            'rv_simple': rv_simple * 100,  # Convert to percentage
            'rv_bv': rv_bv * 100 if not np.isnan(rv_bv) else rv_simple * 100,
            'rv_jump': jump_comp * np.sqrt(252) * 100,  # Annualized jump component
            'rv_continuous': (rv_bv * 100) if not np.isnan(rv_bv) else rv_simple * 100,
            'has_jumps': has_jumps,
            'n_intraday': len(day_returns)
        })
    
    df = pd.DataFrame(results).set_index('date')
    
    # Ensure all measures are positive
    for col in ['rv_simple', 'rv_bv', 'rv_jump', 'rv_continuous']:
        df[col] = np.maximum(df[col], 0.1)  # Minimum 0.1% volatility
    
    return df

# ═══════════════════════════ ENHANCED HAR FEATURE ENGINEERING ═══════════════════════════════

class HARFeatureEngineer:
    """Advanced feature engineering for HAR models."""
    
    @staticmethod
    def calculate_har_cj_features(rv_df: pd.DataFrame, max_lag: int = 22) -> pd.DataFrame:
        """Calculate HAR-RV-CJ features (Continuous + Jump components)."""
        
        data = pd.DataFrame(index=rv_df.index)
        
        # Use continuous component as base RV
        rv_base = rv_df['rv_continuous']
        rv_jump = rv_df['rv_jump']
        
        # Standard HAR components for continuous part
        data['rv_daily'] = rv_base
        data['rv_weekly'] = rv_base.rolling(window=5, min_periods=1).mean()
        data['rv_monthly'] = rv_base.rolling(window=22, min_periods=1).mean()
        
        # Jump components  
        data['jump_daily'] = rv_jump
        data['jump_weekly'] = rv_jump.rolling(window=5, min_periods=1).mean()
        data['jump_monthly'] = rv_jump.rolling(window=22, min_periods=1).mean()
        
        # Lagged features
        for lag in [1, 2, 3]:
            data[f'rv_lag{lag}'] = rv_base.shift(lag)
            data[f'rv_weekly_lag{lag}'] = data['rv_weekly'].shift(lag)
            data[f'rv_monthly_lag{lag}'] = data['rv_monthly'].shift(lag)
            data[f'jump_lag{lag}'] = rv_jump.shift(lag)
        
        # Additional sophisticated features
        
        # 1. Volatility persistence
        data['rv_persistence'] = rv_base.rolling(window=10).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0, raw=True
        )
        
        # 2. Volatility momentum  
        data['rv_momentum'] = (rv_base - rv_base.rolling(window=5).mean()) / rv_base.rolling(window=5).std()
        
        # 3. Volatility mean reversion speed
        rv_mean = rv_base.rolling(window=60).mean()
        data['rv_mean_reversion'] = -(rv_base - rv_mean) / rv_mean
        
        # 4. Volatility regime indicators
        rv_quantiles = rv_base.rolling(window=252).quantile([0.25, 0.75])
        data['rv_regime_low'] = (rv_base < rv_base.rolling(window=252).quantile(0.25)).astype(int)
        data['rv_regime_high'] = (rv_base > rv_base.rolling(window=252).quantile(0.75)).astype(int)
        
        # 5. Jump frequency
        data['jump_frequency'] = rv_df['has_jumps'].rolling(window=22).sum()
        
        # 6. Volatility clustering (GARCH-like)
        data['rv_clustering'] = rv_base.rolling(window=5).std() / rv_base.rolling(window=22).std()
        
        # 7. Leverage effect proxy
        data['leverage_proxy'] = (rv_base / rv_base.shift(1) - 1).rolling(window=5).mean()
        
        # 8. Term structure of volatility
        data['vol_term_structure'] = data['rv_weekly'] / data['rv_monthly']
        
        # 9. Volatility skewness
        data['rv_skewness'] = rv_base.rolling(window=22).skew()
        
        # 10. Economic calendar effects (simplified)
        data['day_of_week'] = data.index.dayofweek
        data['monday'] = (data['day_of_week'] == 0).astype(int)
        data['friday'] = (data['day_of_week'] == 4).astype(int)
        data['month_end'] = (data.index.day >= 28).astype(int)
        
        return data
    
    @staticmethod
    def calculate_volatility_surface_features(rv_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features related to volatility surface dynamics."""
        
        data = pd.DataFrame(index=rv_df.index)
        rv = rv_df['rv_continuous']
        
        # Multi-horizon volatility features
        horizons = [1, 2, 5, 10, 22, 44, 66]  # Different forecast horizons
        
        for h in horizons:
            # Exponentially weighted moving average for different horizons
            alpha = 2.0 / (h + 1)
            data[f'ewma_{h}d'] = rv.ewm(alpha=alpha).mean()
            
            # Rolling quantiles for different horizons
            data[f'q25_{h}d'] = rv.rolling(window=h*5).quantile(0.25)
            data[f'q75_{h}d'] = rv.rolling(window=h*5).quantile(0.75)
            
            # Volatility slope
            if h > 1:
                data[f'vol_slope_{h}d'] = (data[f'ewma_{h}d'] - data[f'ewma_1d']) / (h - 1)
        
        # Volatility convexity
        data['vol_convexity'] = data['ewma_22d'] - 2*data['ewma_10d'] + data['ewma_1d']
        
        # Term structure slope
        data['term_slope'] = data['ewma_66d'] - data['ewma_5d']
        data['term_curvature'] = data['ewma_66d'] - 2*data['ewma_22d'] + data['ewma_5d']
        
        return data

# ═══════════════════════════ ENHANCED HAR MODEL VARIANTS ═══════════════════════════════

class HARModelAdvanced:
    """Advanced HAR model variants."""
    
    def __init__(self, model_type: str = 'har_cj'):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.fitted = False
        
    def _select_features(self, data: pd.DataFrame) -> List[str]:
        """Select features based on model type."""
        
        if self.model_type == 'har_basic':
            return ['rv_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1']
            
        elif self.model_type == 'har_cj':
            return [
                'rv_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1',
                'jump_lag1', 'jump_weekly', 'jump_monthly'
            ]
            
        elif self.model_type == 'har_q':
            # HAR with quarticity (volatility of volatility)
            return [
                'rv_lag1', 'rv_weekly_lag1', 'rv_monthly_lag1',
                'rv_persistence', 'rv_momentum', 'rv_clustering'
            ]
            
        elif self.model_type == 'har_full':
            # Full feature set
            base_features = [
                'rv_lag1', 'rv_lag2', 'rv_weekly_lag1', 'rv_monthly_lag1',
                'jump_lag1', 'jump_weekly', 'jump_monthly',
                'rv_persistence', 'rv_momentum', 'rv_mean_reversion',
                'rv_regime_low', 'rv_regime_high', 'jump_frequency',
                'rv_clustering', 'leverage_proxy', 'vol_term_structure',
                'monday', 'friday', 'month_end'
            ]
            return [f for f in base_features if f in data.columns]
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = 'ridge'):
        """Fit HAR model with enhanced features."""
        
        # Select features
        self.feature_names = self._select_features(X)
        X_selected = X[self.feature_names].dropna()
        y_aligned = y.reindex(X_selected.index).dropna()
        
        # Final alignment
        common_idx = X_selected.index.intersection(y_aligned.index)
        X_final = X_selected.loc[common_idx]
        y_final = y_aligned.loc[common_idx]
        
        if len(X_final) < 50:
            raise ValueError(f"Insufficient data for training: {len(X_final)} observations")
        
        # Choose estimation method
        if method == 'ols':
            X_final = sm.add_constant(X_final)
            self.model = sm.OLS(y_final, X_final).fit()
            
        elif method == 'ridge':
            self.model = Ridge(alpha=0.1, fit_intercept=True)
            self.model.fit(X_final, y_final)
            
        elif method == 'elastic_net':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
            self.model.fit(X_final, y_final)
            
        elif method == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            self.model.fit(X_final, y_final)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.estimation_method = method
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_pred = X[self.feature_names].dropna()
        
        if self.estimation_method == 'ols':
            X_pred = sm.add_constant(X_pred)
            return self.model.predict(X_pred).values
        else:
            return self.model.predict(X_pred)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance/coefficients."""
        if not self.fitted:
            return {}
        
        if self.estimation_method == 'ols':
            return dict(zip(self.model.params.index, self.model.params.values))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, self.model.coef_))
        elif hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            return {}

# ═══════════════════════════ REGIME-AWARE HAR MODELING ═══════════════════════════════

class RegimeAwareHAR:
    """HAR model with volatility regime detection."""
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.regime_model = None
        self.har_models = {}
        self.fitted = False
        
    def _detect_regimes(self, rv_series: pd.Series) -> pd.Series:
        """Simple regime detection using rolling quantiles."""
        
        # Use 60-day rolling window for regime classification
        rolling_window = 60
        
        regimes = pd.Series(index=rv_series.index, dtype=int)
        
        for i in range(len(rv_series)):
            if i < rolling_window:
                regimes.iloc[i] = 0  # Default to low regime
                continue
                
            # Calculate rolling statistics
            window_data = rv_series.iloc[max(0, i-rolling_window):i]
            
            # Simple regime classification based on percentiles
            current_vol = rv_series.iloc[i]
            p25 = window_data.quantile(0.33)
            p75 = window_data.quantile(0.67)
            
            if current_vol <= p25:
                regimes.iloc[i] = 0  # Low volatility regime
            elif current_vol >= p75:
                regimes.iloc[i] = 2 if self.n_regimes == 3 else 1  # High volatility regime
            else:
                regimes.iloc[i] = 1  # Medium volatility regime
        
        return regimes
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit regime-specific HAR models."""
        
        # Detect regimes
        regimes = self._detect_regimes(y)
        
        # Fit separate HAR model for each regime
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            
            if regime_mask.sum() < 30:  # Need minimum observations
                continue
                
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            # Fit HAR model for this regime
            har_model = HARModelAdvanced(model_type='har_cj')
            har_model.fit(X_regime, y_regime, method='ridge')
            
            self.har_models[regime] = har_model
        
        self.regimes = regimes
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, current_regime: Optional[int] = None) -> np.ndarray:
        """Make regime-aware predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # If no regime specified, use most recent
        if current_regime is None:
            current_regime = self.regimes.iloc[-1]
        
        # Use appropriate regime model
        if current_regime in self.har_models:
            return self.har_models[current_regime].predict(X)
        else:
            # Fallback to regime 0 or 1
            fallback_regime = 0 if 0 in self.har_models else list(self.har_models.keys())[0]
            return self.har_models[fallback_regime].predict(X)

# ═══════════════════════════ ENHANCED FORECASTING SYSTEM ═══════════════════════════════

@dataclass
class VolatilityForecast:
    """Container for volatility forecasts."""
    date: pd.Timestamp
    forecast: float
    confidence_interval: Tuple[float, float]
    regime: int
    model_confidence: float

class EnhancedHARForecaster:
    """Enhanced HAR forecasting system with multiple improvements."""
    
    def __init__(
        self,
        model_variants: List[str] = None,
        ensemble_method: str = 'weighted',
        confidence_level: float = 0.95,
        regime_aware: bool = True
    ):
        self.model_variants = model_variants or ['har_basic', 'har_cj', 'har_q', 'har_full']
        self.ensemble_method = ensemble_method
        self.confidence_level = confidence_level
        self.regime_aware = regime_aware
        
        self.models = {}
        self.ensemble_weights = {}
        self.fitted = False
    
    def _calculate_model_weights(self, validation_errors: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble weights based on validation performance."""
        
        if self.ensemble_method == 'equal':
            n_models = len(validation_errors)
            return {model: 1.0/n_models for model in validation_errors.keys()}
        
        elif self.ensemble_method == 'weighted':
            # Inverse error weighting
            inv_errors = {model: 1.0/max(error, 1e-6) for model, error in validation_errors.items()}
            total_inv_error = sum(inv_errors.values())
            return {model: inv_error/total_inv_error for model, inv_error in inv_errors.items()}
        
        elif self.ensemble_method == 'best_only':
            # Use only the best model
            best_model = min(validation_errors.items(), key=lambda x: x[1])[0]
            return {model: 1.0 if model == best_model else 0.0 for model in validation_errors.keys()}
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def fit(self, prices: pd.Series, train_start: str, train_end: str) -> Dict[str, Any]:
        """Fit enhanced HAR models with cross-validation."""
        
        print(f"🔧 Enhanced HAR Training: {train_start} to {train_end}")
        
        # Calculate enhanced realized volatility
        rv_df = calculate_enhanced_realized_volatility(prices)
        print(f"📊 Enhanced RV calculated: {len(rv_df)} observations")
        
        # Calculate HAR-CJ features
        feature_engineer = HARFeatureEngineer()
        har_features = feature_engineer.calculate_har_cj_features(rv_df)
        
        # Add volatility surface features
        surface_features = feature_engineer.calculate_volatility_surface_features(rv_df)
        har_features = har_features.join(surface_features, how='left')
        
        print(f"🔧 Enhanced features: {har_features.shape[1]} features")
        
        # Get training data
        train_mask = (har_features.index >= train_start) & (har_features.index <= train_end)
        train_data = har_features[train_mask].dropna()
        
        if len(train_data) < 100:
            raise ValueError(f"Insufficient training data: {len(train_data)}")
        
        # Target variable (next day's volatility)
        y = train_data['rv_daily'].shift(-1).dropna()
        X = train_data[:-1]  # Remove last row to align with y
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        validation_errors = {}
        
        print(f"🧪 Training {len(self.model_variants)} model variants with CV...")
        
        for model_name in self.model_variants:
            try:
                cv_errors = []
                
                # Cross-validation
                for train_idx, val_idx in tscv.split(X):
                    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train model
                    if self.regime_aware and model_name == 'har_full':
                        model = RegimeAwareHAR(n_regimes=2)
                    else:
                        model = HARModelAdvanced(model_type=model_name)
                    
                    model.fit(X_train_cv, y_train_cv)
                    
                    # Validate
                    y_pred_cv = model.predict(X_val_cv)
                    cv_error = mean_squared_error(y_val_cv, y_pred_cv)
                    cv_errors.append(cv_error)
                
                # Store average CV error
                avg_cv_error = np.mean(cv_errors)
                validation_errors[model_name] = avg_cv_error
                
                # Train final model on full training set
                if self.regime_aware and model_name == 'har_full':
                    final_model = RegimeAwareHAR(n_regimes=2)
                else:
                    final_model = HARModelAdvanced(model_type=model_name)
                
                final_model.fit(X, y)
                self.models[model_name] = final_model
                
                print(f"  ✅ {model_name}: CV-RMSE = {np.sqrt(avg_cv_error):.3f}")
                
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
                continue
        
        # Calculate ensemble weights
        if validation_errors:
            self.ensemble_weights = self._calculate_model_weights(validation_errors)
            print(f"🎯 Ensemble weights: {self.ensemble_weights}")
        
        self.fitted = True
        
        return {
            'validation_errors': validation_errors,
            'ensemble_weights': self.ensemble_weights,
            'n_features': har_features.shape[1],
            'training_samples': len(X)
        }
    
    def forecast_with_confidence(
        self, 
        prices: pd.Series, 
        forecast_dates: List[pd.Timestamp],
        train_start: str,
        train_end: str
    ) -> List[VolatilityForecast]:
        """Generate forecasts with confidence intervals."""
        
        if not self.fitted:
            self.fit(prices, train_start, train_end)
        
        # Prepare features for forecasting
        rv_df = calculate_enhanced_realized_volatility(prices)
        feature_engineer = HARFeatureEngineer()
        har_features = feature_engineer.calculate_har_cj_features(rv_df)
        surface_features = feature_engineer.calculate_volatility_surface_features(rv_df)
        har_features = har_features.join(surface_features, how='left')
        
        forecasts = []
        
        for forecast_date in forecast_dates:
            try:
                # Get features for this date
                if forecast_date not in har_features.index:
                    continue
                    
                X_forecast = har_features.loc[forecast_date:forecast_date]
                
                # Ensemble prediction
                ensemble_pred = 0.0
                ensemble_var = 0.0
                valid_predictions = []
                
                for model_name, model in self.models.items():
                    if model_name in self.ensemble_weights:
                        try:
                            pred = model.predict(X_forecast)[0]
                            weight = self.ensemble_weights[model_name]
                            
                            ensemble_pred += weight * pred
                            valid_predictions.append(pred)
                            
                        except Exception:
                            continue
                
                # Calculate confidence interval
                if len(valid_predictions) > 1:
                    pred_std = np.std(valid_predictions)
                    z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                    ci_lower = ensemble_pred - z_score * pred_std
                    ci_upper = ensemble_pred + z_score * pred_std
                    model_confidence = 1.0 / (1.0 + pred_std)  # Higher confidence for lower std
                else:
                    ci_lower = ensemble_pred * 0.8
                    ci_upper = ensemble_pred * 1.2
                    model_confidence = 0.5
                
                # Determine current regime (simplified)
                recent_vol = rv_df['rv_continuous'].iloc[-5:].mean()
                vol_quantile = rv_df['rv_continuous'].rolling(window=252).quantile(0.5).iloc[-1]
                current_regime = 1 if recent_vol > vol_quantile else 0
                
                forecast = VolatilityForecast(
                    date=forecast_date,
                    forecast=max(0.1, ensemble_pred),  # Ensure positive
                    confidence_interval=(max(0.1, ci_lower), ci_upper),
                    regime=current_regime,
                    model_confidence=model_confidence
                )
                
                forecasts.append(forecast)
                
            except Exception as e:
                print(f"  ❌ Error forecasting {forecast_date}: {e}")
                continue
        
        return forecasts

# ═══════════════════════════ VOLATILITY TRADING SIGNAL GENERATION ═══════════════════════════════

class VolatilityTradingSignals:
    """Generate trading signals based on volatility forecasts and implied volatility."""
    
    def __init__(self, signal_threshold: float = 2.0, confidence_threshold: float = 0.6):
        self.signal_threshold = signal_threshold  # Vol points threshold for signal
        self.confidence_threshold = confidence_threshold
        
    def calculate_volatility_risk_premium(
        self,
        forecast_vol: float,
        implied_vol: float,
        forecast_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate Volatility Risk Premium and generate trading signal.
        
        VRP = Forecast_RV - Implied_Vol
        """
        
        vrp = forecast_vol - implied_vol
        abs_vrp = abs(vrp)
        
        # Signal generation
        if abs_vrp >= self.signal_threshold and forecast_confidence >= self.confidence_threshold:
            if vrp > 0:
                signal = "LONG_GAMMA"  # Buy options
                signal_strength = min(abs_vrp / self.signal_threshold, 3.0)  # Cap at 3x
            else:
                signal = "SHORT_GAMMA"  # Sell options
                signal_strength = min(abs_vrp / self.signal_threshold, 3.0)
        else:
            signal = "NEUTRAL"
            signal_strength = 0.0
        
        return {
            'vrp': vrp,
            'signal': signal,
            'signal_strength': signal_strength,
            'forecast_vol': forecast_vol,
            'implied_vol': implied_vol,
            'confidence': forecast_confidence,
            'threshold_met': abs_vrp >= self.signal_threshold
        }
    
    def generate_position_sizing(
        self,
        signal_info: Dict[str, Any],
        max_position_size: float = 1.0,
        kelly_fraction: float = 0.25
    ) -> Dict[str, float]:
        """Generate position sizing based on signal strength and Kelly criterion."""
        
        if signal_info['signal'] == "NEUTRAL":
            return {'position_size': 0.0, 'kelly_size': 0.0, 'recommended_size': 0.0}
        
        # Base position size from signal strength
        base_size = signal_info['signal_strength'] / 3.0  # Normalize to [0,1]
        
        # Adjust for confidence
        confidence_adj = signal_info['confidence']
        
        # Kelly sizing (simplified)
        edge = abs(signal_info['vrp']) / signal_info['implied_vol']  # Relative edge
        kelly_size = kelly_fraction * edge * confidence_adj
        
        # Final position size
        recommended_size = min(
            max_position_size,
            base_size * confidence_adj,
            kelly_size
        )
        
        return {
            'position_size': base_size,
            'kelly_size': kelly_size,
            'recommended_size': recommended_size,
            'confidence_adjustment': confidence_adj
        }

# ═══════════════════════════ RISK MANAGEMENT ═══════════════════════════════

class VolatilityRiskManager:
    """Risk management for volatility trading strategies."""
    
    def __init__(
        self,
        max_gamma_exposure: float = 1000.0,  # Max gamma per $1M notional
        max_vega_exposure: float = 5000.0,   # Max vega per $1M notional
        stop_loss_pct: float = 0.15,         # 15% stop loss
        profit_target_pct: float = 0.30      # 30% profit target
    ):
        self.max_gamma_exposure = max_gamma_exposure
        self.max_vega_exposure = max_vega_exposure
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        
    def check_risk_limits(
        self,
        current_gamma: float,
        current_vega: float,
        portfolio_value: float,
        new_position_gamma: float,
        new_position_vega: float
    ) -> Dict[str, Any]:
        """Check if new position violates risk limits."""
        
        # Normalize by portfolio value (per $1M)
        scale = portfolio_value / 1_000_000
        
        total_gamma = (current_gamma + new_position_gamma) / scale
        total_vega = (current_vega + new_position_vega) / scale
        
        gamma_ok = abs(total_gamma) <= self.max_gamma_exposure
        vega_ok = abs(total_vega) <= self.max_vega_exposure
        
        return {
            'gamma_ok': gamma_ok,
            'vega_ok': vega_ok,
            'can_trade': gamma_ok and vega_ok,
            'total_gamma': total_gamma,
            'total_vega': total_vega,
            'gamma_utilization': abs(total_gamma) / self.max_gamma_exposure,
            'vega_utilization': abs(total_vega) / self.max_vega_exposure
        }
    
    def calculate_hedging_frequency(
        self,
        current_vol: float,
        forecast_vol: float,
        gamma_exposure: float
    ) -> str:
        """Determine optimal hedging frequency based on conditions."""
        
        vol_ratio = forecast_vol / current_vol if current_vol > 0 else 1.0
        gamma_size = abs(gamma_exposure)
        
        if vol_ratio > 1.5 or gamma_size > 500:
            return "HIGH_FREQ"  # Hedge multiple times per day
        elif vol_ratio > 1.2 or gamma_size > 200:
            return "DAILY"      # Hedge once per day
        else:
            return "LOW_FREQ"   # Hedge every few days

# ═══════════════════════════ ENHANCED BACKTESTING FRAMEWORK ═══════════════════════════════

class VolatilityStrategyBacktester:
    """Enhanced backtesting for volatility strategies."""
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_costs: float = 0.01,  # 1 cent per contract
        slippage: float = 0.005          # 0.5% slippage
    ):
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.slippage = slippage
        
    def backtest_volatility_strategy(
        self,
        forecasts: List[VolatilityForecast],
        implied_vols: pd.Series,
        actual_vols: pd.Series,
        signal_generator: VolatilityTradingSignals
    ) -> Dict[str, Any]:
        """
        Backtest volatility trading strategy.
        
        Simplified P&L calculation focusing on volatility forecasting accuracy.
        """
        
        results = []
        portfolio_value = self.initial_capital
        total_pnl = 0.0
        
        for forecast in forecasts:
            if forecast.date not in implied_vols.index or forecast.date not in actual_vols.index:
                continue
                
            implied_vol = implied_vols[forecast.date]
            actual_vol = actual_vols[forecast.date]
            
            # Generate trading signal
            signal_info = signal_generator.calculate_volatility_risk_premium(
                forecast.forecast,
                implied_vol,
                forecast.model_confidence
            )
            
            # Calculate position sizing
            position_info = signal_generator.generate_position_sizing(signal_info)
            
            # Simplified P&L calculation
            # Assumes we can perfectly delta-hedge and capture vol difference
            vol_difference = actual_vol - implied_vol
            
            if signal_info['signal'] == "LONG_GAMMA" and vol_difference > 0:
                # Profitable long gamma trade
                trade_pnl = vol_difference * position_info['recommended_size'] * 1000
            elif signal_info['signal'] == "SHORT_GAMMA" and vol_difference < 0:
                # Profitable short gamma trade  
                trade_pnl = abs(vol_difference) * position_info['recommended_size'] * 1000
            elif signal_info['signal'] == "NEUTRAL":
                trade_pnl = 0.0
            else:
                # Losing trade
                trade_pnl = -abs(vol_difference) * position_info['recommended_size'] * 1000
            
            # Apply transaction costs
            if signal_info['signal'] != "NEUTRAL":
                trade_pnl -= self.transaction_costs * position_info['recommended_size'] * 100
            
            total_pnl += trade_pnl
            portfolio_value += trade_pnl
            
            results.append({
                'date': forecast.date,
                'forecast_vol': forecast.forecast,
                'implied_vol': implied_vol,
                'actual_vol': actual_vol,
                'signal': signal_info['signal'],
                'signal_strength': signal_info['signal_strength'],
                'position_size': position_info['recommended_size'],
                'trade_pnl': trade_pnl,
                'total_pnl': total_pnl,
                'portfolio_value': portfolio_value,
                'vol_error': abs(forecast.forecast - actual_vol),
                'vrp': signal_info['vrp'],
                'vrp_realized': actual_vol - implied_vol
            })
        
        if not results:
            return {'results': pd.DataFrame(), 'summary': {}}
        
        results_df = pd.DataFrame(results).set_index('date')
        
        # Calculate performance metrics
        summary = self._calculate_backtest_summary(results_df, self.initial_capital)
        
        return {'results': results_df, 'summary': summary}
    
    def _calculate_backtest_summary(self, results_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """Calculate backtest performance summary."""
        
        if results_df.empty:
            return {}
        
        total_return = (results_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital
        
        # Daily returns
        daily_returns = results_df['portfolio_value'].pct_change().dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        rolling_max = results_df['portfolio_value'].expanding().max()
        drawdown = (results_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = results_df[results_df['trade_pnl'] > 0]
        losing_trades = results_df[results_df['trade_pnl'] < 0]
        
        win_rate = len(winning_trades) / len(results_df[results_df['trade_pnl'] != 0]) if len(results_df[results_df['trade_pnl'] != 0]) > 0 else 0
        
        avg_win = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss < 0 else np.inf
        
        # Signal accuracy
        correct_signals = 0
        total_signals = 0
        
        for _, row in results_df.iterrows():
            if row['signal'] != 'NEUTRAL':
                total_signals += 1
                vrp_forecast = row['vrp']
                vrp_realized = row['vrp_realized']
                
                # Check if signal direction was correct
                if (vrp_forecast > 0 and vrp_realized > 0) or (vrp_forecast < 0 and vrp_realized < 0):
                    correct_signals += 1
        
        signal_accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(results_df)) if len(results_df) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(results_df[results_df['trade_pnl'] != 0]),
            'signal_accuracy': signal_accuracy,
            'forecast_mae': results_df['vol_error'].mean(),
            'final_portfolio_value': results_df['portfolio_value'].iloc[-1]
        }

# ═══════════════════════════ ENHANCED VISUALIZATION ═══════════════════════════

def plot_enhanced_results(
    forecaster_results: Dict[str, Any],
    backtest_results: Dict[str, Any],
    ticker: str
):
    """Enhanced plotting with multiple visualizations."""
    
    forecast_df = forecaster_results.get('forecasts', pd.DataFrame())
    backtest_df = backtest_results.get('results', pd.DataFrame())
    
    if forecast_df.empty and backtest_df.empty:
        print("❌ No data to plot")
        return
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Volatility forecasts vs actual (top left)
    ax1 = plt.subplot(3, 3, 1)
    if not backtest_df.empty:
        ax1.plot(backtest_df.index, backtest_df['actual_vol'], 'b-', label='Actual Vol', linewidth=2)
        ax1.plot(backtest_df.index, backtest_df['forecast_vol'], 'r--', label='Forecast Vol', linewidth=2)
        ax1.plot(backtest_df.index, backtest_df['implied_vol'], 'g:', label='Implied Vol', linewidth=2)
        ax1.set_title(f'{ticker} - Volatility Comparison')
        ax1.set_ylabel('Volatility (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. VRP and trading signals (top middle)
    ax2 = plt.subplot(3, 3, 2)
    if not backtest_df.empty:
        ax2.plot(backtest_df.index, backtest_df['vrp'], 'purple', label='VRP Forecast', linewidth=2)
        ax2.plot(backtest_df.index, backtest_df['vrp_realized'], 'orange', label='VRP Realized', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.3, label='Signal Threshold')
        ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.3)
        ax2.set_title('Volatility Risk Premium')
        ax2.set_ylabel('VRP (vol points)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Portfolio performance (top right)
    ax3 = plt.subplot(3, 3, 3)
    if not backtest_df.empty:
        portfolio_returns = (backtest_df['portfolio_value'] / backtest_df['portfolio_value'].iloc[0] - 1) * 100
        ax3.plot(backtest_df.index, portfolio_returns, 'darkgreen', linewidth=2)
        ax3.set_title('Portfolio Performance')
        ax3.set_ylabel('Returns (%)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Signal distribution (middle left)
    ax4 = plt.subplot(3, 3, 4)
    if not backtest_df.empty:
        signal_counts = backtest_df['signal'].value_counts()
        colors = {'LONG_GAMMA': 'green', 'SHORT_GAMMA': 'red', 'NEUTRAL': 'gray'}
        ax4.bar(signal_counts.index, signal_counts.values, 
                color=[colors.get(x, 'blue') for x in signal_counts.index])
        ax4.set_title('Trading Signal Distribution')
        ax4.set_ylabel('Count')
    
    # 5. Forecast errors (middle middle)
    ax5 = plt.subplot(3, 3, 5)
    if not backtest_df.empty:
        ax5.hist(backtest_df['vol_error'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(x=backtest_df['vol_error'].mean(), color='red', linestyle='--', 
                   label=f'Mean Error: {backtest_df["vol_error"].mean():.2f}')
        ax5.set_title('Forecast Error Distribution')
        ax5.set_xlabel('Absolute Error (vol points)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
    
    # 6. Rolling Sharpe ratio (middle right)
    ax6 = plt.subplot(3, 3, 6)
    if not backtest_df.empty and len(backtest_df) > 30:
        daily_rets = backtest_df['portfolio_value'].pct_change().dropna()
        rolling_sharpe = daily_rets.rolling(window=30).mean() / daily_rets.rolling(window=30).std() * np.sqrt(252)
        ax6.plot(rolling_sharpe.index, rolling_sharpe, 'darkblue', linewidth=2)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.set_title('Rolling 30-Day Sharpe Ratio')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.grid(True, alpha=0.3)
    
    # 7. Model confidence over time (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    if 'model_confidence' in forecast_df.columns:
        ax7.plot(forecast_df.index, forecast_df['model_confidence'], 'orange', linewidth=2)
        ax7.set_title('Model Confidence Over Time')
        ax7.set_ylabel('Confidence Score')
        ax7.set_ylim(0, 1)
        ax7.grid(True, alpha=0.3)
    
    # 8. Feature importance (bottom middle)
    ax8 = plt.subplot(3, 3, 8)
    if 'ensemble_weights' in forecaster_results:
        weights = forecaster_results['ensemble_weights']
        if weights:
            models = list(weights.keys())
            model_weights = list(weights.values())
            ax8.bar(models, model_weights, color='lightcoral')
            ax8.set_title('Model Ensemble Weights')
            ax8.set_ylabel('Weight')
            ax8.tick_params(axis='x', rotation=45)
    
    # 9. P&L attribution (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    if not backtest_df.empty:
        cumulative_pnl = backtest_df['trade_pnl'].cumsum()
        ax9.plot(backtest_df.index, cumulative_pnl, 'darkred', linewidth=2)
        ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax9.set_title('Cumulative P&L')
        ax9.set_ylabel('P&L ($)')
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print_enhanced_summary(forecaster_results, backtest_results, ticker)

def print_enhanced_summary(
    forecaster_results: Dict[str, Any],
    backtest_results: Dict[str, Any], 
    ticker: str
):
    """Print comprehensive results summary."""
    
    backtest_summary = backtest_results.get('summary', {})
    
    print(f"\n{'='*80}")
    print(f"🎯 ENHANCED HAR-RV STRATEGY RESULTS: {ticker}")
    print(f"{'='*80}")
    
    # Forecasting performance
    if 'validation_errors' in forecaster_results:
        print(f"\n📊 FORECASTING PERFORMANCE:")
        print(f"{'─'*50}")
        
        for model, error in forecaster_results['validation_errors'].items():
            weight = forecaster_results.get('ensemble_weights', {}).get(model, 0)
            print(f"  {model:<15}: RMSE={np.sqrt(error):.3f}, Weight={weight:.3f}")
    
    # Trading performance  
    if backtest_summary:
        print(f"\n💰 TRADING PERFORMANCE:")
        print(f"{'─'*50}")
        print(f"  Total Return:        {backtest_summary['total_return']:.2%}")
        print(f"  Annualized Return:   {backtest_summary['annualized_return']:.2%}")
        print(f"  Volatility:          {backtest_summary['volatility']:.2%}")
        print(f"  Sharpe Ratio:        {backtest_summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {backtest_summary['max_drawdown']:.2%}")
        print(f"  Win Rate:            {backtest_summary['win_rate']:.1%}")
        print(f"  Profit Factor:       {backtest_summary['profit_factor']:.2f}")
        print(f"  Signal Accuracy:     {backtest_summary['signal_accuracy']:.1%}")
        print(f"  Total Trades:        {backtest_summary['total_trades']}")
        print(f"  Final Portfolio:     ${backtest_summary['final_portfolio_value']:,.0f}")
        
        # Performance assessment
        if backtest_summary['sharpe_ratio'] > 1.5:
            print(f"  📈 Excellent strategy performance!")
        elif backtest_summary['sharpe_ratio'] > 1.0:
            print(f"  📊 Good strategy performance")
        elif backtest_summary['sharpe_ratio'] > 0.5:
            print(f"  ⚠️  Moderate strategy performance")
        else:
            print(f"  ❌ Poor strategy performance")

# ═══════════════════════════ MAIN ENHANCED EXECUTION ═══════════════════════════════

def main_enhanced():
    """Enhanced main execution with full system integration."""
    
    # Parse arguments (same as before)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    train_start = sys.argv[2] if len(sys.argv) > 2 else (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    train_end = sys.argv[3] if len(sys.argv) > 3 else (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"🚀 Enhanced HAR-RV Volatility Strategy for {ticker}")
    print(f"📅 Training: {train_start} to {train_end}")
    
    try:
        # Download data (reuse existing function)
        data_start = (datetime.strptime(train_start, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        data_end = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📡 Downloading price data...")
        # You can reuse your existing download function here
        # prices = download_prices_alpaca(ticker, data_start, data_end)
        
        # For demonstration, let's create a simple download function
        prices = yf.download(ticker, start=data_start, end=data_end, auto_adjust=True)['Close']
        
        # Initialize enhanced forecaster
        print(f"\n🔬 Initializing Enhanced HAR Forecaster...")
        forecaster = EnhancedHARForecaster(
            model_variants=['har_basic', 'har_cj', 'har_q', 'har_full'],
            ensemble_method='weighted',
            regime_aware=True
        )
        
        # Fit models
        fit_results = forecaster.fit(prices, train_start, train_end)
        
        # Generate forecasts for out-of-sample period
        forecast_start = pd.Timestamp(train_end) + pd.Timedelta(days=1)
        forecast_end = pd.Timestamp(data_end)
        forecast_dates = pd.date_range(forecast_start, forecast_end, freq='D')
        forecast_dates = [d for d in forecast_dates if d in prices.index]
        
        print(f"\n🔮 Generating forecasts for {len(forecast_dates)} days...")
        volatility_forecasts = forecaster.forecast_with_confidence(
            prices, forecast_dates, train_start, train_end
        )
        
        if not volatility_forecasts:
            print("❌ No forecasts generated")
            return
        
        # For backtesting, we need implied volatility data
        # In a real implementation, you would fetch this from options data
        # For demo, we'll simulate implied vol as a noisy version of realized vol
        print(f"\n📊 Simulating implied volatility data...")
        
        rv_df = calculate_enhanced_realized_volatility(prices)
        implied_vols = rv_df['rv_continuous'] * (1 + np.random.normal(0, 0.1, len(rv_df)))  # Add noise
        implied_vols = implied_vols.reindex([f.date for f in volatility_forecasts]).dropna()
        
        actual_vols = rv_df['rv_continuous'].reindex([f.date for f in volatility_forecasts]).dropna()
        
        # Initialize trading components
        signal_generator = VolatilityTradingSignals(signal_threshold=2.0, confidence_threshold=0.6)
        backtester = VolatilityStrategyBacktester(initial_capital=1_000_000)
        
        # Run backtest
        print(f"\n📈 Running enhanced backtest...")
        backtest_results = backtester.backtest_volatility_strategy(
            volatility_forecasts, implied_vols, actual_vols, signal_generator
        )
        
        # Combine forecaster results for plotting
        forecast_df = pd.DataFrame([
            {
                'date': f.date,
                'forecast': f.forecast,
                'confidence_lower': f.confidence_interval[0],
                'confidence_upper': f.confidence_interval[1],
                'model_confidence': f.model_confidence,
                'regime': f.regime
            }
            for f in volatility_forecasts
        ]).set_index('date')
        
        forecaster_results = {
            'forecasts': forecast_df,
            'validation_errors': fit_results['validation_errors'],
            'ensemble_weights': fit_results['ensemble_weights']
        }
        
        # Plot results
        print(f"\n📊 Generating enhanced visualizations...")
        plot_enhanced_results(forecaster_results, backtest_results, ticker)
        
        # Save results
        if not backtest_results['results'].empty:
            filename = f"{ticker}_enhanced_har_strategy_{train_start}_{train_end}.csv"
            backtest_results['results'].to_csv(filename)
            print(f"💾 Results saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_enhanced()