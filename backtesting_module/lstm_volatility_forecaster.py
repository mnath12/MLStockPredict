#!/usr/bin/env python3
"""
LSTM Volatility Forecaster for Backtesting System

This module provides a specialized interface for the LSTM volatility model
trained on Google Colab. It handles loading the pre-trained model and scaler,
and provides real-time volatility forecasts during backtesting.

Key Features:
- Loads pre-trained LSTM model (.h5) and scaler (.pkl) from Colab
- Processes hourly realized volatility data in the correct format
- Provides volatility forecasts aligned with backtesting timestamps
- Handles model input preparation and scaling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json

class LSTMVolatilityForecaster:
    """
    Specialized volatility forecaster for the LSTM model trained on Colab.
    
    This class handles:
    1. Loading the pre-trained LSTM model and scaler from Colab
    2. Processing hourly realized volatility data
    3. Making predictions with proper input formatting
    4. Aligning forecasts with backtesting timestamps
    """
    
    def __init__(self, model_path: str, scaler_path: str, memory_window: int = 60):
        """
        Initialize the LSTM volatility forecaster.
        
        Args:
            model_path: Path to the saved LSTM model (.h5 file)
            scaler_path: Path to the saved scaler (.pkl file)
            memory_window: Lookback window for LSTM (default: 60 hours)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.memory_window = memory_window
        self.model = None
        self.scaler = None
        self.is_model_loaded = False
        self.forecast_history = []
        self.actual_history = []
        
        # Load model and scaler
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the pre-trained LSTM model and scaler from Colab files.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Load TensorFlow/Keras model
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
                
            try:
                import tensorflow as tf
                from tensorflow.keras.layers import InputLayer
                
                # Create custom InputLayer class to handle batch_shape compatibility
                class CompatibleInputLayer(InputLayer):
                    def __init__(self, **kwargs):
                        # Convert batch_shape to input_shape for compatibility
                        if 'batch_shape' in kwargs:
                            batch_shape = kwargs.pop('batch_shape')
                            if batch_shape[0] is None:  # Remove batch dimension
                                kwargs['input_shape'] = batch_shape[1:]
                        super().__init__(**kwargs)
                
                # Define custom objects for model loading
                custom_objects = {
                    'InputLayer': CompatibleInputLayer,
                }
                
                # Create a dummy DTypePolicy class for compatibility
                class DummyDTypePolicy:
                    def __init__(self, **kwargs):
                        # Just ignore all arguments for compatibility
                        self.name = kwargs.get('name', 'dtype_policy')
                        self.variable_dtype = kwargs.get('variable_dtype', 'float32')
                        self.compute_dtype = kwargs.get('compute_dtype', 'float32')
                    
                    def __getattr__(self, name):
                        # Return None for any missing attributes
                        return None
                
                custom_objects['DTypePolicy'] = DummyDTypePolicy
                
                # Try loading with custom objects
                try:
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    print(f"✅ LSTM model loaded from {self.model_path}")
                except Exception as load_error:
                    print(f"⚠️  Model loading failed: {load_error}")
                    print("   This might be due to Keras version differences")
                    print("   The model was likely saved with a different Keras version")
                    return False
                    
            except ImportError:
                print("❌ TensorFlow not available for loading LSTM model")
                return False
            except Exception as e:
                print(f"❌ Error loading LSTM model: {e}")
                return False
            
            # Load scaler
            if not os.path.exists(self.scaler_path):
                print(f"❌ Scaler file not found: {self.scaler_path}")
                return False
                
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded from {self.scaler_path}")
            except Exception as e:
                print(f"❌ Error loading scaler: {e}")
                return False
            
            self.is_model_loaded = True
            print("✅ LSTM volatility forecaster initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing LSTM forecaster: {e}")
            self.is_model_loaded = False
            return False
    
    def detect_time_interval(self, bars_df: pd.DataFrame) -> str:
        """
        Detect the time interval of the data based on the index frequency.
        
        Args:
            bars_df: DataFrame with DatetimeIndex
            
        Returns:
            str: Detected interval ('minute', 'hour', 'day', etc.)
        """
        if len(bars_df) < 2:
            return 'unknown'
        
        # Get time differences between consecutive timestamps
        time_diffs = bars_df.index.to_series().diff().dropna()
        
        # Get the most common time difference
        most_common_diff = time_diffs.mode().iloc[0]
        
        # Convert to minutes for easier comparison
        diff_minutes = most_common_diff.total_seconds() / 60
        
        if diff_minutes <= 1.5:  # Allow some tolerance
            return 'minute'
        elif diff_minutes <= 90:  # 1-90 minutes
            return f'{int(diff_minutes)}minute'
        elif diff_minutes <= 1440:  # 1-24 hours
            return 'hour'
        elif diff_minutes <= 10080:  # 1-7 days
            return 'day'
        else:
            return 'unknown'
    
    def prepare_hourly_realized_volatility(self, bars_df: pd.DataFrame, symbol: str = None) -> pd.Series:
        """
        Compute hourly realized volatility from price data, matching the Colab training format.
        Automatically detects the data frequency and processes accordingly.
        
        Args:
            bars_df: DataFrame with OHLCV data
            symbol: Symbol to extract (if MultiIndex)
            
        Returns:
            pd.Series: Hourly realized volatility
        """
        # Detect the time interval of the input data
        interval = self.detect_time_interval(bars_df)
        print(f"🔍 Detected data interval: {interval}")
        
        # Extract price series
        if isinstance(bars_df.index, pd.MultiIndex):
            if symbol is None:
                raise ValueError("Symbol must be provided for MultiIndex data")
            px = (bars_df.xs(symbol, level="symbol")['close']
                             .tz_convert("America/New_York")
                             .sort_index())
        else:
            px = bars_df['close'].tz_convert("America/New_York").sort_index()

        # Compute returns
        ret = px.pct_change().dropna()

        # Square returns and bin into right-closed hourly buckets
        sq = ret.pow(2)
        
        # Handle different time intervals
        if interval == 'minute':
            # For minute data, group by hour
            hourly_sum = sq.groupby(pd.Grouper(freq="h", label="right", closed="right")).sum()
        elif interval.startswith('minute'):
            # For multi-minute data (e.g., 5minute), group by hour
            hourly_sum = sq.groupby(pd.Grouper(freq="h", label="right", closed="right")).sum()
        elif interval == 'hour':
            # For hourly data, use as-is
            hourly_sum = sq
        elif interval == 'day':
            # For daily data, we can't compute hourly RV, so use daily RV
            print("⚠️  Daily data detected - using daily realized volatility instead of hourly")
            daily_sum = sq.groupby(pd.Grouper(freq="D", label="right", closed="right")).sum()
            rv_daily = np.sqrt(daily_sum).rename("rv_daily")
            if rv_daily.index.tz is None:
                rv_daily.index = rv_daily.index.tz_localize("America/New_York")
            return rv_daily
        else:
            # Default to hourly grouping
            hourly_sum = sq.groupby(pd.Grouper(freq="h", label="right", closed="right")).sum()

        # Take sqrt to get realized vol
        rv_hourly = np.sqrt(hourly_sum).rename("rv_hourly")

        # Ensure the index is tz-aware local hours
        if rv_hourly.index.tz is None:
            rv_hourly.index = rv_hourly.index.tz_localize("America/New_York")

        return rv_hourly
    
    def prepare_lstm_input(self, rv_hourly: pd.Series, target_timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """
        Prepare input data for LSTM model prediction.
        
        Args:
            rv_hourly: Hourly realized volatility series
            target_timestamp: Timestamp for which to make prediction
            
        Returns:
            np.ndarray: Prepared input data for LSTM (shape: [1, memory_window, 1])
        """
        try:
            # Get data up to (but not including) the target timestamp
            historical_data = rv_hourly.loc[rv_hourly.index < target_timestamp]
            
            if len(historical_data) < self.memory_window:
                print(f"⚠️  Insufficient data for LSTM: Need {self.memory_window} hours, have {len(historical_data)} hours")
                print(f"   Data range: {historical_data.index.min()} to {historical_data.index.max()}")
                print(f"   Using fallback volatility estimation")
                return None
            
            # Take the last memory_window hours
            recent_data = historical_data.tail(self.memory_window)
            
            # Convert to numpy array and reshape for scaler
            data_vals = recent_data.values.reshape(-1, 1)
            
            # Scale the data using the pre-trained scaler
            scaled_data = self.scaler.transform(data_vals)
            
            # Reshape for LSTM input: [samples, time_steps, features]
            lstm_input = scaled_data.reshape(1, self.memory_window, 1)
            
            return lstm_input
            
        except Exception as e:
            print(f"Error preparing LSTM input: {e}")
            return None
    
    def forecast_volatility(self, bars_df: pd.DataFrame, forecast_timestamp: pd.Timestamp, symbol: str = None) -> float:
        """
        Generate volatility forecast using the LSTM model.
        
        Args:
            bars_df: Historical price data
            forecast_timestamp: Timestamp for which to forecast volatility
            symbol: Symbol to extract (if MultiIndex)
            
        Returns:
            float: Forecasted volatility (annualized)
        """
        try:
            if not self.is_model_loaded:
                print("❌ Model not loaded. Cannot make predictions.")
                return self._fallback_forecast(bars_df, symbol)
            
            # Validate input data
            if bars_df is None or bars_df.empty:
                print("❌ No input data provided")
                return self._fallback_forecast(bars_df, symbol)
            
            # Compute hourly realized volatility
            try:
                rv_hourly = self.prepare_hourly_realized_volatility(bars_df, symbol)
            except Exception as e:
                print(f"❌ Error computing hourly realized volatility: {e}")
                return self._fallback_forecast(bars_df, symbol)
            
            if rv_hourly.empty:
                print("❌ No hourly volatility data available")
                return self._fallback_forecast(bars_df, symbol)
            
            # Prepare LSTM input
            try:
                lstm_input = self.prepare_lstm_input(rv_hourly, forecast_timestamp)
            except Exception as e:
                print(f"❌ Error preparing LSTM input: {e}")
                return self._fallback_forecast(bars_df, symbol)
            
            if lstm_input is None:
                print("❌ Could not prepare LSTM input")
                return self._fallback_forecast(bars_df, symbol)
            
            # Make prediction
            try:
                prediction_scaled = self.model.predict(lstm_input, verbose=0)
            except Exception as e:
                print(f"❌ Error making LSTM prediction: {e}")
                return self._fallback_forecast(bars_df, symbol)
            
            # Inverse transform to get actual volatility
            try:
                prediction_actual = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
                forecast_vol = float(prediction_actual[0, 0])
            except Exception as e:
                print(f"❌ Error inverse transforming prediction: {e}")
                return self._fallback_forecast(bars_df, symbol)
            
            # Validate forecast
            if pd.isna(forecast_vol) or forecast_vol < 0:
                print(f"❌ Invalid forecast value: {forecast_vol}")
                return self._fallback_forecast(bars_df, symbol)
            
            # Store forecast for tracking
            self.forecast_history.append({
                'timestamp': forecast_timestamp,
                'forecast': forecast_vol,
                'method': 'lstm'
            })
            
            return forecast_vol
            
        except Exception as e:
            print(f"❌ Unexpected error in LSTM volatility forecasting: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_forecast(bars_df, symbol)
    
    def _fallback_forecast(self, bars_df: pd.DataFrame, symbol: str = None) -> float:
        """
        Fallback volatility forecasting when LSTM model fails.
        
        Args:
            bars_df: Historical price data
            symbol: Symbol to extract (if MultiIndex)
            
        Returns:
            float: Fallback volatility forecast
        """
        try:
            # Extract price series
            if isinstance(bars_df.index, pd.MultiIndex):
                if symbol is None:
                    return 0.2  # Default volatility
                px = bars_df.xs(symbol, level="symbol")['close']
            else:
                px = bars_df['close']
            
            if len(px) < 2:
                return 0.2  # Default volatility
            
            # Calculate simple volatility estimate
            returns = np.log(px / px.shift(1)).dropna()
            vol = returns.rolling(21).std().iloc[-1] * np.sqrt(252)
            
            return float(vol) if not pd.isna(vol) else 0.2
            
        except Exception as e:
            print(f"Error in fallback forecast: {e}")
            return 0.2  # Default volatility
    
    def update_actual_volatility(self, timestamp: pd.Timestamp, actual_vol: float):
        """
        Update actual volatility for model performance tracking.
        
        Args:
            timestamp: Timestamp of the volatility observation
            actual_vol: Actual realized volatility
        """
        self.actual_history.append({
            'timestamp': timestamp,
            'actual': actual_vol
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the LSTM volatility forecasts.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.forecast_history) == 0 or len(self.actual_history) == 0:
            return {}
        
        # Align forecasts and actuals
        forecast_df = pd.DataFrame(self.forecast_history).set_index('timestamp')
        actual_df = pd.DataFrame(self.actual_history).set_index('timestamp')
        
        # Merge on timestamp
        combined = forecast_df.join(actual_df, how='inner')
        
        if len(combined) == 0:
            return {}
        
        # Calculate metrics
        errors = combined['actual'] - combined['forecast']
        
        metrics = {
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mape': float(np.mean(np.abs(errors / combined['actual'])) * 100),
            'correlation': float(combined['forecast'].corr(combined['actual'])),
            'n_forecasts': len(combined)
        }
        
        return metrics
    
    def get_latest_forecast(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent volatility forecast.
        
        Returns:
            Dictionary with latest forecast info or None
        """
        if len(self.forecast_history) > 0:
            return self.forecast_history[-1]
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_loaded': self.is_model_loaded,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'memory_window': self.memory_window,
            'n_forecasts': len(self.forecast_history)
        }
        
        if self.is_model_loaded and self.model is not None:
            try:
                info['model_summary'] = str(self.model.summary())
            except:
                info['model_summary'] = "Model summary not available"
        
        return info


def create_lstm_volatility_forecaster(
    model_path: str = "volatility_models/volatility_lstm_model.h5",
    scaler_path: str = "volatility_models/volatility_scaler.pkl",
    memory_window: int = 60
) -> LSTMVolatilityForecaster:
    """
    Create an LSTM volatility forecaster with the specified model files.
    
    Args:
        model_path: Path to the LSTM model file
        scaler_path: Path to the scaler file
        memory_window: Lookback window for LSTM
        
    Returns:
        Configured LSTMVolatilityForecaster instance
    """
    return LSTMVolatilityForecaster(
        model_path=model_path,
        scaler_path=scaler_path,
        memory_window=memory_window
    )
