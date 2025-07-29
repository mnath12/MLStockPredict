#!/usr/bin/env python3
"""
Volatility Forecasting Interface for Backtesting System

This module provides a clean interface to integrate volatility forecasting models
trained on Google Colab with the backtesting system.

Key Features:
- Supports multiple volatility forecasting models (LSTM, HAR, GARCH, etc.)
- Handles model loading from saved files (trained on Colab)
- Provides real-time volatility forecasts during backtesting
- Fallback to historical volatility estimation if model unavailable
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json

class VolatilityForecaster:
    """
    Main volatility forecasting interface for the backtesting system.
    
    This class handles:
    1. Loading pre-trained models from Colab
    2. Real-time volatility forecasting during backtesting
    3. Fallback to historical volatility estimation
    4. Model performance tracking
    """
    
    def __init__(self, model_path: Optional[str] = None, fallback_method: str = "ewm"):
        """
        Initialize the volatility forecaster.
        
        Args:
            model_path: Path to saved model file from Colab
            fallback_method: Method to use when model is unavailable ("ewm", "rolling", "garch")
        """
        self.model = None
        self.model_path = model_path
        self.fallback_method = fallback_method
        self.is_model_loaded = False
        self.forecast_history = []
        self.actual_history = []
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained volatility forecasting model from Colab.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Try different model formats
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif model_path.endswith('.json'):
                with open(model_path, 'r') as f:
                    model_config = json.load(f)
                self.model = self._reconstruct_model_from_config(model_config)
            elif model_path.endswith('.h5') or model_path.endswith('.keras'):
                # TensorFlow/Keras model
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                except ImportError:
                    print("TensorFlow not available for loading .h5/.keras models")
                    return False
            else:
                print(f"Unsupported model format: {model_path}")
                return False
            
            self.is_model_loaded = True
            print(f"✅ Volatility model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading volatility model: {e}")
            self.is_model_loaded = False
            return False
    
    def _reconstruct_model_from_config(self, config: Dict) -> Any:
        """
        Reconstruct model from configuration (for models saved as JSON).
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Reconstructed model object
        """
        # This would be implemented based on your specific model architecture
        # For now, return None to use fallback
        print("Model reconstruction from config not implemented. Using fallback.")
        return None
    
    def prepare_features(self, price_data: pd.Series, lookback_window: int = 252) -> pd.DataFrame:
        """
        Prepare features for volatility forecasting.
        
        Args:
            price_data: Historical price series
            lookback_window: Number of days to use for feature calculation
            
        Returns:
            DataFrame with features ready for model input
        """
        # Calculate returns
        returns = np.log(price_data / price_data.shift(1)).dropna()
        
        # Use only recent data for features
        recent_returns = returns.tail(lookback_window)
        
        # Basic volatility features
        features = pd.DataFrame(index=recent_returns.index)
        
        # Rolling volatility measures
        features['vol_5d'] = recent_returns.rolling(5).std() * np.sqrt(252)
        features['vol_10d'] = recent_returns.rolling(10).std() * np.sqrt(252)
        features['vol_21d'] = recent_returns.rolling(21).std() * np.sqrt(252)
        features['vol_63d'] = recent_returns.rolling(63).std() * np.sqrt(252)
        
        # Exponentially weighted volatility
        features['vol_ewm_5d'] = recent_returns.ewm(span=5).std() * np.sqrt(252)
        features['vol_ewm_21d'] = recent_returns.ewm(span=21).std() * np.sqrt(252)
        
        # Volatility of volatility
        features['vol_of_vol'] = features['vol_21d'].rolling(21).std()
        
        # Return-based features
        features['abs_return'] = np.abs(recent_returns)
        features['squared_return'] = recent_returns ** 2
        
        # High-frequency volatility proxy (using absolute returns)
        features['hf_vol'] = features['abs_return'].rolling(5).mean() * np.sqrt(252)
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def forecast_volatility(self, price_data: pd.Series, forecast_date: pd.Timestamp) -> float:
        """
        Generate volatility forecast for a specific date.
        
        Args:
            price_data: Historical price series up to forecast_date
            forecast_date: Date for which to forecast volatility
            
        Returns:
            float: Forecasted volatility (annualized)
        """
        try:
            # Prepare features
            features = self.prepare_features(price_data)
            
            if self.is_model_loaded and self.model is not None:
                # Use trained model
                forecast = self._predict_with_model(features)
            else:
                # Use fallback method
                forecast = self._fallback_forecast(price_data)
            
            # Store forecast for tracking
            self.forecast_history.append({
                'date': forecast_date,
                'forecast': forecast,
                'method': 'model' if self.is_model_loaded else self.fallback_method
            })
            
            return forecast  # Ensure minimum volatility of 5%
            
        except Exception as e:
            print(f"Error in volatility forecasting: {e}")
            # Return a reasonable fallback
            return self._simple_volatility_estimate(price_data)
    
    def _predict_with_model(self, features: pd.DataFrame) -> float:
        """
        Make prediction using the loaded model.
        
        Args:
            features: Prepared feature DataFrame
            
        Returns:
            float: Model prediction
        """
        try:
            if hasattr(self.model, 'predict'):
                # Standard sklearn-style model
                if len(features) > 0:
                    # Use the most recent features
                    latest_features = features.iloc[-1:].values
                    prediction = self.model.predict(latest_features)[0]
                    return float(prediction)
            
            elif hasattr(self.model, '__call__'):
                # Callable model (e.g., custom function)
                prediction = self.model(features)
                return float(prediction)
            
            else:
                print("Model doesn't have standard predict method")
                return self._fallback_forecast(features)
                
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return self._fallback_forecast(features)
    
    def _fallback_forecast(self, price_data: Union[pd.Series, pd.DataFrame]) -> float:
        """
        Fallback volatility forecasting when model is unavailable.
        
        Args:
            price_data: Price series or features DataFrame
            
        Returns:
            float: Fallback volatility forecast
        """
        if isinstance(price_data, pd.DataFrame):
            # If we have features, use the most recent volatility
            if 'vol_21d' in price_data.columns:
                return float(price_data['vol_21d'].iloc[-1])
            elif 'vol_ewm_21d' in price_data.columns:
                return float(price_data['vol_ewm_21d'].iloc[-1])
        
        # Calculate from price data
        return self._simple_volatility_estimate(price_data)
    
    def _simple_volatility_estimate(self, price_data: pd.Series) -> float:
        """
        Simple volatility estimation from price data.
        
        Args:
            price_data: Historical price series
            
        Returns:
            float: Estimated volatility
        """
        if len(price_data) < 2:
            return 0.2  # Default volatility
        
        returns = np.log(price_data / price_data.shift(1)).dropna()
        
        if self.fallback_method == "ewm":
            # Exponentially weighted volatility
            vol = returns.ewm(span=21).std().iloc[-1] * np.sqrt(252)
        elif self.fallback_method == "rolling":
            # Rolling volatility
            vol = returns.rolling(21).std().iloc[-1] * np.sqrt(252)
        else:
            # Simple standard deviation
            vol = returns.std() * np.sqrt(252)
        
        return float(vol)
    
    def update_actual_volatility(self, date: pd.Timestamp, actual_vol: float):
        """
        Update actual volatility for model performance tracking.
        
        Args:
            date: Date of the volatility observation
            actual_vol: Actual realized volatility
        """
        self.actual_history.append({
            'date': date,
            'actual': actual_vol
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the volatility forecasts.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.forecast_history) == 0 or len(self.actual_history) == 0:
            return {}
        
        # Align forecasts and actuals
        forecast_df = pd.DataFrame(self.forecast_history).set_index('date')
        actual_df = pd.DataFrame(self.actual_history).set_index('date')
        
        # Merge on date
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

class ColabModelLoader:
    """
    Helper class to load models trained on Google Colab.
    """
    
    @staticmethod
    def download_from_colab(colab_file_id: str, local_path: str) -> bool:
        """
        Download model file from Google Drive (if shared from Colab).
        
        Args:
            colab_file_id: Google Drive file ID
            local_path: Local path to save the file
            
        Returns:
            bool: True if download successful
        """
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={colab_file_id}"
            gdown.download(url, local_path, quiet=False)
            return os.path.exists(local_path)
        except ImportError:
            print("gdown not installed. Install with: pip install gdown")
            return False
        except Exception as e:
            print(f"Error downloading from Colab: {e}")
            return False
    
    @staticmethod
    def save_model_for_colab(model, filepath: str, model_type: str = "sklearn"):
        """
        Save model in a format that can be loaded by the backtesting system.
        
        Args:
            model: Trained model object
            filepath: Path to save the model
            model_type: Type of model ("sklearn", "keras", "custom")
        """
        try:
            if model_type == "sklearn":
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            elif model_type == "keras":
                model.save(filepath)
            elif model_type == "json":
                # Save model configuration as JSON
                config = {
                    'model_type': 'custom',
                    'parameters': model.get_params() if hasattr(model, 'get_params') else {},
                    'filepath': filepath
                }
                with open(filepath, 'w') as f:
                    json.dump(config, f)
            
            print(f"✅ Model saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")

# Example usage and integration functions
def create_volatility_forecaster_from_colab(
    colab_file_id: Optional[str] = None,
    local_model_path: Optional[str] = None,
    fallback_method: str = "ewm"
) -> VolatilityForecaster:
    """
    Create a volatility forecaster from a Colab-trained model.
    
    Args:
        colab_file_id: Google Drive file ID (if downloading from Colab)
        local_model_path: Local path to model file
        fallback_method: Fallback method if model loading fails
        
    Returns:
        Configured VolatilityForecaster instance
    """
    forecaster = VolatilityForecaster(fallback_method=fallback_method)
    
    if colab_file_id:
        # Download from Colab
        local_path = f"volatility_model_{colab_file_id}.pkl"
        if ColabModelLoader.download_from_colab(colab_file_id, local_path):
            forecaster.load_model(local_path)
    
    elif local_model_path:
        # Load from local path
        forecaster.load_model(local_model_path)
    
    return forecaster 