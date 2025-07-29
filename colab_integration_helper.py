#!/usr/bin/env python3
"""
Colab Integration Helper for Volatility Forecasting

This script provides functions to use in your Google Colab notebook to:
1. Train volatility forecasting models
2. Save models in formats compatible with the backtesting system
3. Generate model performance reports
4. Export models to Google Drive for download

Usage in Colab:
    from google.colab import files
    from colab_integration_helper import *
    
    # Train your model
    model = train_volatility_model(your_data)
    
    # Save for backtesting
    save_model_for_backtesting(model, "my_volatility_model.pkl")
    
    # Download the file
    files.download("my_volatility_model.pkl")
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ColabModelTrainer:
    """
    Helper class for training volatility models in Colab.
    """
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
    
    def prepare_volatility_features(self, price_data: pd.Series, 
                                  lookback_window: int = 252) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for volatility forecasting.
        
        Args:
            price_data: Historical price series
            lookback_window: Number of days to use for feature calculation
            
        Returns:
            Tuple of (features DataFrame, target volatility series)
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
        
        # High-frequency volatility proxy
        features['hf_vol'] = features['abs_return'].rolling(5).mean() * np.sqrt(252)
        
        # Target: next day's realized volatility
        target = features['vol_21d'].shift(-1)  # Next day's 21-day volatility
        
        # Remove NaN values
        features = features.dropna()
        target = target.dropna()
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        return features, target
    
    def train_sklearn_model(self, features: pd.DataFrame, target: pd.Series, 
                           model_type: str = "random_forest") -> Any:
        """
        Train a scikit-learn model for volatility forecasting.
        
        Args:
            features: Feature DataFrame
            target: Target volatility series
            model_type: Type of model ("random_forest", "linear", "svr", "xgboost")
            
        Returns:
            Trained model
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Split data (time series aware)
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = target.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
        
        # Train model
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "svr":
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            except ImportError:
                print("XGBoost not available, using Random Forest instead")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        self.training_history[model_type] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"✅ {model_type.upper()} Model Trained:")
        print(f"   MAE: {mae*100:.2f}%")
        print(f"   RMSE: {rmse*100:.2f}%")
        print(f"   R²: {r2:.3f}")
        
        return model
    
    def train_lstm_model(self, features: pd.DataFrame, target: pd.Series,
                        sequence_length: int = 10) -> Any:
        """
        Train an LSTM model for volatility forecasting.
        
        Args:
            features: Feature DataFrame
            target: Target volatility series
            sequence_length: Number of time steps for LSTM input
            
        Returns:
            Trained LSTM model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
        except ImportError:
            print("TensorFlow not available. Install with: pip install tensorflow")
            return None
        
        # Prepare sequences for LSTM
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X.iloc[i-seq_length:i].values)
                y_seq.append(y.iloc[i])
            return np.array(X_seq), np.array(y_seq)
        
        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
        
        # Create sequences
        X_seq, y_seq = create_sequences(features_scaled, target, sequence_length)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, features.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        self.training_history['lstm'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'scaler': scaler,
            'sequence_length': sequence_length
        }
        
        print(f"✅ LSTM Model Trained:")
        print(f"   MAE: {mae*100:.2f}%")
        print(f"   RMSE: {rmse*100:.2f}%")
        print(f"   R²: {r2:.3f}")
        
        return model
    
    def save_model_for_backtesting(self, model: Any, filepath: str, 
                                  model_type: str = "sklearn") -> bool:
        """
        Save model in a format compatible with the backtesting system.
        
        Args:
            model: Trained model
            filepath: Path to save the model
            model_type: Type of model ("sklearn", "lstm", "custom")
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if model_type == "sklearn":
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            
            elif model_type == "lstm":
                # Save Keras model
                model.save(filepath)
                
                # Also save scaler and metadata
                metadata = {
                    'model_type': 'lstm',
                    'scaler': self.training_history.get('lstm', {}).get('scaler'),
                    'sequence_length': self.training_history.get('lstm', {}).get('sequence_length', 10)
                }
                
                metadata_path = filepath.replace('.h5', '_metadata.pkl')
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
            
            elif model_type == "json":
                # Save model configuration
                config = {
                    'model_type': 'custom',
                    'parameters': model.get_params() if hasattr(model, 'get_params') else {},
                    'training_history': self.training_history
                }
                with open(filepath, 'w') as f:
                    json.dump(config, f, default=str)
            
            print(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def generate_model_report(self) -> str:
        """
        Generate a comprehensive report of all trained models.
        
        Returns:
            str: Model performance report
        """
        if not self.training_history:
            return "No models trained yet."
        
        report = "📊 MODEL PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for model_name, metrics in self.training_history.items():
            report += f"Model: {model_name.upper()}\n"
            report += f"  MAE: {metrics['mae']*100:.2f}%\n"
            report += f"  RMSE: {metrics['rmse']*100:.2f}%\n"
            report += f"  R²: {metrics['r2']:.3f}\n"
            report += f"  Training samples: {metrics['n_train']}\n"
            report += f"  Test samples: {metrics['n_test']}\n\n"
        
        return report

# Convenience functions for Colab usage
def train_volatility_model(price_data: pd.Series, model_type: str = "random_forest") -> Any:
    """
    Train a volatility forecasting model.
    
    Args:
        price_data: Historical price series
        model_type: Type of model to train
        
    Returns:
        Trained model
    """
    trainer = ColabModelTrainer()
    features, target = trainer.prepare_volatility_features(price_data)
    
    if model_type in ["random_forest", "linear", "svr", "xgboost"]:
        return trainer.train_sklearn_model(features, target, model_type)
    elif model_type == "lstm":
        return trainer.train_lstm_model(features, target)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def save_model_for_backtesting(model: Any, filepath: str, model_type: str = "sklearn") -> bool:
    """
    Save a trained model for use in the backtesting system.
    
    Args:
        model: Trained model
        filepath: Path to save the model
        model_type: Type of model
        
    Returns:
        bool: True if saved successfully
    """
    trainer = ColabModelTrainer()
    return trainer.save_model_for_backtesting(model, filepath, model_type)

def download_model_from_colab(filepath: str):
    """
    Download a saved model from Colab.
    
    Args:
        filepath: Path to the model file
    """
    try:
        from google.colab import files
        files.download(filepath)
        print(f"✅ Model downloaded: {filepath}")
    except ImportError:
        print("This function only works in Google Colab")
        print(f"Model saved at: {filepath}")

# Example Colab usage:
"""
# In your Colab notebook:

# 1. Load your data
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
price_data = data['Close']

# 2. Train a model
from colab_integration_helper import *
model = train_volatility_model(price_data, "random_forest")

# 3. Save for backtesting
save_model_for_backtesting(model, "aapl_volatility_model.pkl", "sklearn")

# 4. Download the file
download_model_from_colab("aapl_volatility_model.pkl")

# 5. Get the Google Drive file ID for the backtesting system
# Upload the .pkl file to Google Drive and get the file ID
""" 