#!/usr/bin/env python3
"""
Volatility Model Validator

This module provides tools to validate and test volatility forecasting models
against realized volatility to ensure they are working correctly before
integration with the backtesting system.

Key Features:
- Compare forecasted vs realized volatility
- Calculate forecast accuracy metrics
- Visualize model performance
- Generate validation reports
- Support for LSTM models with proper preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# Import our volatility forecaster
from volatility_forecaster import VolatilityForecaster

class VolatilityModelValidator:
    """
    Validates volatility forecasting models against realized volatility.
    """
    
    def __init__(self, model_path: str, data_handler=None):
        """
        Initialize the validator.
        
        Args:
            model_path: Path to the volatility model file
            data_handler: DataHandler instance for fetching price data
        """
        self.model_path = model_path
        self.data_handler = data_handler
        self.forecaster = None
        self.validation_results = {}
        self.scaler = None
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the volatility forecasting model."""
        try:
            # Try to load the model directly first
            if self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(self.model_path)
                    print(f"✅ LSTM model loaded successfully: {self.model_path}")
                    
                    # Try to load model configuration
                    config_path = self.model_path.replace('.h5', '_config.json').replace('.keras', '_config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            self.model_config = json.load(f)
                        print(f"✅ Model config loaded: {config_path}")
                        # Extract configurable parameters
                        self.memory = self.model_config.get('memory', 60)  # CONFIGURABLE
                        print(f"   Memory window: {self.memory}")
                    else:
                        print(f"⚠️  Model config not found at: {config_path}")
                        self.memory = 60  # Default fallback
                        
                except ImportError:
                    print("TensorFlow not available for loading .h5/.keras models")
                    raise
            else:
                # Use the VolatilityForecaster
                self.forecaster = VolatilityForecaster(model_path=self.model_path)
                if not self.forecaster.is_model_loaded:
                    raise ValueError("Model failed to load")
                print(f"✅ Model loaded successfully: {self.model_path}")
                self.memory = 60  # Default for non-LSTM models
            
            # Try to load the scaler
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded: {scaler_path}")
            else:
                print(f"⚠️  Scaler not found at: {scaler_path}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def calculate_hourly_realized_volatility(self, price_data: pd.Series, tz: str = "America/New_York") -> pd.Series:
        """
        Calculate hourly realized volatility during regular US trading hours.
        This matches the preprocessing used in the LSTM training notebook.
        
        Args:
            price_data: Series of closing prices
            tz: Timezone for processing
            
        Returns:
            Series of hourly realized volatility values
        """
        # Calculate returns
        ret = price_data.pct_change().dropna()
        
        # Square returns and bin into hourly buckets
        sq = ret.pow(2)
        hourly_sum = sq.groupby(pd.Grouper(freq="H", label="right", closed="right")).sum()
        
        # Take sqrt to get realized vol
        rv_hourly = np.sqrt(hourly_sum).rename("rv_hourly")
        
        # Ensure timezone-aware index
        if rv_hourly.index.tz is None:
            rv_hourly.index = rv_hourly.index.tz_localize(tz)
        
        return rv_hourly
    
    def calculate_daily_realized_volatility(self, price_data: pd.Series, window: int = 21) -> pd.Series:
        """
        Calculate daily realized volatility using rolling window.
        
        Args:
            price_data: Series of closing prices
            window: Rolling window size (default: 21 days)
            
        Returns:
            Series of realized volatility values
        """
        # Calculate log returns
        log_returns = np.log(price_data / price_data.shift(1))
        
        # Calculate rolling realized volatility (annualized)
        realized_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        return realized_vol
    
    def prepare_lstm_features(self, rv_series: pd.Series, memory: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for LSTM model prediction.
        This matches the preprocessing used in the training notebook.
        
        Args:
            rv_series: Series of realized volatility values
            memory: Lookback window size (uses model config if None)
            
        Returns:
            Tuple of (X, y) arrays for LSTM
        """
        # Use model config memory if available, otherwise use parameter
        if memory is None:
            memory = getattr(self, 'memory', 60)  # CONFIGURABLE
        
        # Reshape to 2D array
        vals = rv_series.values.reshape(-1, 1)
        
        # Scale the data
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_vals = self.scaler.fit_transform(vals)
        else:
            scaled_vals = self.scaler.transform(vals)
        
        # Build LSTM sequences
        X, y = [], []
        for i in range(memory, len(scaled_vals)):
            X.append(scaled_vals[i - memory : i, 0])
            y.append(scaled_vals[i, 0])
        
        # Reshape into [samples, time_steps, features]
        X = np.array(X).reshape(-1, memory, 1)
        y = np.array(y)
        
        return X, y
    
    def predict_with_lstm(self, rv_series: pd.Series, memory: int = None) -> np.ndarray:
        """
        Make predictions using the LSTM model.
        
        Args:
            rv_series: Series of realized volatility values
            memory: Lookback window size (uses model config if None)
            
        Returns:
            Array of predicted volatility values
        """
        if self.model is None:
            raise ValueError("LSTM model not loaded")
        
        # Prepare features using configurable memory
        X, _ = self.prepare_lstm_features(rv_series, memory)
        
        # Make predictions
        pred_scaled = self.model.predict(X)
        
        # Inverse transform
        pred_rv = self.scaler.inverse_transform(pred_scaled).flatten()
        
        return pred_rv
    
    def generate_forecasts(self, price_data: pd.Series, forecast_horizon: int = 1, 
                          use_hourly: bool = True) -> pd.DataFrame:
        """
        Generate volatility forecasts for the entire dataset.
        
        Args:
            price_data: Series of closing prices
            forecast_horizon: Number of periods ahead to forecast
            use_hourly: Whether to use hourly or daily realized volatility
            
        Returns:
            DataFrame with dates, actual volatility, and forecasts
        """
        print("Generating volatility forecasts...")
        
        # Calculate realized volatility
        if use_hourly:
            realized_vol = self.calculate_hourly_realized_volatility(price_data)
            print(f"Using hourly realized volatility (shape: {realized_vol.shape})")
        else:
            realized_vol = self.calculate_daily_realized_volatility(price_data)
            print(f"Using daily realized volatility (shape: {realized_vol.shape})")
        
        # Generate forecasts
        forecasts = []
        dates = []
        actual_vols = []
        
        # We need at least 252 periods for meaningful forecasts
        min_data_points = 252 if not use_hourly else 252 * 6  # 6 hours per day
        
        if len(realized_vol) < min_data_points:
            print(f"Warning: Insufficient data. Need at least {min_data_points} periods, got {len(realized_vol)}")
            min_data_points = min(60, len(realized_vol) - forecast_horizon)
        
        for i in range(min_data_points, len(realized_vol) - forecast_horizon):
            current_date = realized_vol.index[i]
            
            # Get volatility data up to current date
            historical_rv = realized_vol.iloc[:i+1]
            
            try:
                # Generate forecast based on model type
                if self.model is not None:
                    # LSTM model
                    forecast = self.predict_with_lstm(historical_rv)[-1]  # Get last prediction
                elif self.forecaster is not None:
                    # Other model types
                    forecast = self.forecaster.forecast_volatility(price_data.iloc[:i+1], current_date)
                else:
                    raise ValueError("No model available for forecasting")
                
                # Get actual volatility for comparison
                if i + forecast_horizon < len(realized_vol):
                    actual_vol = realized_vol.iloc[i + forecast_horizon]
                    
                    forecasts.append(forecast)
                    dates.append(current_date)
                    actual_vols.append(actual_vol)
                    
            except Exception as e:
                print(f"Warning: Could not generate forecast for {current_date}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'forecast_vol': forecasts,
            'actual_vol': actual_vols
        })
        results_df.set_index('date', inplace=True)
        
        print(f"✅ Generated {len(results_df)} forecasts")
        return results_df
    
    def calculate_accuracy_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate accuracy metrics for the forecasts.
        
        Args:
            results_df: DataFrame with forecast and actual volatility
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Remove any NaN values
        clean_df = results_df.dropna()
        
        if len(clean_df) == 0:
            return {"error": "No valid data for comparison"}
        
        forecast_vol = clean_df['forecast_vol'].values
        actual_vol = clean_df['actual_vol'].values
        
        # Calculate metrics
        mse = mean_squared_error(actual_vol, forecast_vol)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_vol, forecast_vol)
        r2 = r2_score(actual_vol, forecast_vol)
        
        # Calculate correlation
        correlation, p_value = pearsonr(actual_vol, forecast_vol)
        
        # Calculate directional accuracy (sign of change)
        forecast_changes = np.diff(forecast_vol)
        actual_changes = np.diff(actual_vol)
        directional_accuracy = np.mean(np.sign(forecast_changes) == np.sign(actual_changes))
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs((actual_vol - forecast_vol) / actual_vol)) * 100
        
        # Calculate Theil's U (forecast vs naive forecast)
        naive_forecast = np.roll(actual_vol, 1)[1:]  # Previous value
        actual_vals = actual_vol[1:]
        forecast_vals = forecast_vol[1:]
        
        theil_u = np.sqrt(np.mean((forecast_vals - actual_vals)**2)) / np.sqrt(np.mean((naive_forecast - actual_vals)**2))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'correlation': correlation,
            'correlation_p_value': p_value,
            'directional_accuracy': directional_accuracy,
            'mape': mape,
            'theil_u': theil_u,
            'n_forecasts': len(clean_df)
        }
        
        return metrics
    
    def plot_validation_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualization plots for validation results.
        
        Args:
            results_df: DataFrame with forecast and actual volatility
            save_path: Optional path to save the plot
        """
        # Remove NaN values
        clean_df = results_df.dropna()
        
        if len(clean_df) == 0:
            print("No data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Volatility Model Validation Results\nModel: {os.path.basename(self.model_path)}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Time series comparison
        axes[0, 0].plot(clean_df.index, clean_df['actual_vol'], label='Realized Volatility', 
                       color='blue', alpha=0.7, linewidth=1)
        axes[0, 0].plot(clean_df.index, clean_df['forecast_vol'], label='Forecasted Volatility', 
                       color='red', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Volatility Forecasts vs Realized')
        axes[0, 0].set_ylabel('Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[0, 1].scatter(clean_df['actual_vol'], clean_df['forecast_vol'], alpha=0.6)
        max_val = max(clean_df['actual_vol'].max(), clean_df['forecast_vol'].max())
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Forecast')
        axes[0, 1].set_xlabel('Realized Volatility')
        axes[0, 1].set_ylabel('Forecasted Volatility')
        axes[0, 1].set_title('Forecast vs Actual Scatter Plot')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        residuals = clean_df['forecast_vol'] - clean_df['actual_vol']
        axes[1, 0].plot(clean_df.index, residuals, color='green', alpha=0.7)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Forecast Residuals')
        axes[1, 0].set_ylabel('Residual (Forecast - Actual)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        
        plt.show()
    
    def generate_validation_report(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            results_df: DataFrame with forecast and actual volatility
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        # Calculate metrics
        metrics = self.calculate_accuracy_metrics(results_df)
        
        # Generate report
        report = f"""
VOLATILITY MODEL VALIDATION REPORT
==================================

Model: {os.path.basename(self.model_path)}
Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of Forecasts: {metrics.get('n_forecasts', 'N/A')}

ACCURACY METRICS:
----------------
Mean Squared Error (MSE): {metrics.get('mse', 'N/A'):.6f}
Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.6f}
Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.6f}
Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 'N/A'):.2f}%
R-squared (R²): {metrics.get('r2', 'N/A'):.4f}
Correlation: {metrics.get('correlation', 'N/A'):.4f}
Correlation p-value: {metrics.get('correlation_p_value', 'N/A'):.6f}
Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2%}
Theil's U: {metrics.get('theil_u', 'N/A'):.4f}

INTERPRETATION:
--------------
• RMSE: Lower is better (measures forecast error magnitude)
• MAE: Lower is better (measures absolute forecast error)
• MAPE: Lower is better (measures relative forecast error)
• R²: Higher is better (measures explained variance, max 1.0)
• Correlation: Higher is better (measures linear relationship)
• Directional Accuracy: Higher is better (measures trend prediction)
• Theil's U: < 1.0 means model beats naive forecast

RECOMMENDATIONS:
---------------
"""
        
        # Add recommendations based on metrics
        if metrics.get('r2', 0) > 0.7:
            report += "✅ R² > 0.7: Model shows good explanatory power\n"
        elif metrics.get('r2', 0) > 0.5:
            report += "⚠️  R² between 0.5-0.7: Model shows moderate explanatory power\n"
        else:
            report += "❌ R² < 0.5: Model shows poor explanatory power\n"
        
        if metrics.get('correlation', 0) > 0.7:
            report += "✅ Correlation > 0.7: Strong linear relationship with actual volatility\n"
        elif metrics.get('correlation', 0) > 0.5:
            report += "⚠️  Correlation between 0.5-0.7: Moderate linear relationship\n"
        else:
            report += "❌ Correlation < 0.5: Weak linear relationship\n"
        
        if metrics.get('directional_accuracy', 0) > 0.6:
            report += "✅ Directional accuracy > 60%: Good at predicting volatility direction\n"
        else:
            report += "❌ Directional accuracy < 60%: Poor at predicting volatility direction\n"
        
        if metrics.get('mape', 100) < 20:
            report += "✅ MAPE < 20%: Low relative forecast error\n"
        elif metrics.get('mape', 100) < 40:
            report += "⚠️  MAPE between 20-40%: Moderate relative forecast error\n"
        else:
            report += "❌ MAPE > 40%: High relative forecast error\n"
        
        if metrics.get('theil_u', 1) < 1.0:
            report += "✅ Theil's U < 1.0: Model beats naive forecast\n"
        else:
            report += "❌ Theil's U > 1.0: Model worse than naive forecast\n"
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to: {save_path}")
        
        return report
    
    def validate_model(self, symbol: str, start_date: str, end_date: str, 
                      freq: str = "1D", save_results: bool = True, 
                      use_hourly: bool = True) -> Dict[str, Any]:
        """
        Complete validation process for a volatility model.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for validation
            end_date: End date for validation
            freq: Data frequency
            save_results: Whether to save plots and reports
            use_hourly: Whether to use hourly or daily realized volatility
            
        Returns:
            Dictionary with validation results
        """
        print(f"🔍 Validating volatility model for {symbol} from {start_date} to {end_date}")
        print(f"Using {'hourly' if use_hourly else 'daily'} realized volatility")
        
        # Fetch price data
        if self.data_handler is None:
            raise ValueError("DataHandler is required for validation")
        
        # For hourly volatility, we need higher frequency data
        if use_hourly:
            data_freq = "1Min"  # Use 1-minute data for hourly RV calculation
        else:
            data_freq = freq
        
        price_data = self.data_handler.get_stock_bars(symbol, start_date, end_date, data_freq)
        if price_data.empty:
            raise ValueError(f"No price data found for {symbol}")
        
        # Generate forecasts
        results_df = self.generate_forecasts(price_data['close'], use_hourly=use_hourly)
        
        # Calculate metrics
        metrics = self.calculate_accuracy_metrics(results_df)
        
        # Generate report
        report = self.generate_validation_report(results_df)
        
        # Create output directory
        if save_results:
            output_dir = f"validation_results_{symbol}_{start_date}_{end_date}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save plots
            plot_path = os.path.join(output_dir, "validation_plots.png")
            self.plot_validation_results(results_df, plot_path)
            
            # Save report
            report_path = os.path.join(output_dir, "validation_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Save results data
            results_path = os.path.join(output_dir, "validation_results.csv")
            results_df.to_csv(results_path)
            
            print(f"✅ Validation results saved to: {output_dir}/")
        
        # Print summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(report)
        
        return {
            'metrics': metrics,
            'results_df': results_df,
            'report': report
        }


def validate_all_models(symbol: str, start_date: str, end_date: str, 
                       models_dir: str = "../volatility_models",
                       custom_memory: Optional[int] = None,
                       use_hourly: bool = True) -> Dict[str, Any]:
    """
    Validate all models in the volatility_models directory.
    
    Args:
        symbol: Stock symbol to test
        start_date: Start date for validation
        end_date: End date for validation
        models_dir: Directory containing model files
        
    Returns:
        Dictionary with validation results for all models
    """
    from data_handler import DataHandler
    
    # Initialize data handler
    data_handler = DataHandler()
    
    # Get all model files
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith(('.pkl', '.h5', '.keras', '.json')):
            model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        print("No model files found in volatility_models directory")
        return {}
    
    print(f"Found {len(model_files)} model files to validate")
    
    # Validate each model
    results = {}
    for model_path in model_files:
        try:
            print(f"\n{'='*60}")
            print(f"Validating: {os.path.basename(model_path)}")
            print(f"{'='*60}")
            
            validator = VolatilityModelValidator(model_path, data_handler)
            
            # Override memory if custom memory is specified
            if custom_memory is not None:
                validator.memory = custom_memory
                print(f"   Using custom memory window: {custom_memory}")
            
            # Determine if this is an LSTM model (use hourly RV)
            model_use_hourly = model_path.endswith(('.h5', '.keras'))
            final_use_hourly = use_hourly if use_hourly != model_use_hourly else model_use_hourly
            
            result = validator.validate_model(symbol, start_date, end_date, use_hourly=final_use_hourly)
            results[os.path.basename(model_path)] = result
            
        except Exception as e:
            print(f"❌ Error validating {model_path}: {e}")
            results[os.path.basename(model_path)] = {'error': str(e)}
    
    # Generate comparison report
    comparison_report = generate_model_comparison_report(results)
    print("\n" + comparison_report)
    
    return results


def generate_model_comparison_report(results: Dict[str, Any]) -> str:
    """
    Generate a comparison report for multiple models.
    
    Args:
        results: Dictionary with validation results for each model
        
    Returns:
        Comparison report text
    """
    report = """
MODEL COMPARISON REPORT
======================

"""
    
    # Extract metrics for comparison
    comparison_data = []
    for model_name, result in results.items():
        if 'error' in result:
            report += f"{model_name}: ❌ ERROR - {result['error']}\n"
            continue
        
        metrics = result.get('metrics', {})
        comparison_data.append({
            'model': model_name,
            'rmse': metrics.get('rmse', float('inf')),
            'mae': metrics.get('mae', float('inf')),
            'r2': metrics.get('r2', float('-inf')),
            'correlation': metrics.get('correlation', 0),
            'directional_accuracy': metrics.get('directional_accuracy', 0),
            'mape': metrics.get('mape', float('inf')),
            'theil_u': metrics.get('theil_u', float('inf'))
        })
    
    if not comparison_data:
        return report + "No valid models to compare"
    
    # Sort by R² (best first)
    comparison_data.sort(key=lambda x: x['r2'], reverse=True)
    
    report += "\nRANKING BY R² (Best to Worst):\n"
    report += "-" * 40 + "\n"
    
    for i, data in enumerate(comparison_data, 1):
        report += f"{i}. {data['model']}\n"
        report += f"   R²: {data['r2']:.4f}\n"
        report += f"   RMSE: {data['rmse']:.6f}\n"
        report += f"   Correlation: {data['correlation']:.4f}\n"
        report += f"   Directional Accuracy: {data['directional_accuracy']:.2%}\n"
        report += f"   MAPE: {data['mape']:.2f}%\n"
        report += f"   Theil's U: {data['theil_u']:.4f}\n\n"
    
    # Find best model in each category
    best_rmse = min(comparison_data, key=lambda x: x['rmse'])
    best_r2 = max(comparison_data, key=lambda x: x['r2'])
    best_correlation = max(comparison_data, key=lambda x: x['correlation'])
    best_directional = max(comparison_data, key=lambda x: x['directional_accuracy'])
    best_theil = min(comparison_data, key=lambda x: x['theil_u'])
    
    report += "BEST MODELS BY METRIC:\n"
    report += "-" * 25 + "\n"
    report += f"Lowest RMSE: {best_rmse['model']} ({best_rmse['rmse']:.6f})\n"
    report += f"Highest R²: {best_r2['model']} ({best_r2['r2']:.4f})\n"
    report += f"Best Correlation: {best_correlation['model']} ({best_correlation['correlation']:.4f})\n"
    report += f"Best Directional: {best_directional['model']} ({best_directional['directional_accuracy']:.2%})\n"
    report += f"Best Theil's U: {best_theil['model']} ({best_theil['theil_u']:.4f})\n"
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Volatility Model Validator")
    print("=" * 30)
    
    # You can run this directly to validate models
    # Example:
    # validate_all_models("TSLA", "2024-01-01", "2024-12-31") 