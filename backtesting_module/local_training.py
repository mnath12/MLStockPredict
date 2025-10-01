#!/usr/bin/env python3
"""
Local LSTM Training Script

This script extracts the training pipeline from LSTM_Volatility.ipynb
and makes it runnable locally with configurable parameters.
"""

import sys
import os
import json
import argparse
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass

# TensorFlow configuration for Mac M1/M2
try:
    import tensorflow as tf
    
    # Mac-specific configuration
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("🍎 Detected Mac M1/M2 - configuring TensorFlow for compatibility...")
        # Force CPU usage on Mac M1/M2 to avoid compatibility issues
        tf.config.set_visible_devices([], 'GPU')
        # Set memory growth to prevent hanging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Configure TensorFlow for Mac
    if tf.config.list_physical_devices('GPU'):
        print("GPU available, using GPU")
        # Set memory growth to prevent hanging
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")
    else:
        print("No GPU found, using CPU")
            
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    sys.exit(1)
except Exception as e:
    print(f"TensorFlow configuration failed: {e}")
    print("Trying to continue with basic configuration...")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    print(f"Keras import failed: {e}")
    sys.exit(1)

try:
    import kerastuner as kt
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    print("Keras Tuner not available, will use default model")
    KERAS_TUNER_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Local imports
# Prefer package import; if run directly, fall back to adding project root to sys.path
try:
    from backtesting_module.data_handler import DataHandler
except ModuleNotFoundError:  # Allows: python backtesting_module/local_training.py
    import sys as _sys
    import pathlib as _pathlib
    _sys.path.append(str(_pathlib.Path(__file__).resolve().parents[1]))
    from backtesting_module.data_handler import DataHandler

# =========================
# Configuration Data Class
# =========================
@dataclass
class LSTMTrainingConfig:
    symbol: str
    start_date: str
    end_date: str
    data_frequency: str = "1Min"  # CONFIGURABLE: Use 1Min for hourly RV
    timezone: str = "America/New_York"  # CONFIGURABLE

    # LSTM sequence parameters
    memory_window: int = 60  # CONFIGURABLE: lookback periods

    # Training parameters
    batch_size: int = 16  # CONFIGURABLE
    epochs: int = 25  # CONFIGURABLE (bump to 50+ for fuller training)
    validation_split: float = 0.2  # CONFIGURABLE
    early_stopping_patience: int = 5  # CONFIGURABLE

    # Hyperparameter tuning (if Keras Tuner present)
    use_keras_tuner: bool = False  # CONFIGURABLE
    tuner_max_trials: int = 10  # CONFIGURABLE

    # Output paths
    models_dir: str = "volatility_models"  # CONFIGURABLE
    model_basename: str = "volatility_lstm_model_local"  # CONFIGURABLE


# =========================
# Data Preparation Helpers
# =========================

def compute_hourly_realized_volatility(
    price_series: pd.Series,
    tz: str = "America/New_York",
) -> pd.Series:
    """Compute hourly realized volatility from high-frequency close prices.

    RV_hourly = sqrt(sum of squared simple returns) aggregated per hour.
    """
    # Ensure tz-aware, sorted index
    px = price_series.copy()
    if px.index.tz is None:
        px.index = px.index.tz_localize(tz)
    px = px.tz_convert(tz).sort_index()

    # Simple returns and squared returns
    simple_returns = px.pct_change().dropna()
    squared = simple_returns.pow(2)

    # Right-closed hourly bins → label each bucket by its right edge time
    hourly_sum = squared.groupby(pd.Grouper(freq="H", label="right", closed="right")).sum()
    rv_hourly = np.sqrt(hourly_sum).rename("rv_hourly")

    # Ensure tz-aware index
    if rv_hourly.index.tz is None:
        rv_hourly.index = rv_hourly.index.tz_localize(tz)
    return rv_hourly.dropna()


def train_test_split_last_n_days(series: pd.Series, n_days: int = 7) -> Tuple[pd.Series, pd.Series]:
    """Split a time series into train and test where test is the last n calendar days."""
    s = series.sort_index()
    last_ts = s.index.max()
    cutoff = last_ts - pd.Timedelta(days=n_days)
    train = s.loc[s.index <= cutoff]
    test = s.loc[s.index > cutoff]
    return train, test


def prepare_lstm_sequences(
    series: pd.Series,
    scaler: Optional[MinMaxScaler],
    memory: int,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale a 1D series and build supervised sequences [samples, time, features]."""
    values_2d = series.values.reshape(-1, 1)

    fitted_scaler = scaler or MinMaxScaler(feature_range=(0, 1))
    if scaler is None:
        scaled = fitted_scaler.fit_transform(values_2d)
    else:
        scaled = fitted_scaler.transform(values_2d)

    X, y = [], []
    for i in range(memory, len(scaled)):
        X.append(scaled[i - memory : i, 0])
        y.append(scaled[i, 0])

    X_arr = np.array(X).reshape(-1, memory, 1)
    y_arr = np.array(y)
    return X_arr, y_arr, fitted_scaler


# =========================
# Model Builders
# =========================

def build_lstm_model_fast(memory: int) -> Sequential:
    """Build a compact LSTM suitable for fast local training."""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(memory, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def build_lstm_model_tuner(hp: "kt.HyperParameters", memory: int) -> Sequential:  # type: ignore
    """Hypermodel for Keras Tuner (optional)."""
    model = Sequential()
    n_layers = hp.Int("n_layers", min_value=1, max_value=2)
    for i in range(n_layers):
        units = hp.Int(f"units_{i}", min_value=16, max_value=64, step=16)
        dropout = hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.3, step=0.1)
        return_seq = i < (n_layers - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq, input_shape=(memory, 1)))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    lr = hp.Float("lr", min_value=1e-3, max_value=1e-2, sampling="log")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mean_squared_error")
    return model


# =========================
# Trainer
# =========================

def train_lstm_volatility_model(config: LSTMTrainingConfig) -> Dict[str, str]:
    """Train an LSTM model for realized volatility and save artifacts.

    Returns a dict with saved file paths.
    """
    if not tf.keras.backend.is_keras_available():
        raise RuntimeError(
            "TensorFlow is required for training. Please install tensorflow (e.g., pip install tensorflow)."
        )

    try:
        # Ensure models directory exists
        os.makedirs(config.models_dir, exist_ok=True)

        # 1) Fetch data
        print("📊 Fetching stock data...")
        data_handler = DataHandler()
        stock_df = data_handler.get_stock_bars(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            freq=config.data_frequency,
        )
        if stock_df.empty:
            raise ValueError(f"No data returned for {config.symbol} {config.start_date} → {config.end_date}")

        print(f"✅ Fetched {len(stock_df)} data points")
        close_series = stock_df["close"]

        # 2) Compute hourly RV
        print("📈 Computing hourly realized volatility...")
        rv_hourly = compute_hourly_realized_volatility(
            price_series=close_series,
            tz=config.timezone,
        )
        print(f"✅ Computed {len(rv_hourly)} hourly RV values")

        # 3) Train/test split (predict next period → shift target by -1 when building sequences)
        print("✂️ Preparing training data...")
        train_series, test_series = train_test_split_last_n_days(rv_hourly, n_days=7)

        # For supervised labels at t, target is RV at t+1 → drop last to keep alignment
        training_target_series = train_series.shift(-1).dropna()

        # Align inputs to target (remove tail element from train_series)
        aligned_train_inputs = train_series.iloc[: len(training_target_series)]

        # 4) Prepare sequences
        print(f"🧠 Preparing LSTM sequences with memory={config.memory_window}...")
        X_train, y_train, scaler = prepare_lstm_sequences(
            series=aligned_train_inputs,
            scaler=None,
            memory=config.memory_window,
        )
        print(f"✅ Prepared {len(X_train)} training sequences")

        # 5) Build model (tuner optional)
        if config.use_keras_tuner and KERAS_TUNER_AVAILABLE:
            print("🔍 Using Keras Tuner for hyperparameter optimization...")
            tuner = kt.RandomSearch(
                lambda hp: build_lstm_model_tuner(hp, config.memory_window),
                objective="val_loss",
                max_trials=config.tuner_max_trials,
                executions_per_trial=1,
                directory=os.path.join(config.models_dir, "lstm_tuning_fast"),
                project_name=f"{config.model_basename}_tuning",
            )
            
            print(f"Starting hyperparameter search with {config.tuner_max_trials} trials...")
            tuner.search(
                X_train,
                y_train,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_split=config.validation_split,
                callbacks=[EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True)],
                verbose=1,
            )
            model = tuner.get_best_models(num_models=1)[0]
            print("✅ Hyperparameter tuning completed!")
        else:
            print("🏗️ Building simple LSTM model...")
            model = build_lstm_model_fast(config.memory_window)

        # 6) Train model
        print(f"🏋️ Training model for {config.epochs} epochs...")
        history = model.fit(
            X_train,
            y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=[EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True)],
            verbose=1,
        )
        
        best_val_loss = min(history.history['val_loss'])
        print(f"✅ Training completed! Best validation loss: {best_val_loss:.6f}")

        # 7) Save artifacts
        print("💾 Saving model artifacts...")
        model_path = os.path.join(config.models_dir, f"{config.model_basename}.h5")
        scaler_path = os.path.join(config.models_dir, f"{config.model_basename}_scaler.pkl")
        config_path = os.path.join(config.models_dir, f"{config.model_basename}_config.json")

        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(config_path, "w") as f:
            json.dump(
                {
                    "symbol": config.symbol,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "data_frequency": config.data_frequency,
                    "timezone": config.timezone,
                    "memory": config.memory_window,  # CONFIGURABLE
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "validation_split": config.validation_split,
                    "early_stopping_patience": config.early_stopping_patience,
                    "use_keras_tuner": config.use_keras_tuner,
                    "best_val_loss": best_val_loss,
                    "trained_date": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        print("✅ All artifacts saved successfully!")
        return {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "config_path": config_path,
        }
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("💡 Try reducing batch size or memory window if you encounter memory issues")
        raise


# =========================
# CLI Entrypoint
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM for realized volatility (local)")
    parser.add_argument("--symbol", type=str, default="TSLA", help="Stock symbol (default TSLA)")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--memory", type=int, default=60, help="Lookback window (default 60)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (default 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--use-tuner", action="store_true", help="Use Keras Tuner if available")
    parser.add_argument("--test-only", action="store_true", help="Test TensorFlow setup only, don't train")

    args = parser.parse_args()

    # Test TensorFlow setup first
    if args.test_only:
        print("🧪 Testing TensorFlow setup only...")
        try:
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Keras available: {tf.keras.backend.is_keras_available()}")
            print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
            print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
            
            # Test simple model creation
            print("Testing simple model creation...")
            test_model = Sequential([
                Dense(10, input_shape=(1,)),
                Dense(1)
            ])
            test_model.compile(optimizer='adam', loss='mse')
            print("✅ Simple model creation successful!")
            
            # Test simple training
            print("Testing simple training...")
            X_test = np.random.random((100, 1))
            y_test = np.random.random((100, 1))
            test_model.fit(X_test, y_test, epochs=1, verbose=0)
            print("✅ Simple training successful!")
            
            print("🎉 TensorFlow setup test passed!")
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ TensorFlow setup test failed: {e}")
            sys.exit(1)

    cfg = LSTMTrainingConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        memory_window=args.memory,
        epochs=args.epochs,
        batch_size=args.batch,
        use_keras_tuner=bool(args.use_tuner and KERAS_TUNER_AVAILABLE),
    )

    if args.use_tuner and not KERAS_TUNER_AVAILABLE:
        print("[WARN] Keras Tuner not installed; proceeding without tuner.")

    try:
        artifacts = train_lstm_volatility_model(cfg)
        print("\n✅ Training complete. Artifacts saved:")
        for k, v in artifacts.items():
            print(f"  - {k}: {v}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Try --test-only to test TensorFlow setup")
        print("  2. Reduce --batch size (try 8 or 4)")
        print("  3. Reduce --memory window (try 30)")
        print("  4. Check if you have API keys set up")
        sys.exit(1)