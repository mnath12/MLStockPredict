# Volatility Models Folder

This folder is used to store trained volatility forecasting models that can be used by the backtesting system.

## Supported Model Formats

The backtesting system supports the following model file formats:

- **`.pkl`** - Pickled scikit-learn models (Random Forest, Linear Regression, SVR, XGBoost)
- **`.h5`** - Keras/TensorFlow models (LSTM, Neural Networks)
- **`.keras`** - Keras model format (alternative to .h5)
- **`.json`** - Model configuration files with parameters

## How to Use

1. **Train your model** in Google Colab or locally
2. **Save the model** in one of the supported formats
3. **Place the model file** in this `volatility_models/` folder
4. **Run the backtesting system** and select option 1 for local models
5. **Choose your model** from the list of available models

## Example Model Training (Google Colab)

```python
# Train a Random Forest model
from sklearn.ensemble import RandomForestRegressor
import pickle

# Your training code here...
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
with open('volatility_model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

# Download the file from Colab
from google.colab import files
files.download('volatility_model_rf.pkl')
```

## Example Model Training (LSTM)

```python
# Train an LSTM model
import tensorflow as tf
from tensorflow import keras

# Your LSTM training code here...
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
    keras.layers.LSTM(50),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Save the model
model.save('volatility_model_lstm.h5')

# Download the file from Colab
from google.colab import files
files.download('volatility_model_lstm.h5')
```

## Model Requirements

Your trained model should:

1. **Accept price data** as input features
2. **Output volatility forecasts** as decimal values (e.g., 0.25 for 25% volatility)
3. **Be compatible** with the `VolatilityForecaster` class interface

## Dependencies

To use LSTM/Keras models (.h5/.keras files), you need:
- **TensorFlow**: `pip install tensorflow`
- **Keras**: Usually included with TensorFlow

If TensorFlow is not available, the system will automatically fall back to the EWMA method.

## Fallback Method

If no models are available or if model loading fails, the system will automatically use the fallback method:
- **Exponentially Weighted Moving Average (EWMA)** of historical volatility
- **Rolling standard deviation** of returns

This ensures the backtesting system always has a volatility estimate available.

## File Naming Convention

For better organization, consider naming your models descriptively:
- `aapl_rf_volatility_model.pkl`
- `tsla_lstm_volatility_model.h5`
- `spy_xgboost_volatility_model.pkl` 