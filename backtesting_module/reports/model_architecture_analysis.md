# LSTM Model Architecture Analysis

## Model Architecture Summary

Based on the `LSTM_Volatility.ipynb` notebook, the current LSTM model has the following architecture:

### Core Architecture
```python
def build_lstm_model_fast(hp):
    model = Sequential()
    # only 1–2 LSTM layers
    for i in range(hp.Int('n_layers', 1, 2)):
        units     = hp.Int(f'units_{i}', min_value=16, max_value=64, step=16)
        dropout   = hp.Float(f'dropout_{i}', 0.0, max_value=0.3, step=0.1)
        return_seq = (i < hp.get('n_layers') - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=(memory, 1)))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-3, 1e-2, sampling='log')
        ),
        loss='mean_squared_error'
    )
    return model
```

## Configurable Parameters

### Hyperparameter Search Space
- **n_layers**: 1-2 LSTM layers
- **units_{i}**: 16, 32, 48, 64 (step=16)
- **dropout_{i}**: 0.0, 0.1, 0.2, 0.3 (step=0.1)
- **lr**: Learning rate between 1e-3 and 1e-2 (log sampling)

### Fixed Parameters
- **memory**: 60 (lookback window) - **CONFIGURABLE**
- **input_shape**: (memory, 1) - **CONFIGURABLE**
- **loss**: 'mean_squared_error'
- **optimizer**: Adam
- **batch_size**: 16 (during training)
- **epochs**: 10 (during hyperparameter search)

### Data Preprocessing Parameters
- **scaler**: MinMaxScaler(feature_range=(0, 1)) - **CONFIGURABLE**
- **realized_volatility_window**: Hourly aggregation from 1-minute bars - **CONFIGURABLE**

## Model Files Status

### Current Files in volatility_models/
1. **volatility_lstm_model.h5** - Trained LSTM model (binary)
2. **volatility_scaler.pkl** - MinMaxScaler object (binary)

### Missing Information
- **Actual hyperparameters**: The specific values chosen by Keras Tuner
- **Training history**: Loss curves, validation metrics
- **Model summary**: Layer details, parameter counts
- **Performance metrics**: Training/validation accuracy

## Recommendations for Local Training

### 1. Extract Model Configuration
```python
# Add to training notebook
best_model = tuner_fast.get_best_models(num_models=1)[0]
best_params = tuner_fast.get_best_hyperparameters(num_trials=1)[0]

# Save configuration
config = {
    'n_layers': best_params.get('n_layers'),
    'units': [best_params.get(f'units_{i}') for i in range(best_params.get('n_layers'))],
    'dropout': [best_params.get(f'dropout_{i}') for i in range(best_params.get('n_layers'))],
    'learning_rate': best_params.get('lr'),
    'memory': memory,
    'input_shape': (memory, 1)
}

with open('volatility_models/model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### 2. Add Model Summary Logging
```python
# Add to training notebook
model.summary()
print(f"Total parameters: {model.count_params()}")
print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])}")
```

### 3. Save Training History
```python
# Add to training notebook
history = model.fit(...)
with open('volatility_models/training_history.json', 'w') as f:
    json.dump(history.history, f, indent=2)
```

## Integration Requirements

### For Backtesting System
1. **Memory window**: Must be configurable (currently hardcoded to 60)
2. **Prediction horizon**: Currently 1-step ahead, should support multi-step
3. **Retraining frequency**: Should be configurable (weekly, monthly, etc.)
4. **Data frequency**: Should support different input frequencies

### For Validation Framework
1. **Model loading**: Must handle both .h5 and config files
2. **Parameter extraction**: Should read actual hyperparameters
3. **Scaler handling**: Must properly load and apply scaler
4. **Memory management**: Handle large datasets efficiently

## Next Steps

1. **Extract actual hyperparameters** from the trained model
2. **Create configuration file** with all parameters
3. **Add model summary** to understand architecture details
4. **Implement retraining pipeline** with configurable frequency
5. **Add multi-step prediction** capability
6. **Create local training script** based on notebook 