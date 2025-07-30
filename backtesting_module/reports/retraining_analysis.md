# Retraining Analysis and Implementation Requirements

## Current State Analysis

### What's Already Implemented
1. **Basic model loading** in `volatility_forecaster.py`
2. **Batch prediction setup** in `main.py` (lines 490-510)
3. **Individual prediction** capability
4. **Fallback methods** (EWMA, rolling std)

### What's Missing for Proper Retraining

## 1. Retraining Frequency Configuration

### Current Implementation Gap
The current system loads a static model and uses it for the entire backtest period. No retraining occurs during the backtest.

### Required Implementation
```python
# CONFIGURABLE PARAMETERS (should be user-settable)
RETRAINING_FREQUENCY = "weekly"  # Options: daily, weekly, monthly, quarterly
RETRAINING_LOOKBACK_DAYS = 252   # How much historical data to use for retraining
MIN_RETRAINING_DATA_POINTS = 1000 # Minimum data points required for retraining
MODEL_PERFORMANCE_THRESHOLD = 0.3 # R² threshold below which to retrain
```

## 2. Rolling Window Training Process

### Current Process (Static)
1. Load pre-trained model
2. Use model for entire backtest period
3. No adaptation to changing market conditions

### Required Process (Dynamic)
1. **Initial Setup**: Load pre-trained model or train from scratch
2. **During Backtest**: 
   - Monitor model performance
   - Retrain at specified frequency
   - Update model weights
   - Maintain prediction continuity

### Implementation Requirements
```python
class RollingWindowTrainer:
    def __init__(self, retraining_freq="weekly", lookback_days=252):
        self.retraining_freq = retraining_freq
        self.lookback_days = lookback_days
        self.last_retraining_date = None
        self.model_performance_history = []
    
    def should_retrain(self, current_date, current_performance):
        # Check if retraining is due based on frequency
        # Check if performance has degraded
        pass
    
    def retrain_model(self, historical_data, current_date):
        # Implement retraining logic
        # Save new model version
        # Update performance tracking
        pass
```

## 3. Multi-Step Prediction Capability

### Current Limitation
The LSTM model is trained for 1-step ahead prediction, but the backtesting system might need multi-step forecasts.

### Required Enhancement
```python
def predict_multiple_steps(self, data, steps_ahead=5):
    """
    Predict multiple steps ahead using recursive prediction
    or train a multi-output model
    """
    predictions = []
    current_data = data.copy()
    
    for step in range(steps_ahead):
        pred = self.model.predict(current_data)
        predictions.append(pred)
        
        # Update input data for next prediction
        # This requires careful handling of the rolling window
        current_data = self.update_input_window(current_data, pred)
    
    return predictions
```

## 4. Model Versioning and Performance Tracking

### Current Gap
No tracking of model versions or performance over time.

### Required Implementation
```python
class ModelVersionManager:
    def __init__(self):
        self.model_versions = []
        self.performance_history = []
    
    def save_model_version(self, model, performance_metrics, date):
        version_info = {
            'date': date,
            'model_path': f"volatility_models/model_v{len(self.model_versions)}.h5",
            'performance': performance_metrics,
            'config': self.extract_model_config(model)
        }
        self.model_versions.append(version_info)
        model.save(version_info['model_path'])
    
    def get_best_model(self, date):
        # Return the best performing model up to the given date
        pass
```

## 5. Integration with Main Backtesting System

### Current Integration Points
- `main.py` lines 490-510: Batch prediction setup
- `volatility_forecaster.py`: Model loading and prediction

### Required Enhancements

#### In `main.py`:
```python
# Add retraining configuration
print("🔄 Retraining Configuration:")
print("1. No retraining (static model)")
print("2. Weekly retraining")
print("3. Monthly retraining")
print("4. Performance-based retraining")
retraining_choice = input("Choose retraining strategy (1-4) [default: 1]: ")

# Add retraining frequency
if retraining_choice in ["2", "3", "4"]:
    lookback_days = int(input("Enter lookback days for retraining [default: 252]: ") or "252")
    performance_threshold = float(input("Enter performance threshold (R²) [default: 0.3]: ") or "0.3")
```

#### In `volatility_forecaster.py`:
```python
class VolatilityForecaster:
    def __init__(self, model_path=None, retraining_config=None):
        self.retraining_config = retraining_config
        self.rolling_trainer = RollingWindowTrainer(retraining_config)
    
    def forecast_volatility(self, price_data, forecast_date):
        # Check if retraining is needed
        if self.should_retrain(forecast_date):
            self.retrain_model(price_data, forecast_date)
        
        # Make prediction
        return self.predict(price_data)
```

## 6. Local Training Pipeline

### Current Gap
Training is done in Colab notebook, no local training capability.

### Required Implementation
```python
# Create local_training.py
class LocalTrainer:
    def __init__(self, config):
        self.config = config
    
    def prepare_data(self, symbol, start_date, end_date):
        # Fetch data using data_handler
        # Calculate realized volatility
        # Prepare sequences
        pass
    
    def train_model(self, X_train, y_train, X_val, y_val):
        # Implement training with hyperparameter tuning
        # Save model and configuration
        pass
    
    def evaluate_model(self, model, X_test, y_test):
        # Calculate performance metrics
        # Generate validation plots
        pass
```

## Implementation Priority

### Phase 1: Basic Retraining (Week 1)
1. Add retraining frequency configuration
2. Implement basic rolling window retraining
3. Add model versioning

### Phase 2: Advanced Features (Week 2)
1. Performance-based retraining
2. Multi-step prediction
3. Local training pipeline

### Phase 3: Optimization (Week 3)
1. Memory optimization
2. Parallel training
3. Advanced model selection

## Code Changes Required

### Files to Modify:
1. `volatility_forecaster.py` - Add retraining logic
2. `main.py` - Add retraining configuration
3. `volatility_model_validator.py` - Support for model versioning

### New Files to Create:
1. `rolling_window_trainer.py` - Core retraining logic
2. `model_version_manager.py` - Model versioning
3. `local_training.py` - Local training pipeline
4. `retraining_config.py` - Configuration management 