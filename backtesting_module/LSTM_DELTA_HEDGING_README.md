# LSTM Volatility Strategy with Delta Hedging

This module implements a sophisticated trading strategy that combines LSTM (Long Short-Term Memory) neural networks for volatility prediction with delta hedging for risk management.

## Overview

The strategy works as follows:

1. **Volatility Prediction**: Uses an LSTM model to predict realized volatility based on historical price data
2. **Position Sizing**: Adjusts option positions based on predicted volatility levels
3. **Delta Hedging**: Automatically maintains delta-neutral portfolio by hedging option positions with underlying stock
4. **Risk Management**: Implements position size limits and volatility thresholds

## Key Components

### 1. LSTMVolatilityStrategy
Base strategy that implements LSTM volatility prediction and position sizing.

**Key Features:**
- Trains LSTM model on historical realized volatility data
- Predicts future volatility using sliding window approach
- Adjusts option positions based on volatility predictions
- High volatility → Increase long option positions (long gamma)
- Low volatility → Reduce option positions

### 2. LSTMVolatilityDeltaHedgedStrategy
Enhanced strategy that combines LSTM volatility prediction with automatic delta hedging.

**Key Features:**
- All features of LSTMVolatilityStrategy
- Automatic delta hedging to maintain delta-neutral portfolio
- Configurable delta tolerance and hedge symbol
- Real-time portfolio rebalancing

## Quick Start

### 1. Basic Usage

```python
from backtesting_module import LSTMVolatilityDeltaHedgedStrategy, DataHandler
import pandas as pd

# Initialize strategy
strategy = LSTMVolatilityDeltaHedgedStrategy(
    lookback_window=20,           # LSTM input window
    volatility_threshold=0.02,    # Volatility threshold for position adjustments
    delta_tolerance=0.1,          # Delta hedging tolerance
    hedge_symbol="AAPL"           # Stock symbol for hedging
)

# Train the model
data_handler = DataHandler(alpaca_key, alpaca_secret, polygon_key)
bars_df = data_handler.get_stock_bars("AAPL", "2023-01-01", "2024-01-01", "5Min")
strategy.train_lstm_model(bars_df, "AAPL")

# Make predictions
recent_volatility_data = np.array([0.02, 0.025, 0.018, ...])  # Last 20 volatility points
predicted_vol = strategy.predict_volatility(recent_volatility_data)
print(f"Predicted volatility: {predicted_vol:.4f}")
```

### 2. Full Backtest Example

```python
from backtesting_module import BacktestEngine, LSTMVolatilityDeltaHedgedStrategy

# Initialize backtest engine
engine = BacktestEngine(initial_cash=100000)

# Set up strategy
strategy = LSTMVolatilityDeltaHedgedStrategy(
    lookback_window=20,
    volatility_threshold=0.02,
    delta_tolerance=0.1,
    hedge_symbol="AAPL"
)

# Train model and run backtest
engine.strategy = strategy
engine.setup_data_handler(alpaca_key, alpaca_secret, polygon_key)

# Fetch data and train model
bars_df = engine.data_handler.get_stock_bars("AAPL", "2023-01-01", "2024-01-01", "5Min")
strategy.train_lstm_model(bars_df, "AAPL")

# Run backtest
results = engine.run_backtest()
```

### 3. Demo Script

Run the complete demo:

```bash
cd backtesting_module
python demo_lstm_delta_hedging.py
```

## Strategy Parameters

### LSTM Model Parameters
- `lookback_window`: Number of time steps for LSTM input (default: 20)
- `prediction_horizon`: Number of steps ahead to predict (default: 1)
- `model_path`: Path to pre-trained model (optional)

### Trading Parameters
- `volatility_threshold`: Threshold for volatility-based position adjustments (default: 0.02)
- `position_size_multiplier`: Multiplier for position sizing (default: 1.0)

### Delta Hedging Parameters
- `delta_tolerance`: Tolerance for delta hedging (default: 0.1)
- `hedge_symbol`: Symbol to use for delta hedging (default: "SPY")

## Strategy Logic

### Volatility-Based Position Sizing

```python
if predicted_volatility > volatility_threshold:
    # High volatility: increase long option positions
    target_position = current_position + 10 * position_size_multiplier
else:
    # Low volatility: reduce positions
    target_position = current_position * 0.8
```

### Delta Hedging

The strategy automatically:
1. Calculates current portfolio delta
2. Determines required hedge position
3. Generates orders to achieve delta-neutral portfolio
4. Maintains delta within specified tolerance

## Data Requirements

### Stock Data
- OHLCV data at desired frequency (e.g., 5-minute bars)
- Minimum 6 months of historical data for LSTM training
- Real-time or historical data from Alpaca/Polygon

### Option Data
- Option chain data with strikes, expiries, and prices
- Greeks data (delta, gamma, theta, vega)
- Implied volatility data

### Volatility Data
- Realized volatility calculated from price data
- Historical volatility for model training
- Rolling volatility windows for predictions

## Model Training

### Data Preparation
1. Calculate realized volatility from price data
2. Scale data using MinMaxScaler
3. Create sequences for LSTM training
4. Split into training/validation sets

### Training Process
```python
# Train the model
history = strategy.train_lstm_model(bars_df, symbol)

# Save trained model
strategy.save_model("lstm_volatility_model.h5")

# Load pre-trained model
strategy.load_model("lstm_volatility_model.h5")
```

## Performance Monitoring

### Strategy Metrics
- Predicted vs actual volatility accuracy
- Delta hedging effectiveness
- Portfolio P&L and Sharpe ratio
- Maximum drawdown

### Key Performance Indicators
```python
strategy_info = strategy.get_strategy_info()
print(f"Model trained: {strategy_info['model_trained']}")
print(f"Last predicted volatility: {strategy_info['last_predicted_volatility']}")
print(f"Current delta: {strategy_info.get('current_delta', 'N/A')}")
```

## Risk Management

### Position Limits
- Maximum position size per option
- Maximum portfolio delta exposure
- Volatility-based position scaling

### Stop Losses
- Volatility-based stop losses
- Delta-based stop losses
- Portfolio-level risk limits

## Integration with Existing Architecture

The strategy integrates seamlessly with your existing backtesting framework:

```
DataHandler → GreeksEngine → Portfolio → LSTMVolatilityDeltaHedgedStrategy → PositionSizer → ExecutionHandler
```

### Data Flow
1. **DataHandler**: Provides stock and option data
2. **GreeksEngine**: Calculates option Greeks
3. **LSTM Strategy**: Predicts volatility and generates targets
4. **Delta Hedging**: Applies delta hedging to targets
5. **PositionSizer**: Converts targets to orders
6. **ExecutionHandler**: Simulates order execution

## Advanced Features

### Custom Volatility Models
You can extend the strategy with different volatility models:

```python
class CustomVolatilityStrategy(LSTMVolatilityStrategy):
    def _build_lstm_model(self):
        # Custom LSTM architecture
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        return model
```

### Multi-Asset Support
The strategy can be extended to handle multiple assets:

```python
strategy = LSTMVolatilityDeltaHedgedStrategy(
    hedge_symbols=["AAPL", "SPY", "QQQ"],
    volatility_thresholds={"AAPL": 0.02, "SPY": 0.015, "QQQ": 0.025}
)
```

## Troubleshooting

### Common Issues

1. **Model Training Fails**
   - Check data quality and quantity
   - Ensure sufficient historical data
   - Verify data preprocessing

2. **Poor Volatility Predictions**
   - Adjust lookback window
   - Tune model architecture
   - Check feature engineering

3. **Delta Hedging Issues**
   - Verify Greeks calculations
   - Check option data quality
   - Adjust delta tolerance

### Debug Mode
Enable debug mode for detailed logging:

```python
strategy.set_parameters(debug=True)
```

## Future Enhancements

### Planned Features
- Real-time volatility forecasting
- Multi-timeframe analysis
- Advanced risk models
- Machine learning ensemble methods

### Research Areas
- Alternative volatility models (GARCH, HAR)
- Deep learning architectures
- Reinforcement learning integration
- Market microstructure effects

## Contributing

To contribute to this module:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This module is part of the MLStockPredict project. See the main project license for details. 