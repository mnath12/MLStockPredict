# LSTM Volatility Model Integration Guide

This guide explains how to integrate and use your LSTM volatility model trained on Google Colab with the backtesting system.

## Overview

The integration allows you to use your pre-trained LSTM model for volatility forecasting during backtesting, providing more accurate volatility predictions for your straddle strategy.

## Files Structure

```
volatility_models/
├── volatility_lstm_model.h5      # Your trained LSTM model from Colab
├── volatility_scaler.pkl          # Your trained scaler from Colab
└── README.md                      # This file

backtesting_module/
├── lstm_volatility_forecaster.py  # LSTM-specific forecaster
├── volatility_forecaster.py       # General volatility forecaster
└── main.py                        # Main backtesting script
```

## Setup Instructions

### 1. Place Your Model Files

Ensure your Colab-trained model files are in the `volatility_models/` folder:

- `volatility_lstm_model.h5` - Your trained LSTM model
- `volatility_scaler.pkl` - Your trained scaler

### 2. Install Dependencies

Make sure you have TensorFlow installed:

```bash
pip install tensorflow
```

### 3. Test the Integration

Run the test script to verify everything works:

```bash
python test_lstm_integration.py
```

This will test:
- Model loading
- Data fetching
- Volatility forecasting

## Using the LSTM Model in Backtesting

### 1. Run the Main Script

```bash
python backtesting_module/main.py
```

### 2. Choose LSTM Volatility Forecasting

When prompted for volatility forecasting method, choose option **1**:

```
🔮 Volatility Forecasting Setup:
1. Use LSTM model from volatility_models folder (recommended)
2. Use other local model from volatility_models folder
3. Use fallback method (exponentially weighted)

Choose volatility forecasting method (1/2/3) [default: 1]: 1
```

### 3. Enter Your Parameters

The system will prompt you for:

- **Stock symbol**: e.g., `AAPL`, `TSLA`, `GOOGL`
- **Start date**: e.g., `2024-01-01`
- **End date**: e.g., `2024-12-31`
- **Frequency**: e.g., `1D` (daily), `1H` (hourly)
- **Rebalancing frequency**: Daily, weekly, monthly, or every bar
- **Delta threshold**: Minimum delta deviation to trigger rebalancing

### 4. Select Options

Choose your straddle options (call and put with same strike and expiry).

## How the LSTM Model Works

### Data Processing

1. **Price Data**: The system fetches minute-level price data for the stock
2. **Hourly Realized Volatility**: Computes hourly realized volatility using the same method as your Colab training
3. **LSTM Input**: Prepares 60-hour lookback windows for the LSTM model
4. **Scaling**: Uses your trained scaler to normalize the data
5. **Prediction**: Makes volatility forecasts using the LSTM model

### Integration Points

- **Initial Forecast**: Used to determine initial straddle position
- **Rebalancing**: Used at each rebalancing point to update positions
- **Performance Tracking**: Monitors forecast accuracy vs. actual volatility

## Model Parameters

The LSTM forecaster uses these parameters (matching your Colab training):

- **Memory Window**: 60 hours (lookback period)
- **Input Format**: Hourly realized volatility
- **Scaling**: MinMaxScaler (0, 1)
- **Timezone**: America/New_York

## Error Handling

The system includes comprehensive error handling:

- **Model Loading Failures**: Falls back to EWMA volatility estimation
- **Data Issues**: Handles missing or insufficient data gracefully
- **Prediction Errors**: Falls back to simple volatility estimation
- **Invalid Forecasts**: Validates forecast values and uses fallbacks

## Performance Monitoring

The system tracks LSTM model performance:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Correlation**: Forecast vs. actual correlation
- **Forecast Count**: Number of predictions made

## Troubleshooting

### Common Issues

1. **Model Files Not Found**
   ```
   ❌ LSTM model files not found
   ```
   - Ensure `volatility_lstm_model.h5` and `volatility_scaler.pkl` are in `volatility_models/`

2. **TensorFlow Not Available**
   ```
   ❌ TensorFlow not available for loading LSTM model
   ```
   - Install TensorFlow: `pip install tensorflow`

3. **Insufficient Data**
   ```
   ❌ Not enough historical data for LSTM input
   ```
   - Use a longer date range or higher frequency data

4. **API Key Issues**
   ```
   ❌ Error fetching stock data
   ```
   - Check your Alpaca and Polygon API keys

### Debug Mode

For detailed error information, check the console output. The system provides detailed error messages and stack traces for debugging.

## Example Usage

Here's a complete example of running the backtesting system with LSTM:

```bash
python backtesting_module/main.py

# Enter API keys (or leave blank for demo)
Enter Alpaca API key (or leave blank): 
Enter Alpaca API secret (or leave blank): 
Enter Polygon API key (or leave blank): 

# Choose LSTM volatility forecasting
Choose volatility forecasting method (1/2/3) [default: 1]: 1

# Enter parameters
Enter stock symbol (e.g., AAPL): TSLA
Enter start date (YYYY-MM-DD): 2024-01-01
Enter end date (YYYY-MM-DD): 2024-12-31
Enter frequency (e.g., 1D, 5Min) [default: 1D]: 1D

# Select straddle options
# ... (follow prompts for option selection)

# System will use LSTM for volatility forecasting
🤖 Loading LSTM volatility model...
✅ LSTM volatility model loaded successfully
   Using hourly realized volatility forecasting
```

## Performance Expectations

With your LSTM model, you should see:

- More accurate volatility forecasts compared to simple methods
- Better timing of straddle entries and exits
- Improved risk-adjusted returns
- More stable delta hedging

The system will show LSTM-specific performance metrics at the end of the backtest.

## Next Steps

1. **Test with Different Stocks**: Try the system with various liquid stocks
2. **Adjust Parameters**: Experiment with different rebalancing frequencies and thresholds
3. **Monitor Performance**: Track the LSTM model's forecasting accuracy
4. **Retrain Model**: If needed, retrain the LSTM model with more recent data

## Support

If you encounter issues:

1. Run the test script: `python test_lstm_integration.py`
2. Check the error messages in the console
3. Verify your model files are correctly placed
4. Ensure all dependencies are installed

The system is designed to be robust and will fall back to simpler methods if the LSTM model fails, ensuring your backtesting can continue.