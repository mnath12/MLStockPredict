MLStockPredict is a React web app with a Flask backend which displays stock market data and uses an LSTM neural network to make stock price predictions

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

This project requires API keys for accessing financial data. Follow these steps:

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   # Alpaca Trading API
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_SECRET_KEY=your_alpaca_secret_key_here
   
   # Polygon.io API (for options data)
   POLYGON_API_KEY=your_polygon_api_key_here
   
   # FRED API (for risk-free rate data)
   FRED_API_KEY=your_fred_api_key_here
   
   # Optional: Environment settings
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   ```

3. Get API keys from:
   - **Alpaca**: https://alpaca.markets/ (for stock and options data)
   - **Polygon.io**: https://polygon.io/ (for historical options data)
   - **FRED**: https://fred.stlouisfed.org/ (for risk-free rates)

4. The `.env` file is automatically excluded from version control (see `.gitignore`)

### 3. Verify Configuration

Test that your configuration loads correctly:
```bash
python config.py
```

This will display whether all required API keys are present.
