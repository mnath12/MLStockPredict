import os
from data_handler import DataHandler
from delta_gamma_hedging import IVForecastingPipeline

# --- Load API keys from environment variables ---
ALPACA_API_KEY = "PKCLL4TXCDLRN76OGRAB"
ALPACA_SECRET = "ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T"
POLYGON_KEY = "ejp0y0ppSQJzIX1W8qSoTIvL5ja3ctO9"
FRED_KEY = "8f0c4222bcc6a35c9b38c03da6674824"

if not all([ALPACA_API_KEY, ALPACA_SECRET, POLYGON_KEY, FRED_KEY]):
    raise RuntimeError("One or more API keys are missing. Please set ALPACA_API_KEY, ALPACA_SECRET, POLYGON_KEY, and FRED_KEY as environment variables.")

# --- Instantiate DataHandler ---
data_handler = DataHandler(
    alpaca_api_key=ALPACA_API_KEY,
    alpaca_secret=ALPACA_SECRET,
    polygon_key=POLYGON_KEY
)

# --- Instantiate the IV Forecasting Pipeline ---
pipeline = IVForecastingPipeline(
    data_handler=data_handler,
    fred_api_key=FRED_KEY,
    k_bins=11,
    j_bins=6
)

# --- User parameters ---
symbol = "AAPL"  # Change to your desired symbol
start_date = "2025-04-01"
end_date = "2025-06-01"
forecast_date = "2025-06-02"

# --- Build historical IV surfaces ---
print(f"Building historical IV surfaces for {symbol}...")
# Note: build_historical_surfaces now uses only trading days (NYSE calendar if available, else business days)
surfaces = pipeline.build_historical_surfaces(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    frequency="1W"  # Weekly surfaces
)
print(f"Built {len(surfaces)} historical surfaces.")

# --- Train HAR-RV-J model ---
print("Training HAR-RV-J model...")
pipeline.train_har_model(
    symbol=symbol,
    bars_start_date="2025-04-01",
    bars_end_date=end_date
)

# --- Generate forecast and create report ---
print("Generating forecast and report...")
report = pipeline.create_comprehensive_report(
    symbol=symbol,
    forecast_date=forecast_date
)

# --- Display plots (if running interactively) ---
try:
    import matplotlib.pyplot as plt
    for fig in report['plots'].values():
        if hasattr(fig, 'show'):
            fig.show()
        else:
            plt.show()
except ImportError:
    print("matplotlib not installed; skipping plot display.")

print("IV forecasting pipeline run complete.") 