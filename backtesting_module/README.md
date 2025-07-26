# Backtesting Module

This module provides a flexible, object-oriented framework for backtesting stock and option trading strategies, including support for advanced hedging and RL-based strategies.

## Folder Structure

```
backtesting_module/
├── __init__.py
├── data_handler.py
├── entities.py
├── execution_handler.py
├── greeks_engine.py
├── portfolio.py
├── position_sizer.py
├── strategy.py
└── tests/
    └── data_test.py
```

## Main Components

- **DataHandler**: Unified interface for fetching stock and option data from Alpaca and Polygon APIs.
- **GreeksEngine**: Computes option Greeks and related analytics.
- **Portfolio**: Tracks current holdings of stocks and options.
- **PositionSizer**: Determines order sizes based on strategy targets and portfolio state.
- **ExecutionHandler**: Simulates order execution and fills.
- **Strategy**: Contains base strategy logic, mixins for hedging, and custom strategies (e.g., RL, LSTM, jump diffusion).
- **Entities**: Data classes for Order, Fill, Position, Option, and Stock.
- **tests/**: Unit tests for module components.

## Usage Example

```python
from backtesting_module.data_handler import DataHandler

dh = DataHandler(
    alpaca_api_key="YOUR_ALPACA_KEY",
    alpaca_secret="YOUR_ALPACA_SECRET",
    polygon_key="YOUR_POLYGON_KEY"
)

bars = dh.get_stock_bars("AAPL", "2023-01-01", "2023-06-30", "5Min")
print(bars.head())
```

## Requirements
- Python 3.11+
- alpaca-py==0.42.0
- polygon-api-client==1.15.1
- pandas, numpy, matplotlib, etc.

Because `tests/` sits **inside** the `backtesting_module` package, treat the file as an *in-package* module and use a **relative import**, then launch it as a module:

```python
# backtesting_module/tests/data_test.py
from ..data_handler import DataHandler   # ← one level up (backtesting_module)

def main() -> None:
    dh = DataHandler(
        alpaca_api_key="YOUR_ALPACA_KEY",
        alpaca_secret="YOUR_ALPACA_SECRET",
        polygon_key="YOUR_POLYGON_KEY",
    )

    # 1 ▸ get stock bars
    bars = dh.get_stock_bars("AAPL", "2023-01-01", "2023-06-30", "5Min")
    print("Found", len(bars), "bars")

    # 2 ▸ discover calls
    calls = dh.list_option_tickers(
        "AAPL",
        exp_from="2024-03-15",
        exp_to="2024-03-22",
        strike_min=160,
        strike_max=190,
        opt_type="call",
        as_of="2024-01-01",
        limit=100,
    )
    print("Search returned:", calls)

    # 3 ▸ pull minute bars
    opt_bars = dh.get_option_aggregates(calls[0], "2024-01-15", "2024-02-28")
    print("Aggregates for first option:\n", opt_bars.head())

if __name__ == "__main__":
    main()
```

Run it from the project root (the directory that contains `backtesting_module/`) so Python sets the package context correctly:

```bash
python -m backtesting_module.tests.data_test
```

The `-m` flag tells Python to execute the file *as a module*, which makes relative imports like `from ..data_handler` resolve cleanly without tweaking `sys.path` or `PYTHONPATH`.

## Notes
- See each module for more details and docstrings.
- Designed for extensibility: add new strategies, data sources, or analytics as needed. 

## Performance: Parallelized Option Ranking

When you run the main backtesting loop, the system helps you select the most liquid or active option contracts for your chosen stock and date range. To do this efficiently, the code uses Python's `ThreadPoolExecutor` to query Polygon's API for all available option contracts **in parallel**.

- **Why?**
  - Polygon's API only returns aggregate (OHLCV) data for contracts that actually traded. Many contracts are illiquid or inactive.
  - Ranking all contracts by the amount of available data (number of bars) is slow if done sequentially, especially when there are hundreds or thousands of contracts.

- **How?**
  - The code launches multiple parallel API requests (default: 8 at a time) to fetch aggregate data for each contract.
  - It then sorts all contracts by the number of bars returned and presents the top 20 to the user for selection.

- **User Benefit:**
  - This makes the workflow much faster and more responsive, even for stocks with large option chains.
  - You are less likely to pick an illiquid or inactive contract by accident.

**Note:** The number of parallel workers is set to 8 by default, balancing speed and API rate limits. You can adjust this in the code if needed. 

# IV Forecasting System — Quickstart

## 1. Install Requirements (Jupyter cell)

Paste this at the top of your notebook:

```python
!pip install -r ../requirements.txt
```

Or, if running outside Jupyter:

```bash
pip install -r requirements.txt
```

## 2. Usage

- The main forecasting code is in `delta_gamma_hedging.py`.
- You must provide a working `DataHandler` class for real data access (see stub in the script).
- Example usage and demo functions are provided in the script (see `quick_start_demo()`).

## 3. DataHandler

- The provided `DataHandler` is a stub. You must implement:
  - `get_stock_bars()`
  - `options_search()`
  - `get_option_aggregates()`
- These should return pandas DataFrames as expected by the pipeline.

## 4. Plotting

- The system provides ATM term structure, 30-day smile, and volatility surface plots.
- See the demo functions for example plotting code.

## 5. API Keys

- Do **not** hardcode API keys in scripts. Use environment variables or a config file.

## 6. Support

- For questions, see the comments in `delta_gamma_hedging.py` or ask your AI assistant. 