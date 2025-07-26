#!/usr/bin/env python3
"""
Main backtesting loop following the architecture diagram.

This implements the main loop that connects all components:
DataHandler -> GreeksEngine -> Portfolio -> Strategy -> PositionSizer -> ExecutionHandler

Key Features:
- Modular, extensible architecture for stock and option backtesting.
- User-driven workflow for selecting stock, date range, and frequency.
- Fetches and displays available option contracts for a given date.
- **Parallelized option contract ranking:**
    - When ranking options by available data (number of bars), the code now uses Python's ThreadPoolExecutor to query Polygon's API for all contracts in parallel.
    - This dramatically speeds up the process of finding the top 20 most active/liquid contracts, making the workflow much more responsive for the user.
    - The number of parallel workers is set to 8 by default, balancing speed and API rate limits.
- User selects from the top 20 contracts with the most data, reducing the chance of picking illiquid or inactive options.

(Backtest execution is stubbed out as requested.)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
import re
import concurrent.futures

from backtesting_module import (
    DataHandler, GreeksEngine, Portfolio, PositionSizer, ExecutionHandler,
    BaseStrategy, LSTMStrategy, Fill
)
from backtesting_module.strategy import BuyAndHoldStrategy
from backtesting_module.data_handler import DataHandler

def parse_freq(freq: str):
    # Remove spaces and lowercase
    freq = freq.replace(' ', '').lower()
    m = re.fullmatch(r"(\d+)([a-z]+)", freq)
    if not m:
        raise ValueError(f"Invalid frequency: {freq}")
    n, unit = int(m.group(1)), m.group(2)
    if unit in ("min", "t"):
        return n, "minute"
    if unit in ("hour", "h"):
        return n, "hour"
    if unit in ("day", "d"):
        return n, "day"
    if unit in ("week", "w"):
        return n, "week"
    if unit in ("month", "m"):
        return n, "month"
    if unit in ("quarter", "q"):
        return n, "quarter"
    if unit in ("year", "y"):
        return n, "year"
    raise ValueError(f"Unsupported frequency unit: {unit}")

def parse_option_ticker(ticker):
    # Remove O: prefix if present
    if ticker.startswith("O:"):
        ticker = ticker[2:]
    # Underlying: all letters up to first digit
    m = re.match(r"([A-Z]+)(\d{6})([CP])(\d+)", ticker)
    if not m:
        raise ValueError(f"Could not parse option ticker: {ticker}")
    underlying, date_part, opt_type, strike_part = m.groups()
    expiry = pd.to_datetime(f"20{date_part}", format='%Y%m%d')
    strike = float(strike_part) / 1000.0
    return underlying, expiry, opt_type.lower(), strike

class BacktestEngine:
    """
    Main backtesting engine that orchestrates all components.
    """
    def __init__(self, initial_cash: float = 100000):
        self.data_handler: Optional[DataHandler] = None
        self.greeks_engine = GreeksEngine(pricing_model="Black-Scholes")
        self.portfolio = Portfolio(initial_cash)
        self.position_sizer = PositionSizer()
        self.execution_handler = ExecutionHandler()
        self.strategy: Optional[BaseStrategy] = None
        self.big_df: Optional[pd.DataFrame] = None
        self.results = {}

    def setup_data_handler(self, alpaca_key: str = "", alpaca_secret: str = "", polygon_key: str = "") -> None:
        # Always use the real DataHandler
        print("Setting up DataHandler...")
        self.data_handler = DataHandler(alpaca_key, alpaca_secret, polygon_key)

    def prompt_user(self):
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        return symbol, start_date, end_date

    def run_backtest(self) -> Dict[str, Any]:
        if not all([self.data_handler, self.strategy, self.big_df is not None]):
            raise ValueError("Must setup data handler, strategy, and load data before running backtest")

        print("\n" + "="*50)
        print("STARTING MAIN BACKTESTING LOOP")
        print("="*50)

        total_bars = len(self.big_df)
        print(f"Processing {total_bars} bars...")

        for i, (timestamp, row) in enumerate(self.big_df.iterrows()):
            market_data = self._extract_market_data(row)
            greeks_data = self._calculate_greeks(market_data, timestamp)
            market_data.update(greeks_data)
            current_prices = {k: v for k, v in market_data.items() if not k.endswith(('_delta', '_gamma', '_vega', '_theta', '_rho'))}
            self.portfolio.update_portfolio_value(current_prices, timestamp)
            portfolio_view = self.portfolio.portfolio_view()
            targets = self.strategy.on_bar(self.big_df, i, portfolio_view, market_data)
            if targets:
                orders = self.position_sizer.get_orders(targets, self.portfolio, timestamp)
                if orders:
                    fills = self.execution_handler.get_fills(orders, current_prices)
                    for fill in fills:
                        self.portfolio.update_with_fill(fill)
        print("✓ Backtesting completed")
        return {}

    def _extract_market_data(self, row: pd.Series) -> Dict[str, float]:
        market_data = {}
        for col in row.index:
            if isinstance(row[col], (float, int)) and not pd.isna(row[col]):
                if isinstance(col, str) and col.endswith('_close'):
                    symbol = col.replace('_close', '')
                    market_data[symbol] = float(row[col])
        return market_data

    def _calculate_greeks(self, market_data: Dict[str, float], timestamp: Any) -> Dict[str, float]:
        # Dummy implementation for now
        return {}

def main():
    print("Initializing Backtesting Engine...")
    # Prompt for API keys or use empty strings for demo
    alpaca_key = input("Enter Alpaca API key (or leave blank): ").strip()
    alpaca_secret = input("Enter Alpaca API secret (or leave blank): ").strip()
    polygon_key = input("Enter Polygon API key (or leave blank): ").strip()

    data_handler = DataHandler(
        alpaca_api_key="PKCLL4TXCDLRN76OGRAB",
        alpaca_secret="ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T",
        polygon_key="ejp0y0ppSQJzIX1W8qSoTIvL5ja3ctO9",
    )
    # Prompt user for stock, date, and frequency
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    today_str = datetime.today().strftime("%Y-%m-%d")
    start_date = input(f"Enter start date (YYYY-MM-DD) [default: {today_str}]: ").strip()
    if not start_date:
        start_date = today_str
    end_date = input(f"Enter end date (YYYY-MM-DD) [default: {today_str}]: ").strip()
    if not end_date:
        end_date = today_str
    freq = input("Enter frequency (e.g., 1D, 5Min) [default: 1D]: ").strip()
    if not freq:
        freq = "1D"

    # Fetch and print stock time series
    stock_df = data_handler.get_stock_bars(symbol, start_date, end_date, freq)
    print(f"\nStock time series for {symbol} from {start_date} to {end_date} (freq={freq}):")
    print(stock_df)

    # Display current price of the stock
    if not stock_df.empty:
        last_price = stock_df['close'].iloc[-1]
        print(f"\033[1;36mCurrent price of {symbol}: {last_price:.2f}\033[0m")
    else:
        last_price = None
        print(f"\033[1;31mWarning: No stock data found for {symbol}.\033[0m")

    # Optionally filter for strike price and expiry date
    print("\nYou can filter options by strike price and expiry date (press Enter to skip any filter).")
    strike_min = input("Enter minimum strike price (or leave blank): ").strip()
    strike_max = input("Enter maximum strike price (or leave blank): ").strip()
    exp_from = input("Enter earliest expiry date (YYYY-MM-DD, or leave blank): ").strip()
    exp_to = input("Enter latest expiry date (YYYY-MM-DD, or leave blank): ").strip()
    strike_min = float(strike_min) if strike_min else None
    strike_max = float(strike_max) if strike_max else None
    exp_from = exp_from if exp_from else None
    exp_to = exp_to if exp_to else None

    # Fetch available options for that day with filters
    options = data_handler.options_search(
        symbol,
        exp_from=exp_from,
        exp_to=exp_to,
        strike_min=strike_min,
        strike_max=strike_max,
        as_of=start_date
    )
    if not options:
        print("No options found for this stock and date.")
        return

    # Print a colored breakdown of the option ticker format
    print("\n\033[1;33mOption Ticker Format:\033[0m")
    print("  \033[1;34mO:\033[0m\033[1;32mAAPL\033[0m\033[1;35m240322\033[0m\033[1;36mC\033[0m\033[1;31m00185000\033[0m")
    print("  |    |      |     |         ")
    print("  |    |      |     |-- Strike price (e.g., 00185000 = $185.00)")
    print("  |    |      |-- Option type (C = Call, P = Put)")
    print("  |    |-- Expiry date (YYMMDD, e.g., 240322 = 2024-03-22)")
    print("  |-- Underlying symbol (e.g., AAPL)")
    print("  O: = Option contract prefix required by Polygon API\n")

    print(f"\033[1;36mCurrent price of {symbol}: {last_price:.2f}\033[0m" if last_price is not None else "")

    print("\nRanking options by available data...")
    multiplier, timespan = parse_freq(freq)

    def get_option_bar_count(opt):
        try:
            ticker = opt if opt.startswith("O:") else f"O:{opt}"
            data = data_handler._poly.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=50000,
            )
            count = len(data) if isinstance(data, list) else 0
        except Exception:
            count = 0
        return (opt, count)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        option_data_counts = list(executor.map(get_option_bar_count, options))

    top_options = sorted(option_data_counts, key=lambda x: x[1], reverse=True)[:20]
    if not top_options or top_options[0][1] == 0:
        print("No options with aggregate data found for this stock and date.")
        return

    print("\nTop 20 options by available data:")
    for idx, (opt, count) in enumerate(top_options):
        print(f"{idx}: {opt} (bars: {count})")

    print(f"\033[1;36mCurrent price of {symbol}: {last_price:.2f}\033[0m" if last_price is not None else "")

    opt_idx = int(input("Select an option by number: "))
    option_ticker = top_options[opt_idx][0]

    # Fetch and print option time series
    option_df = data_handler.get_option_aggregates(option_ticker, start_date, end_date, timespan, multiplier)
    print(f"\nOption time series for {option_ticker} from {start_date} to {end_date} (freq={freq}):")
    print(option_df)

    # Align stock and option data on timestamps
    combined_df = stock_df[['close']].rename(columns={'close': f'{symbol}_close'})
    combined_df = combined_df.join(option_df[['close']].rename(columns={'close': f'{option_ticker}_close'}), how='inner')
    combined_df = combined_df.dropna()
    if combined_df.empty:
        print("No overlapping data between stock and option. Exiting.")
        return

    # Initialize GreeksEngine
    greeks_engine = GreeksEngine()

    # Initialize Portfolio
    portfolio = Portfolio(initial_cash=100000)

    # Initial positions: long 1 call, short enough stock to delta-hedge
    first_row = combined_df.iloc[0]
    S0 = first_row[f'{symbol}_close']
    # Parse strike and expiry from option_ticker
    try:
        underlying, expiry, opt_type, K = parse_option_ticker(option_ticker)
    except Exception as e:
        print(f"Could not parse option ticker for strike/expiry: {e}. Exiting.")
        return
    T0 = max((expiry - combined_df.index[0]).days / 365.0, 1/365)
    sigma = 0.2  # Placeholder, could estimate from data
    r = 0.02
    # Compute initial delta
    greeks = greeks_engine.compute(
        symbol=option_ticker,
        underlying_price=float(S0),
        strike=K,
        time_to_expiry=T0,
        volatility=sigma,
        risk_free_rate=r,
        option_type=opt_type
    )
    option_delta = greeks['delta']
    # Long 1 call
    option_qty = 1
    # Short enough stock to delta-hedge
    stock_qty = -option_delta * 100 * option_qty
    # Round to nearest integer
    stock_qty = int(round(stock_qty))
    # Initial fills
    t0 = combined_df.index[0]
    portfolio.update_with_fill(Fill(symbol=option_ticker, qty=option_qty, price=float(first_row[f'{option_ticker}_close']), timestamp=t0))
    portfolio.update_with_fill(Fill(symbol=symbol, qty=stock_qty, price=float(first_row[f'{symbol}_close']), timestamp=t0))
    print(f"\nInitial position: Long 1 {option_ticker}, Short {abs(stock_qty)} {symbol} (delta-hedged)")

    # Main backtest loop
    for i, (ts, row) in enumerate(combined_df.iterrows()):
        S = float(row[f'{symbol}_close'])
        opt_price = float(row[f'{option_ticker}_close'])
        ts = pd.to_datetime(ts)
        T = max((expiry - ts).days / 365.0, 1/365)
        greeks = greeks_engine.compute(
            symbol=option_ticker,
            underlying_price=S,
            strike=K,
            time_to_expiry=T,
            volatility=sigma,
            risk_free_rate=r,
            option_type=opt_type
        )
        option_delta = greeks['delta']
        # Portfolio delta: option_qty * option_delta * 100 + stock_qty * 1.0
        portfolio_view = portfolio.portfolio_view()
        current_option_qty = portfolio_view['options'].get(option_ticker, None)
        current_stock_qty = portfolio_view['stocks'].get(symbol, None)
        if current_option_qty is not None:
            option_qty = current_option_qty.qty
        if current_stock_qty is not None:
            stock_qty = current_stock_qty.qty
        portfolio_delta = option_qty * option_delta * 100 + stock_qty
        # Target: keep portfolio delta ≈ 0 by adjusting stock position
        target_stock_qty = -option_qty * option_delta * 100
        # Round to nearest integer
        target_stock_qty = int(round(target_stock_qty))
        # If adjustment needed, trade stock
        if stock_qty != target_stock_qty:
            trade_qty = target_stock_qty - stock_qty
            portfolio.update_with_fill(Fill(symbol=symbol, qty=trade_qty, price=S, timestamp=ts))
        # Update portfolio value
        current_prices = {symbol: S, option_ticker: opt_price}
        portfolio.update_portfolio_value(current_prices, ts)
    print("\nBacktest complete.")
    metrics = portfolio.get_performance_metrics(risk_free_rate=r)
    print(f"Total return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()