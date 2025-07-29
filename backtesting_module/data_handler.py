from __future__ import annotations

import datetime as dt
import re
from typing import List, Optional, Union
import numpy as np

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests   import StockBarsRequest
from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit
from polygon import RESTClient

# ──────────────────────  utility: convert "5Min" → TimeFrame  ────────────────
_TIMEFRAME_RE = re.compile(r"(\d+)([A-Za-z]+)")

def _parse_timeframe(freq: str) -> TimeFrame:
    m = _TIMEFRAME_RE.fullmatch(freq)
    if not m:
        raise ValueError(
            f"Bad timeframe '{freq}'. Use formats like '5Min', '12H', '1D'."
        )
    n, unit = int(m.group(1)), m.group(2).lower()

    if unit in ("min", "t"):
        return TimeFrame(n, TimeFrameUnit.Minute) # type: ignore[attr-defined]
    if unit in ("hour", "h"):
        return TimeFrame(n, TimeFrameUnit.Hour) # type: ignore[attr-defined]
    if unit in ("day", "d"):
        return TimeFrame(n, TimeFrameUnit.Day) # type: ignore[attr-defined]
    if unit in ("week", "w"):
        return TimeFrame(n, TimeFrameUnit.Week) # type: ignore[attr-defined]
    if unit in ("month", "m"):
        return TimeFrame(n, TimeFrameUnit.Month) # type: ignore[attr-defined]

    raise ValueError(f"Unsupported timeframe unit '{unit}' in '{freq}'")

class DataHandler:
    """
    Unified wrapper around:

    • Alpaca historical stock bars (OHLCV)
    • Polygon reference endpoints (contract discovery)
    • Polygon aggregates for options (minute/hour/day bars)

    Only the three methods required for your Δ-hedge, ΔΓ-hedge and jump-premium
    back-tests are implemented.  Anything extra (quotes, snapshots, Greeks)
    can be added as tiny helpers later.
    """

    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_secret: str,
        polygon_key: str,
        tz: dt.tzinfo = dt.timezone.utc,
    ):
        self._alpaca = StockHistoricalDataClient(alpaca_api_key, alpaca_secret)
        self._poly   = RESTClient(polygon_key)
        self._tz     = tz

    def get_stock_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """
        Pull OHLCV bars for one equity symbol via Alpaca and return a tidy
        `pandas.DataFrame` indexed by (tz-aware) timestamp.
        """
        tf = _parse_timeframe(timeframe)

        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=dt.datetime.fromisoformat(start_date).replace(tzinfo=self._tz),
            end=dt.datetime.fromisoformat(end_date).replace(tzinfo=self._tz),
        )

        bars = self._alpaca.get_stock_bars(req).df.sort_index() # type: ignore[attr-defined]

        if isinstance(bars.index, pd.MultiIndex):          # flatten 1-symbol query
            bars = bars.xs(ticker, level="symbol")

        return bars.tz_convert(self._tz)

    def options_search(
        self,
        underlying: str,
        exp_from: str | None = None,
        exp_to:   str | None = None,
        strike_min: float | None = None,
        strike_max: float | None = None,
        opt_type:   str | None = None,     # 'call' | 'put'
        as_of:      str | None = None,
        limit: int = 1000,
    ) -> list[str]:
        """
        Return contract tickers (no 'O:' prefix) using Polygon's
        `list_options_contracts` generator.  Works with historical chains
        via `as_of='YYYY-MM-DD'`.
        """
        contracts_iter = self._poly.list_options_contracts(
            underlying_ticker   = underlying,
            contract_type       = opt_type,
            expiration_date_gte = exp_from,
            expiration_date_lte = exp_to,
            as_of               = as_of,
            limit               = limit,
        )

        tickers = []
        for c in contracts_iter:
            if isinstance(c, bytes):
                # If bytes, decode to string and append
                tickers.append(c.decode())
            else:
                k = c.strike_price  # float
                if strike_min is not None and k is not None and k < strike_min:
                    continue
                if strike_max is not None and k is not None and k > strike_max:
                    continue
                tickers.append(c.ticker)  # e.g. 'AAPL240322C00185000'
        return tickers

    def get_option_aggregates(
        self,
        option_ticker: str,              # WITHOUT 'O:'
        start_date: str,
        end_date: str,
        timespan: str = "minute",        # 'minute', 'hour', 'day'
        multiplier: int = 1,
        adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Historical OHLCV bars for a single option contract.
        Polygon's newer SDK returns a *list*; older SDK returns an Aggs obj.
        This helper accommodates both.
        """
        resp = self._poly.get_aggs(
            ticker     = f"{option_ticker}",   # NOTE the 'O:' prefix
            multiplier = multiplier,
            timespan   = timespan,
            from_      = start_date,
            to         = end_date,
            adjusted   = adjust,
            sort       = "asc",
            limit      = 50_000,
        )

        # ── normalise to a list of Agg objects ────────────────────────────
        data = resp  # No need for hasattr(resp, "results")
        if not data:
            raise ValueError(f"No aggregate data for {option_ticker}")

        # ── build tidy DataFrame ───────────────────────────────────────────
        df = pd.DataFrame([a.__dict__ for a in data]).set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert(self._tz)

        return (
            df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low":  "low",
                    "close": "close",
                    "volume": "volume",
                    "vwap":   "vwap",
                    "transactions": "trades",
                }
            )
            .sort_index()
            .astype(
                {
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": int,
                    "vwap": float,
                    "trades": int,
                }
            )
        ) 

    def get_option_price_series(
        self,
        option_ticker: str,
        start_date: str,
        end_date: str,
        timespan: str = "day",
        multiplier: int = 1,
        adjust: bool = True,
        price_type: str = "mid"
    ) -> pd.Series:
        """
        Get option price series using mid price (average of bid/ask) or other price type.
        
        Args:
            option_ticker: Option ticker without 'O:' prefix
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timespan: 'minute', 'hour', 'day'
            multiplier: Time multiplier
            adjust: Whether to adjust for splits
            price_type: 'mid', 'close', 'open', 'high', 'low', 'vwap'
            
        Returns:
            pd.Series: Option price series indexed by timestamp
        """
        # Get full OHLCV data
        bars_df = self.get_option_aggregates(
            option_ticker=option_ticker,
            start_date=start_date,
            end_date=end_date,
            timespan=timespan,
            multiplier=multiplier,
            adjust=adjust
        )
        
        # Extract the requested price type
        if price_type == "mid":
            # Calculate mid price as average of high and low
            # Note: This is an approximation since we don't have bid/ask
            price_series = (bars_df['high'] + bars_df['low']) / 2
        elif price_type in bars_df.columns:
            price_series = bars_df[price_type]
        else:
            raise ValueError(f"Invalid price_type: {price_type}. Available: {list(bars_df.columns)}")
        
        return price_series.rename(f"{option_ticker}_{price_type}")

    def get_realized_volatility(
        self,
        bars_df: pd.DataFrame,
        symbol: str = None,
        price_col: str = "close",
        tz: str = "America/New_York"
    ) -> pd.Series:
        """
        Compute daily realized volatility from price data.
        
        Args:
            bars_df: DataFrame with OHLCV data
            symbol: Symbol to extract (if MultiIndex)
            price_col: Column name for prices
            tz: Timezone for output
            
        Returns:
            pd.Series: Daily realized volatility
        """
        # Extract price series
        if isinstance(bars_df.index, pd.MultiIndex):
            if symbol is None:
                raise ValueError("Symbol must be provided for MultiIndex data")
            px = (bars_df.xs(symbol, level="symbol")[price_col]
                             .tz_convert(tz)
                             .sort_index())
        else:
            px = bars_df[price_col].tz_convert(tz).sort_index()

        # Compute returns and realized volatility
        ret = px.pct_change().dropna()
        lret = np.log(px).diff().dropna()

        # Realized vol: √(∑ intraday r²) per calendar day
        rv_daily = (ret.pow(2)
                       .groupby(ret.index.date).sum()
                       .pipe(np.sqrt)
                       .rename("rv"))
        rv_daily.index = pd.to_datetime(rv_daily.index).tz_localize(tz)
        return rv_daily

    def get_hourly_realized_volatility(
        self,
        bars_df: pd.DataFrame,
        symbol: str = None,
        price_col: str = "close",
        tz: str = "America/New_York"
    ) -> pd.Series:
        """
        Compute hourly realized volatility during regular US trading hours.
        Labels each hour by its end time (e.g. returns in (09:00,10:00] → 10:00).
        
        Args:
            bars_df: DataFrame with OHLCV data
            symbol: Symbol to extract (if MultiIndex)
            price_col: Column name for prices
            tz: Timezone for output
            
        Returns:
            pd.Series: Hourly realized volatility
        """
        # Extract price series
        if isinstance(bars_df.index, pd.MultiIndex):
            if symbol is None:
                raise ValueError("Symbol must be provided for MultiIndex data")
            px = (bars_df.xs(symbol, level="symbol")[price_col]
                             .tz_convert(tz)
                             .sort_index())
        else:
            px = bars_df[price_col].tz_convert(tz).sort_index()

        # Compute returns
        ret = px.pct_change().dropna()

        # Square returns and bin into right-closed hourly buckets
        sq = ret.pow(2)
        # label='right' means (H-1,H] → H
        hourly_sum = sq.groupby(pd.Grouper(freq="H", label="right", closed="right")).sum()

        # Take sqrt to get realized vol
        rv_hourly = np.sqrt(hourly_sum).rename("rv_hourly")

        # Ensure the index is tz-aware local hours
        if rv_hourly.index.tz is None:
            rv_hourly.index = rv_hourly.index.tz_localize(tz)

        return rv_hourly

    def plot_price_returns_volatility(
        self,
        bars_df: pd.DataFrame,
        symbol: str = None,
        price_col: str = "close",
        tz: str = "America/New_York"
    ):
        """
        Plot price, returns, and realized volatility.
        
        Args:
            bars_df: DataFrame with OHLCV data
            symbol: Symbol to extract (if MultiIndex)
            price_col: Column name for prices
            tz: Timezone for output
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        # Extract price series
        if isinstance(bars_df.index, pd.MultiIndex):
            if symbol is None:
                raise ValueError("Symbol must be provided for MultiIndex data")
            px = (bars_df.xs(symbol, level="symbol")[price_col]
                             .tz_convert(tz)
                             .sort_index())
        else:
            px = bars_df[price_col].tz_convert(tz).sort_index()

        # Compute returns and volatility
        ret = px.pct_change().dropna()
        lret = np.log(px).diff().dropna()
        rv_daily = self.get_realized_volatility(bars_df, symbol, price_col, tz)

        # Helper function to plot series with stats
        def plot_series(series, title, ylabel):
            fig, ax = plt.subplots()
            series.plot(ax=ax)
            mu, sd = series.mean(), series.std()
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.95,
                    f"μ = {mu:+.4f}\nσ = {sd:.4f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
            plt.show()

        # Plot all series
        symbol_name = symbol if symbol else "Stock"
        plot_series(px, f"{symbol_name} price", "price")
        plot_series(ret, f"{symbol_name} simple return rₜ", "return")
        plot_series(lret, f"{symbol_name} log return ℓₜ", "log-return")
        plot_series(rv_daily,
                    f"{symbol_name} daily realized volatility σᴿ (√∑ r²)",
                    "σᴿ")

    def train_test_split_last_n_days(
        self,
        df: pd.DataFrame,
        n_days: int = 7
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train and test sets,
        where test set is the last n_days calendar days.
        
        Args:
            df: DataFrame with DateTimeIndex
            n_days: Number of days to reserve for test set
            
        Returns:
            tuple: (train_df, test_df)
        """
        # Ensure sorted by time
        df_sorted = df.sort_index()

        # Compute cutoff timestamp
        last_ts = df_sorted.index.max()
        cutoff = last_ts - pd.Timedelta(days=n_days)

        # Split
        train_df = df_sorted.loc[df_sorted.index <= cutoff]
        test_df = df_sorted.loc[df_sorted.index > cutoff]

        return train_df, test_df

    def scan_options_by_strike(
        self,
        underlying: str,
        target_strike: float,
        strike_tolerance: float = 5.0,
        exp_from: str | None = None,
        exp_to: str | None = None,
        opt_type: str | None = None,
        as_of: str | None = None,
        limit: int = 1000
    ) -> list[str]:
        """
        Scan for options around a specific strike price.
        
        Args:
            underlying: Underlying stock symbol
            target_strike: Target strike price
            strike_tolerance: Tolerance around target strike (in dollars)
            exp_from: Earliest expiration date
            exp_to: Latest expiration date
            opt_type: Option type ('call' or 'put')
            as_of: Historical date for option chain
            limit: Maximum number of contracts to return
            
        Returns:
            List of option tickers within strike tolerance
        """
        strike_min = target_strike - strike_tolerance
        strike_max = target_strike + strike_tolerance
        
        return self.options_search(
            underlying=underlying,
            exp_from=exp_from,
            exp_to=exp_to,
            strike_min=strike_min,
            strike_max=strike_max,
            opt_type=opt_type,
            as_of=as_of,
            limit=limit
        )
    
    def scan_options_by_expiry(
        self,
        underlying: str,
        target_expiry: str,
        expiry_tolerance_days: int = 7,
        strike_min: float | None = None,
        strike_max: float | None = None,
        opt_type: str | None = None,
        as_of: str | None = None,
        limit: int = 1000
    ) -> list[str]:
        """
        Scan for options around a specific expiration date.
        
        Args:
            underlying: Underlying stock symbol
            target_expiry: Target expiration date (YYYY-MM-DD)
            expiry_tolerance_days: Tolerance around target expiry (in days)
            strike_min: Minimum strike price
            strike_max: Maximum strike price
            opt_type: Option type ('call' or 'put')
            as_of: Historical date for option chain
            limit: Maximum number of contracts to return
            
        Returns:
            List of option tickers within expiry tolerance
        """
        target_dt = pd.to_datetime(target_expiry)
        exp_from = (target_dt - pd.Timedelta(days=expiry_tolerance_days)).strftime('%Y-%m-%d')
        exp_to = (target_dt + pd.Timedelta(days=expiry_tolerance_days)).strftime('%Y-%m-%d')
        
        return self.options_search(
            underlying=underlying,
            exp_from=exp_from,
            exp_to=exp_to,
            strike_min=strike_min,
            strike_max=strike_max,
            opt_type=opt_type,
            as_of=as_of,
            limit=limit
        )
    
    def get_iv_surface_data(
        self,
        underlying: str,
        start_date: str,
        end_date: str,
        moneyness_min: float = 0.8,
        moneyness_max: float = 1.2,
        max_options_per_expiry: int = 10,
        as_of: str | None = None
    ) -> dict:
        """
        Get option data organized for IV surface construction.
        
        Args:
            underlying: Underlying stock symbol
            start_date: Start date for price data
            end_date: End date for price data
            moneyness_min: Minimum moneyness (strike/spot)
            moneyness_max: Maximum moneyness (strike/spot)
            max_options_per_expiry: Maximum options per expiration
            as_of: Historical date for option chain
            
        Returns:
            Dictionary with organized option data
        """
        # Get current stock price for moneyness calculation
        try:
            stock_bars = self.get_stock_bars(
                ticker=underlying,
                start_date=start_date,
                end_date=start_date,
                timeframe="1D"
            )
            
            if stock_bars.empty:
                print(f"❌ No stock data returned for {underlying} on {start_date}")
                return {}
            
            print(f"📊 Stock bars columns: {list(stock_bars.columns)}")
            print(f"📊 Stock bars shape: {stock_bars.shape}")
            
            # Try different possible column names for close price
            close_col = None
            for col in ['close', 'Close', 'CLOSE', 'c', 'C']:
                if col in stock_bars.columns:
                    close_col = col
                    break
            
            if close_col is None:
                print(f"❌ No close price column found. Available columns: {list(stock_bars.columns)}")
                return {}
            
            current_price = stock_bars[close_col].iloc[-1]
            print(f"📊 Using column '{close_col}' for close price")
            
        except Exception as e:
            print(f"❌ Failed to get current stock price: {e}")
            print(f"   Stock bars info: {type(stock_bars)}")
            if hasattr(stock_bars, 'columns'):
                print(f"   Available columns: {list(stock_bars.columns)}")
            return {}
        
        print(f"📊 Current {underlying} price: ${current_price:.2f}")
        
        # Calculate strike range based on moneyness
        strike_min = current_price * moneyness_min
        strike_max = current_price * moneyness_max
        
        print(f"🎯 Strike range for moneyness {moneyness_min}-{moneyness_max}: ${strike_min:.2f} - ${strike_max:.2f}")
        
        # Get all available options
        all_options = self.options_search(
            underlying=underlying,
            strike_min=strike_min,
            strike_max=strike_max,
            as_of=as_of,
            limit=5000
        )
        
        print(f"📋 Found {len(all_options)} options in strike range")
        
        # Group options by expiration
        expiry_groups = {}
        for option_ticker in all_options:
            # Extract expiration from ticker (YYMMDD format)
            exp_str = option_ticker[-15:-8]
            year = "20" + exp_str[:2]
            month = exp_str[2:4]
            day = exp_str[4:6]
            expiry = f"{year}-{month}-{day}"
            
            if expiry not in expiry_groups:
                expiry_groups[expiry] = []
            expiry_groups[expiry].append(option_ticker)
        
        # Select options for each expiry (prioritize by liquidity/volume if available)
        selected_options = []
        for expiry, options in expiry_groups.items():
            # Sort by strike price for even distribution
            options.sort(key=lambda x: float(x[-7:]) / 1000.0)  # Extract strike price
            
            # Take evenly distributed strikes
            if len(options) <= max_options_per_expiry:
                selected_options.extend(options)
            else:
                # Select evenly spaced options
                step = len(options) // max_options_per_expiry
                selected_options.extend(options[::step][:max_options_per_expiry])
        
        print(f"🎯 Selected {len(selected_options)} options across {len(expiry_groups)} expiries")
        
        return {
            'underlying': underlying,
            'current_price': current_price,
            'start_date': start_date,
            'end_date': end_date,
            'selected_options': selected_options,
            'expiry_groups': expiry_groups
        }

    def train_test_split_by_percentage(
        self,
        df: pd.DataFrame,
        test_percentage: float = 0.2,
        validation_percentage: float = 0.0
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train, validation, and test sets by percentage.
        Maintains temporal order for time series data.
        
        Args:
            df: DataFrame with DateTimeIndex
            test_percentage: Percentage of data for test set (0.0 to 1.0)
            validation_percentage: Percentage of data for validation set (0.0 to 1.0)
            
        Returns:
            tuple: (train_df, validation_df, test_df)
            
        Example:
            train, val, test = data_handler.train_test_split_by_percentage(
                df, test_percentage=0.2, validation_percentage=0.1
            )
        """
        if not 0 <= test_percentage <= 1:
            raise ValueError("test_percentage must be between 0 and 1")
        if not 0 <= validation_percentage <= 1:
            raise ValueError("validation_percentage must be between 0 and 1")
        if test_percentage + validation_percentage >= 1:
            raise ValueError("test_percentage + validation_percentage must be less than 1")

        # Ensure sorted by time
        df_sorted = df.sort_index()
        total_rows = len(df_sorted)

        # Calculate split indices
        test_size = int(total_rows * test_percentage)
        val_size = int(total_rows * validation_percentage)
        
        # Split from the end (most recent data)
        test_end = total_rows
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - val_size
        train_end = val_start

        # Create splits
        train_df = df_sorted.iloc[:train_end]
        validation_df = df_sorted.iloc[val_start:val_end] if val_size > 0 else pd.DataFrame()
        test_df = df_sorted.iloc[test_start:test_end]

        return train_df, validation_df, test_df
