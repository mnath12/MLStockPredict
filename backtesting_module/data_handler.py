from __future__ import annotations

import datetime as dt
import re
from typing import List, Optional

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