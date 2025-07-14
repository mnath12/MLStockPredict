#!/usr/bin/env python
"""
Integrated HAR-RV + GARCH Volatility Forecaster    v2.1  (June-2025)
Combines daily HAR features with a GARCH(1,1) model for improved accuracy, using Alpaca or yfinance for data.
Usage:
    python har_garch_forecaster.py --ticker AAPL --start 2023-01-01 --end 2024-06-01
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

# Alpaca imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# ──────── CLI & Logging ────────
def parse_args():
    parser = argparse.ArgumentParser(description="HAR+GARCH Volatility Forecaster")
    parser.add_argument('--ticker', '-t', default='AAPL')
    parser.add_argument('--start', '-s', default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    parser.add_argument('--end',   '-e', default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    parser.add_argument('--verbose','-v', action='store_true')
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# ──────── Data Download ────────
def download_prices(ticker: str, start: str, end: str, tries: int = 3) -> pd.DataFrame:
    """Download daily OHLC data using Alpaca API or fallback to yfinance."""
    # Try Alpaca first
    if ALPACA_AVAILABLE:
        key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
        if key and secret:
            client = StockHistoricalDataClient(key, secret)
            for attempt in range(tries):
                try:
                    logging.debug(f"Alpaca attempt {attempt+1}/{tries}")
                    req = StockBarsRequest(
                        symbol_or_symbols=[ticker],
                        timeframe=TimeFrame.Day,
                        start=pd.Timestamp(start),
                        end=pd.Timestamp(end),
                        adjustment=Adjustment.ALL
                    )
                    bars = client.get_stock_bars(req).df
                    if bars.empty:
                        raise ValueError("No data returned from Alpaca")
                    df = bars.droplevel(0)[["open","high","low","close"]]
                    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
                    logging.info(f"Alpaca: Downloaded {len(df)} days for {ticker}")
                    return df
                except Exception as e:
                    logging.warning(f"Alpaca attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
            logging.warning("Alpaca failed, falling back to yfinance")
        else:
            logging.info("No Alpaca credentials, using yfinance")
    else:
        logging.info("Alpaca SDK not available, using yfinance")
    # Fallback to yfinance
    for attempt in range(tries):
        try:
            logging.debug(f"yfinance attempt {attempt+1}/{tries}")
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            logging.info(f"yfinance: Downloaded {len(df)} days for {ticker}")
            return df[["Open","High","Low","Close"]]
        except Exception as e:
            logging.warning(f"yfinance attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download data for {ticker}")

# ──────── Feature Engineering ────────
def realized_volatility(prices: pd.Series) -> pd.Series:
    lr = np.log(prices / prices.shift(1)).dropna()
    rv = lr.abs() * np.sqrt(np.pi/2)
    rv.name = 'rv'
    return rv


def har_features(rv: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=rv.index)
    df['rv']   = rv
    df['rv_w'] = rv.rolling(5).mean()
    df['rv_m'] = rv.rolling(22).mean()
    df['rq']   = (rv**4).rolling(22).mean()
    return df.dropna()

# ──────── Model: HAR + GARCH ────────
class HARGARCHForecaster:
    def __init__(self, weight: float = 0.5):
        self.weight = weight
        self.har_model = LinearRegression()
        self.garch_res = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # HAR regression
        X_har = X[['rv','rv_w','rv_m','rq']].shift(1).dropna()
        y_har = y.reindex(X_har.index)
        self.har_model.fit(X_har, y_har)
        # GARCH on RV*100
        am = arch_model(y*100, vol='GARCH', p=1, q=1, dist='normal')
        self.garch_res = am.fit(disp='off')

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_har = X[['rv','rv_w','rv_m','rq']].shift(1).dropna()
        har_pred = self.har_model.predict(X_har)
        garch_fore = self.garch_res.forecast(horizon=1, reindex=False)
        var_pred = garch_fore.variance.values[-1, 0] / 10000
        garch_vol = np.sqrt(var_pred)
        combined = self.weight * har_pred + (1 - self.weight) * garch_vol
        combined = pd.Series(combined, index=X_har.index, name='vol_forecast')
        return combined

# ──────── Main ────────
def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.info(f"Fetching {args.ticker} from {args.start} to {args.end}")

    df = download_prices(args.ticker, args.start, args.end)
    rv = realized_volatility(df['Close'])
    feats = har_features(rv)

    y = feats['rv']
    X = feats

    # CV for weight
    ws = np.linspace(0.0, 1.0, 11)
    tscv = TimeSeriesSplit(n_splits=5)
    best_w, best_err = 0.5, np.inf
    for w in ws:
        errs = []
        for train_i, test_i in tscv.split(X):
            fore = HARGARCHForecaster(weight=w)
            fore.fit(X.iloc[train_i], y.iloc[train_i])
            pred = fore.predict(X.iloc[test_i])
            errs.append(np.mean((y.iloc[test_i] - pred)**2))
        mse = np.mean(errs)
        logging.debug(f"w={w:.2f} MSE={mse:.4f}")
        if mse < best_err:
            best_err, best_w = mse, w

    logging.info(f"Selected weight w={best_w:.2f} with CV-MSE={best_err:.4f}")

    # Final fit & forecast
    fore = HARGARCHForecaster(weight=best_w)
    fore.fit(X, y)
    forecast = fore.predict(X)

    out = pd.DataFrame({'actual': y, 'forecast': forecast}).dropna()
    print(out.tail(10))

if __name__ == '__main__':
    main()
