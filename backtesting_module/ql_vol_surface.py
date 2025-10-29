#!/usr/bin/env python
"""
ql_vol_surface.py  ─ build IV surface with QuantLib

Requires:
    pip install quantlib-python pandas numpy matplotlib scipy tqdm
    pip install alpaca-py polygon-api-client       # for DataHandler
"""

from __future__ import annotations
import os, sys, argparse, datetime as dt, math, warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
from tqdm import tqdm
import QuantLib as ql                                             # ← QuantLib

# Import GreeksEngine for enhanced IV surface building
from .greeks_engine import GreeksEngine

# ───────────────────────────────────────────────────────────────
# 1) tiny helpers to parse Polygon equity-option tickers
#    e.g.  AAPL250418P00215000  → 215.00 put exp 2025-04-18
# ───────────────────────────────────────────────────────────────
def decode_ticker(tkr: str) -> Dict:
    # Remove O: prefix if present
    if tkr.startswith("O:"):
        tkr = tkr[2:]
    
    root   = tkr[:-15]
    yymmdd = tkr[-15:-9]
    cp     = tkr[-9]
    strike = int(tkr[-8:]) / 1000.0
    expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date()
    return {"root": root, "type": "call" if cp == "C" else "put",
            "strike": strike, "expiry": expiry}

# ───────────────────────────────────────────────────────────────
# 2) Black-Scholes price – fallback for IV search & sanity
# ───────────────────────────────────────────────────────────────
def _bs_price(S, K, tau, r, sigma, cp):
    if tau <= 0 or sigma <= 0:                # expiry or sigma degeneracy
        return max(cp * (S - K), 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*tau)/(sigma*math.sqrt(tau))
    d2 = d1 - sigma*math.sqrt(tau)
    if cp == 1:
        return S*norm.cdf(d1) - K*math.exp(-r*tau)*norm.cdf(d2)
    else:
        return K*math.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)

# ───────────────────────────────────────────────────────────────
# 3) Implied-vol inverter – model-aware
# ───────────────────────────────────────────────────────────────
def implied_vol_qllib(opt: ql.VanillaOption,
                      process: ql.GeneralizedBlackScholesProcess,
                      price: float,
                      low: float = 1e-4,
                      high: float = 5.0,
                      tol: float = 1e-5,
                      maxeval: int = 100) -> float:
    """
    Generic QuantLib implied-vol helper that works for *any* engine
    whose price depends on the process’s flat vol Quote.
    """
    vol_quote: ql.SimpleQuote = process.blackVolatility().link.volatilityQuote()
    def f(sigma: float) -> float:
        vol_quote.setValue(sigma)
        return opt.NPV() - price
    try:
        return brentq(f, low, high, xtol=tol, maxiter=maxeval)
    except ValueError:
        return np.nan

# ───────────────────────────────────────────────────────────────
# 4) Build QuantLib ingredients for each engine family
# ───────────────────────────────────────────────────────────────
def make_process_engine(model: str,
                        S0: float,
                        r: float,
                        q: float,
                        sigma0: float,
                        jump_lambda: float = 1.0,
                        jump_mu: float = -0.1,
                        jump_sigma: float = 0.2,
                        tree_steps: int = 501):
    today = ql.Date.todaysDate()
    day_counter = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    rf_ts   = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_counter))
    div_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_counter))
    vol_q   = ql.SimpleQuote(sigma0)
    vol_ts  = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_q), day_counter))
    spot_q  = ql.QuoteHandle(ql.SimpleQuote(S0))

    if model == "bs":
        proc = ql.BlackScholesMertonProcess(spot_q, div_ts, rf_ts, vol_ts)
        engine = ql.AnalyticEuropeanEngine(proc)                 # European
    elif model == "binom":
        proc = ql.BlackScholesMertonProcess(spot_q, div_ts, rf_ts, vol_ts)
        engine = ql.BinomialVanillaEngine(proc, "crr", tree_steps)   # American-ready
    elif model == "merton":
        # jump parameters as SimpleQuotes so sigma search only touches vol_q
        lam = ql.QuoteHandle(ql.SimpleQuote(jump_lambda))
        mu  = ql.QuoteHandle(ql.SimpleQuote(jump_mu))
        jv  = ql.QuoteHandle(ql.SimpleQuote(jump_sigma))
        proc = ql.Merton76Process(spot_q, div_ts, rf_ts, vol_ts, lam, mu, jv)  # :contentReference[oaicite:0]{index=0}
        engine = ql.JumpDiffusionEngine(proc)                                   # :contentReference[oaicite:1]{index=1}
    else:
        raise ValueError("model must be 'bs', 'binom', or 'merton'")

    return proc, engine, vol_q

# ───────────────────────────────────────────────────────────────
# 5) Surface builder
# ───────────────────────────────────────────────────────────────
def build_surface(chain: list[str],
                  spot: float,
                  snapshot: dt.datetime,
                  model: str,
                  r: float = 0.04,
                  q: float = 0.00) -> pd.DataFrame:

    proc, engine, vol_q = make_process_engine(model, spot, r, q, sigma0=0.30)

    rows = []
    expired_count = 0
    api_fail_count = 0
    iv_fail_count = 0
    success_count = 0
    
    for tkr in tqdm(chain, desc="IV invert", unit="opt"):
        meta = decode_ticker(tkr)
        K     = meta["strike"]
        cp    = +1 if meta["type"] == "call" else -1
        tau   = (dt.datetime.combine(meta["expiry"], dt.time()).replace(tzinfo=tz_utc) - snapshot).days/365.0
        if tau <= 0:   # already expired
            expired_count += 1
            continue

        # ---- option contract in QuantLib
        expiry_date = ql.Date(meta["expiry"].day,
                              meta["expiry"].month,
                              meta["expiry"].year)
        if model == "binom":
            exercise = ql.AmericanExercise(expiry_date, expiry_date)
        else:
            exercise = ql.EuropeanExercise(expiry_date)
        payoff   = ql.PlainVanillaPayoff(ql.Option.Call if cp==+1 else ql.Option.Put, K)
        opt      = ql.VanillaOption(payoff, exercise)
        opt.setPricingEngine(engine)

        # ---- market mid quote via Polygon
        try:
            q_last = client.get_last_quote(ticker=f"O:{tkr}")
            mkt_px = (q_last.ask_price + q_last.bid_price)/2 if q_last.ask_price and q_last.bid_price else q_last.last_price
        except Exception as e:
            api_fail_count += 1
            if api_fail_count <= 3:  # Only print first few errors
                print(f"API fail for {tkr}: {e}")
            continue

        # ---- implied σ search
        try:
            iv = implied_vol_qllib(opt, proc, mkt_px)
            if np.isnan(iv):
                iv_fail_count += 1
                if iv_fail_count <= 3:  # Only print first few errors
                    print(f"IV calculation failed for {tkr}: got NaN")
                continue
            rows.append({"tau": tau, "strike": K, "iv": iv})
            success_count += 1
        except Exception as e:
            iv_fail_count += 1
            if iv_fail_count <= 3:  # Only print first few errors
                print(f"IV calculation error for {tkr}: {e}")
            continue

    # Debug: check what's in rows
    print(f"\n=== DEBUG SUMMARY ===")
    print(f"Total options processed: {len(chain)}")
    print(f"Expired options: {expired_count}")
    print(f"API failures: {api_fail_count}")
    print(f"IV calculation failures: {iv_fail_count}")
    print(f"Successful IV calculations: {success_count}")
    print(f"Final rows length: {len(rows)}")
    
    if rows:
        print(f"Debug: first row keys = {list(rows[0].keys())}")
        print(f"Debug: first row = {rows[0]}")
    
    if not rows:
        print("No valid option data found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    surf = (pd.DataFrame(rows)
              .pivot_table(index="tau", columns="strike", values="iv")
              .sort_index().sort_index(axis=1))
    return surf

# ───────────────────────────────────────────────────────────────
# 6) Enhanced IV Surface Builder using GreeksEngine
# ───────────────────────────────────────────────────────────────
def build_iv_surface_from_timeseries(
    data_handler,
    surface_data: dict,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    exercise_style: str = "american",
    tree: str = "crr"
) -> pd.DataFrame:
    """
    Build IV surface using the refactored GreeksEngine approach.
    
    Args:
        data_handler: DataHandler instance
        surface_data: Dictionary from data_handler.get_iv_surface_data()
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield
        exercise_style: 'american' or 'european'
        tree: Binomial tree type
        
    Returns:
        DataFrame with IV surface data
    """
    from .greeks_engine import GreeksEngine
    
    # Initialize GreeksEngine
    greeks_engine = GreeksEngine()
    
    # Extract data
    underlying = surface_data['underlying']
    selected_options = surface_data['selected_options']
    start_date = surface_data['start_date']
    end_date = surface_data['end_date']
    
    if not selected_options:
        print("❌ No options available for IV surface construction")
        return pd.DataFrame()
    
    print(f"🔬 Building IV surface for {underlying} with {len(selected_options)} options...")
    
    # Build IV surface data
    surface_data_points = []
    successful_options = 0
    
    for i, option_ticker in enumerate(selected_options):
        print(f"  [{i+1}/{len(selected_options)}] Processing {option_ticker}...")
        
        try:
            # Calculate IV series for this option
            iv_series = greeks_engine.calculate_implied_volatility_series(
                option_ticker=option_ticker,
                data_handler=data_handler,
                start_date=start_date,
                end_date=end_date,
                underlying_symbol=underlying,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                exercise_style=exercise_style,
                tree=tree
            )
            
            if len(iv_series) > 0:
                # Extract option details
                strike_price, expiration_date, option_type = greeks_engine._decode_option_ticker(option_ticker)
                
                # Add each IV point to surface data
                for timestamp, iv in iv_series.items():
                    # Calculate time to maturity in years
                    exp_dt = pd.to_datetime(expiration_date)
                    if timestamp.tz is not None:
                        t_naive = timestamp.tz_localize(None)
                    else:
                        t_naive = timestamp
                    ttm_years = (exp_dt - t_naive).days / 365.25
                    
                    surface_data_points.append({
                        'timestamp': timestamp,
                        'ttm': ttm_years,
                        'strike': strike_price,
                        'iv': iv,
                        'option_type': option_type
                    })
                
                successful_options += 1
                print(f"    ✅ Added {len(iv_series)} surface points")
            else:
                print(f"    ❌ No valid IV data for {option_ticker}")
                
        except Exception as e:
            print(f"    ❌ Error processing {option_ticker}: {e}")
            continue
    
    print(f"📊 IV Calculation Summary:")
    print(f"  Successful options: {successful_options}/{len(selected_options)}")
    print(f"  Total surface data points: {len(surface_data_points)}")
    
    if not surface_data_points:
        print("❌ No valid IV data found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Create DataFrame and sort by timestamp, then TTM, then strike
    surface_df = pd.DataFrame(surface_data_points)
    surface_df = surface_df.sort_values(['timestamp', 'ttm', 'strike'])
    
    print(f"✅ IV surface built with {len(surface_df)} data points")
    return surface_df



def build_iv_surface_interactive(
    stock_data: pd.DataFrame,
    option_data: Dict[str, pd.Series],
    expiration_dates: Dict[str, str],
    **kwargs
) -> pd.DataFrame:
    """
    Interactive IV surface builder with user prompts for parameters.
    """
    print("\n=== Interactive IV Surface Builder ===")
    
    # Get user preferences
    print("\nSelect exercise style:")
    print("1) American (default)")
    print("2) European")
    choice = input("Choice [1-2, default=1]: ").strip()
    exercise_style = "american" if choice != "2" else "european"
    
    print("\nSelect binomial tree type:")
    print("1) CRR (Cox-Ross-Rubinstein) - default")
    print("2) JR (Jarrow-Rudd)")
    print("3) EQP (Equal Probability)")
    print("4) Trigeorgis")
    tree_choice = input("Choice [1-4, default=1]: ").strip()
    tree_map = {"1": "crr", "2": "jr", "3": "eqp", "4": "trigeorgis"}
    tree = tree_map.get(tree_choice, "crr")
    
    # Get risk-free rate
    rfr_input = input(f"Risk-free rate (default=0.04): ").strip()
    risk_free_rate = float(rfr_input) if rfr_input else 0.04
    
    # Get dividend yield
    div_input = input(f"Dividend yield (default=0.0): ").strip()
    dividend_yield = float(div_input) if div_input else 0.0
    
    print(f"\nBuilding IV surface with:")
    print(f"  Exercise style: {exercise_style}")
    print(f"  Tree type: {tree}")
    print(f"  Risk-free rate: {risk_free_rate:.4f}")
    print(f"  Dividend yield: {dividend_yield:.4f}")
    
    return build_iv_surface_from_timeseries(
        stock_data=stock_data,
        option_data=option_data,
        expiration_dates=expiration_dates,
        exercise_style=exercise_style,
        tree=tree,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        **kwargs
    )


# ───────────────────────────────────────────────────────────────
# 7) Minimal Polygon & Alpaca wrapper (reuse your DataHandler if you prefer)
# ───────────────────────────────────────────────────────────────
from polygon import RESTClient                               # polygon-api-client
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
tz_utc = dt.timezone.utc

def get_spot(alpaca: StockHistoricalDataClient,
             sym: str,
             snap: dt.datetime):
    req = StockBarsRequest(symbol_or_symbols=sym,
                           timeframe=TimeFrame.Minute,
                           start=snap - dt.timedelta(minutes=3),
                           end=snap)
    df = alpaca.get_stock_bars(req).df.sort_index()
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(sym, level="symbol")
    return df["close"].iloc[-1]

# ───────────────────────────────────────────────────────────────
# 8) CLI
# ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",  required=True, help="Underlying ticker")
    p.add_argument("--date",    required=True, help="Snapshot date YYYY-MM-DD")
    p.add_argument("--time",    default="20:00:00", help="HH:MM:SS UTC")
    p.add_argument("--moneyness", nargs=2, type=float, default=(0.8, 1.2))
    p.add_argument("--mode", choices=["realtime", "timeseries"], default="realtime",
                   help="Build surface from real-time data or time series data")
    p.add_argument("--interactive", action="store_true", 
                   help="Use interactive mode for parameter selection")
    args = p.parse_args()

    # ── credentials
    alp_key = os.getenv("ALPACA_API_KEY")    or input("Alpaca key: ")
    alp_sec = os.getenv("ALPACA_API_SECRET") or input("Alpaca secret: ")
    pol_key = os.getenv("POLYGON_API_KEY")   or input("Polygon key: ")

    global client
    client = RESTClient(pol_key)
    alpaca  = StockHistoricalDataClient(alp_key, alp_sec)

    snapshot = dt.datetime.fromisoformat(f"{args.date}T{args.time}").replace(tzinfo=tz_utc)

    spot = get_spot(alpaca, args.symbol, snapshot)
    print(f"→ Spot price {args.symbol} = {spot:.2f}")

    # ---- fetch chain (max 1000)
    chain_raw = [c.ticker for c in
                 client.list_options_contracts(underlying_ticker=args.symbol,
                                                as_of=args.date, limit=1000)]
    chain = [t for t in chain_raw
             if args.moneyness[0]*spot <= decode_ticker(t)["strike"] <= args.moneyness[1]*spot]

    # ---- choose model interactively
    print("\nSelect pricing model:")
    print(" 1) Black-Scholes (analytic, European)")
    print(" 2) Binomial CRR tree (American OK)")
    print(" 3) Merton-76 Jump Diffusion (European)")
    choice = input("Your choice [1-3]: ").strip()
    model = { "1":"bs", "2":"binom", "3":"merton" }.get(choice, "bs")

    if args.mode == "realtime":
        # Original real-time surface building
        surf = build_surface(chain, spot, snapshot, model)
        print(f"Surface built: {surf.shape[0]} maturities × {surf.shape[1]} strikes")
        
        # Plot real-time surface
        if not surf.empty:
            from mpl_toolkits.mplot3d import Axes3D          # noqa
            X,Y = np.meshgrid(surf.columns.values, surf.index.values)
            fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, surf.values, linewidth=0.1, antialiased=True)
            ax.set_xlabel("Strike"); ax.set_ylabel("τ (yrs)"); ax.set_zlabel("σ_imp")
            ax.set_title(f"{args.symbol} IV surface – {model.upper()} – {args.date}")
            plt.tight_layout(); plt.show()
    
    elif args.mode == "timeseries":
        # Enhanced time series surface building
        print(f"\nBuilding IV surface from time series data for {args.symbol}...")
        
        # This would require historical data - for demo purposes, we'll show the interface
        print("Note: Time series mode requires historical stock and option data.")
        print("Use the build_iv_surface_from_timeseries() function directly with your data.")
        
        # Example usage (commented out as it requires actual data)
        """
        # Example: How to use the enhanced surface builder
        stock_data = pd.DataFrame({'price': [...]}, index=pd.DatetimeIndex([...]))
        option_data = {
            'AAPL250418C00200000': pd.Series([...], index=pd.DatetimeIndex([...])),
            'AAPL250418P00200000': pd.Series([...], index=pd.DatetimeIndex([...]))
        }
        expiration_dates = {
            'AAPL250418C00200000': '2025-04-18',
            'AAPL250418P00200000': '2025-04-18'
        }
        
        if args.interactive:
            surf = build_iv_surface_interactive(stock_data, option_data, expiration_dates)
        else:
            surf = build_iv_surface_from_timeseries(stock_data, option_data, expiration_dates)
        
        print(f"Time series surface built: {surf.shape[0]} maturities × {surf.shape[1]} strikes")
        """

def demonstrate_enhanced_iv_surface():
    """
    Interactive IV surface builder that prompts for real market data parameters.
    Similar to main.py workflow but focused on IV surface construction.
    """
    print("=== Enhanced IV Surface Builder with Real Market Data ===")
    print("This builds IV surfaces using real market data from Polygon/Alpaca APIs.")
    
    # Prompt for API keys
    print("\n🔑 API Configuration:")
    alpaca_key = input("Enter Alpaca API key (or leave blank for demo): ").strip()
    alpaca_secret = input("Enter Alpaca API secret (or leave blank for demo): ").strip()
    polygon_key = input("Enter Polygon API key (or leave blank for demo): ").strip()
    
    # Use demo keys if none provided
    if not alpaca_key:
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY
        alpaca_key = ALPACA_API_KEY
        alpaca_secret = ALPACA_SECRET_KEY
        print("Using Alpaca credentials from config")
    
    if not polygon_key:
        from config import POLYGON_API_KEY
        polygon_key = POLYGON_API_KEY
        print("Using Polygon credentials from config")
    
    # Initialize data handler
    try:
        from .data_handler import DataHandler
        data_handler = DataHandler(
            alpaca_api_key=alpaca_key,
            alpaca_secret=alpaca_secret,
            polygon_key=polygon_key
        )
        print("✅ Data handler initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize data handler: {e}")
        return
    
    # Prompt for underlying symbol
    print("\n📈 Underlying Selection:")
    print("Recommended highly liquid options:")
    print("  - TSLA (Tesla) - Very liquid, high volatility")
    print("  - AAPL (Apple) - Highly liquid, moderate volatility")
    print("  - SPY (S&P 500 ETF) - Most liquid, low volatility")
    print("  - QQQ (NASDAQ ETF) - Liquid, tech-focused")
    
    symbol = input("Enter underlying symbol (e.g., TSLA): ").strip().upper()
    if not symbol:
        symbol = "TSLA"
        print("Using TSLA as default (highly liquid)")
    
    # Prompt for date range
    print("\n📅 Date Range Selection:")
    today_str = dt.datetime.today().strftime("%Y-%m-%d")
    earliest_date = (dt.datetime.today() - dt.timedelta(days=2*365)).strftime("%Y-%m-%d")
    
    print(f"Polygon has 2 years of historical options data")
    print(f"Earliest available date: {earliest_date}")
    
    start_date = input(f"Enter start date (YYYY-MM-DD) [default: {today_str}]: ").strip()
    if not start_date:
        start_date = today_str
    
    end_date = input(f"Enter end date (YYYY-MM-DD) [default: {today_str}]: ").strip()
    if not end_date:
        end_date = today_str
    
    # Prompt for moneyness range
    print("\n💰 Moneyness Range:")
    print("Moneyness = Strike Price / Current Stock Price")
    print("  - 0.8 = 20% out-of-the-money")
    print("  - 1.0 = at-the-money")
    print("  - 1.2 = 20% in-the-money")
    
    min_moneyness = input("Enter minimum moneyness (0.5-1.0) [default: 0.8]: ").strip()
    if not min_moneyness:
        min_moneyness = 0.8
    else:
        min_moneyness = float(min_moneyness)
    
    max_moneyness = input("Enter maximum moneyness (1.0-1.5) [default: 1.2]: ").strip()
    if not max_moneyness:
        max_moneyness = 1.2
    else:
        max_moneyness = float(max_moneyness)
    
    # Prompt for risk-free rate
    print("\n🏦 Risk-Free Rate:")
    rfr_input = input("Enter risk-free rate as decimal (e.g., 0.05 for 5%) [default: 0.04]: ").strip()
    risk_free_rate = float(rfr_input) if rfr_input else 0.04
    
    # Prompt for dividend yield
    print("\n📊 Dividend Yield:")
    div_input = input("Enter dividend yield as decimal (e.g., 0.02 for 2%) [default: 0.0]: ").strip()
    dividend_yield = float(div_input) if div_input else 0.0
    
    # Prompt for exercise style
    print("\n🎯 Exercise Style:")
    print("1) American (default - most US equity options)")
    print("2) European")
    choice = input("Choose exercise style (1/2) [default: 1]: ").strip()
    exercise_style = "american" if choice != "2" else "european"
    
    # Prompt for tree type
    print("\n🌳 Binomial Tree Type:")
    print("1) CRR (Cox-Ross-Rubinstein) - default")
    print("2) JR (Jarrow-Rudd)")
    print("3) EQP (Equal Probability)")
    print("4) Trigeorgis")
    tree_choice = input("Choose tree type (1-4) [default: 1]: ").strip()
    tree_map = {"1": "crr", "2": "jr", "3": "eqp", "4": "trigeorgis"}
    tree = tree_map.get(tree_choice, "crr")
    
    print(f"\n🔍 Fetching market data for {symbol} from {start_date} to {end_date}...")
    print(f"Moneyness range: {min_moneyness:.2f} - {max_moneyness:.2f}")
    print(f"Risk-free rate: {risk_free_rate*100:.2f}%")
    print(f"Dividend yield: {dividend_yield*100:.2f}%")
    print(f"Exercise style: {exercise_style}")
    print(f"Tree type: {tree}")
    
    try:
        # Get IV surface data using the new DataHandler method
        print(f"\n🔬 Getting IV surface data...")
        surface_data = data_handler.get_iv_surface_data(
            underlying=symbol,
            start_date=start_date,
            end_date=end_date,
            moneyness_min=min_moneyness,
            moneyness_max=max_moneyness,
            max_options_per_expiry=10,
            as_of=start_date
        )
        
        if not surface_data:
            print("❌ Failed to get IV surface data")
            return
        
        # Build IV surface using the new approach
        print(f"\n🔬 Building IV surface using refactored approach...")
        surface = build_iv_surface_from_timeseries(
            data_handler=data_handler,
            surface_data=surface_data,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            exercise_style=exercise_style,
            tree=tree
        )
        
        if not surface.empty:
            print(f"✅ Success! IV surface built with {len(surface)} data points")
            print("\nSurface preview:")
            print(surface.head())
            
            # Plot the surface
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                # Create pivot table for 3D plotting
                pivot_surface = surface.pivot_table(
                    values='iv', 
                    index='ttm', 
                    columns='strike', 
                    aggfunc='mean'
                )
                
                if not pivot_surface.empty:
                    X, Y = np.meshgrid(pivot_surface.columns.values, pivot_surface.index.values)
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection="3d")
                    ax.plot_surface(X, Y, pivot_surface.values, linewidth=0.1, antialiased=True, alpha=0.8)
                    ax.set_xlabel("Strike Price")
                    ax.set_ylabel("Time to Expiry (years)")
                    ax.set_zlabel("Implied Volatility")
                    ax.set_title(f"{symbol} IV Surface - {exercise_style.title()} Options\n{start_date} to {end_date}")
                    plt.tight_layout()
                    plt.show()
                else:
                    print("⚠️ Not enough data points for 3D surface plot")
                
                # Save surface data
                filename = f"{symbol}_iv_surface_{start_date}_{end_date}.csv"
                surface.to_csv(filename, index=False)
                print(f"💾 Surface data saved to {filename}")
                
            except Exception as e:
                print(f"⚠️ Plotting failed: {e}")
                print("Surface data is still available for analysis")
        else:
            print("❌ No valid IV surface could be built from the market data.")
            print("\nDiagnostic information:")
            print(f"  - Surface data available: {bool(surface_data)}")
            print(f"  - Date range: {start_date} to {end_date}")
            print(f"  - Moneyness range: {min_moneyness} - {max_moneyness}")
            print("\nPossible issues:")
            print("  - No options found in the specified moneyness range")
            print("  - All options may have expired before the start date")
            print("  - IV calculations may be failing due to invalid option prices")
            print("  - Timezone issues in date calculations")
            print("  - Insufficient overlapping data between stock and options")
            
    except Exception as e:
        print(f"❌ Error building IV surface: {e}")
        print("Check your API keys and network connection.")


# ============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Check if we want to run the demonstration
    if len(sys.argv) == 1:
        print("No arguments provided. Running demonstration...")
        demonstrate_enhanced_iv_surface()
    else:
        main()
