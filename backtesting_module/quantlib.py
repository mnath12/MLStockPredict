import QuantLib as ql

def compute_iv_quantlib(
    spot_price: float,
    option_price: float,
    strike_price: float,
    days_to_maturity: int,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",          # "call" or "put"
    exercise_style: str = "american",   # "american" or "european"
    tree: str = "crr",                  # binomial tree: "crr", "jr", "eqp", "trigeorgis", etc.
    steps: int = 1000,
    accuracy: float = 1e-4,
    max_evals: int = 1000,
    min_vol: float = 1e-8,
    max_vol: float = 4.0,
    vol_guess: float = 0.20,
    rates_day_count: ql.DayCounter = ql.Actual360(),
    vol_day_count: ql.DayCounter = ql.Actual365Fixed(),
    calendar: ql.Calendar = ql.NullCalendar(),
) -> float:
    """
    Return Black implied volatility for a plain-vanilla option priced at option_price.
    Units: rates as decimals (e.g., 0.05), vol as decimal (e.g., 0.2).
    days_to_maturity is in *calendar days* when using NullCalendar.
    """

    if days_to_maturity <= 0:
        raise ValueError("days_to_maturity must be positive.")

    # Evaluation date
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Term structures
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate, rates_day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield, rates_day_count)
    )

    # Spot & vol (constant vol handle via SimpleQuote)
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    vol_quote = ql.SimpleQuote(vol_guess)
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_quote), vol_day_count)
    )

    # Process
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, risk_free_ts, vol_ts)

    # Option spec
    expiration = today + ql.Period(days_to_maturity, ql.Days)
    ql_opt_type = ql.Option.Call if option_type.lower() == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_opt_type, strike_price)

    if exercise_style.lower() == "american":
        exercise = ql.AmericanExercise(today, expiration)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.BinomialVanillaEngine(bsm_process, tree, steps)
    else:
        exercise = ql.EuropeanExercise(expiration)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.AnalyticEuropeanEngine(bsm_process)

    option.setPricingEngine(engine)

    # Root-finding for implied vol
    iv = option.impliedVolatility(
        option_price, bsm_process, accuracy, max_evals, min_vol, max_vol
    )
    return float(iv)
    
iv = compute_iv_quantlib(
    spot_price=316.06,
    option_price=7.015,
    strike_price=317.5,
    days_to_maturity=5,
    risk_free_rate=0.0442,
    option_type="put",
    exercise_style="american",  # or "european"
    tree="crr",
    steps=1000
)
print(iv)
