"""
Minimal sanity-check for GreeksEngine.

Run:
    pytest -q tests/test_greeks_engine.py
"""

from math import isfinite
import pytest
from typing import cast, Literal

from ..greeks_engine import GreeksEngine


# Common option parameters
OPT_ARGS = dict(
    underlying_price = 100.0,
    strike_price     = 100.0,
    time_to_expiry   = 0.50,   # 6 months
    risk_free_rate   = 0.05,
    volatility       = 0.20,
    option_type      = "call",
)


@pytest.mark.parametrize("model", ["black_scholes", "jump_diffusion"])
def test_greeks_ranges(model: str) -> None:
    """
    Ensure each engine returns finite, sign-correct Greeks.
    We keep the bands loose—this is a smoke test, not a validator.
    """
    eng = GreeksEngine(pricing_model=cast(Literal["black_scholes", "jump_diffusion"], model))
    g   = eng.compute_single_option(**OPT_ARGS) # type: ignore[arg-type]

    # Delta of an at-the-money call ~0.5
    assert 0.35 < g["delta"] < 0.65

    # Core Greeks should be positive for a plain call
    assert g["gamma"] > 0
    assert g["vega"]  > 0

    # Theta (per day) is usually negative for long calls
    assert g["theta"] < 0

    # Rho positive for calls when r > 0
    assert g["rho"] > 0

    # Price must be finite
    assert isfinite(g["price"])
