"""
Smoke-test the jump-diffusion branch of GreeksEngine.

Run:
    pytest -q tests/test_jumpdiffusion_only.py
"""

from math import isfinite
import pytest
import QuantLib as ql

from backtesting_module.greeks_engine import GreeksEngine


def test_quantlib_has_jump_engine() -> None:
    """QuantLib ≥1.31 should expose JumpDiffusionEngine."""
    assert hasattr(ql, "JumpDiffusionEngine"), (
        "QuantLib is too old - upgrade to >=1.31"
    )


def test_greeks_engine_jump_diffusion() -> None:
    """GreeksEngine(jump_diffusion) returns sensible, finite values."""
    eng = GreeksEngine(pricing_model="jump_diffusion")

    greeks = eng.compute_single_option(
        underlying_price = 100.0,
        strike_price     = 100.0,
        time_to_expiry   = 0.25,         # 3 months
        risk_free_rate   = 0.03,
        volatility       = 0.25,
        option_type      = "call",
        jump_intensity   = 0.5,          # λ
        jump_volatility  = 0.2,          # σ_J
        jump_mean        = -0.05,        # μ_J
    )

    # Ensure the dict has all expected keys and finite values
    expected = {"delta", "gamma", "vega", "theta", "rho", "price"}
    assert expected.issubset(greeks.keys())

    for k in expected:
        assert isfinite(greeks[k]), f"{k} is not finite: {greeks[k]}"

    print(greeks)

test_greeks_engine_jump_diffusion()