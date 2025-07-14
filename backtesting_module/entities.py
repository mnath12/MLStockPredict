from __future__ import annotations

import pandas as pd
from typing import Optional, Any


class Order:
    def __init__(self, symbol: str, qty: int, side: str, type: str, limit: Optional[float] = None, 
                 time_in_force: Optional[str] = None, **kwargs: Any):
        self.symbol = symbol
        self.qty = qty
        self.side = side  # 'buy' or 'sell'
        self.type = type  # 'market' or 'limit'
        self.limit = limit
        self.time_in_force = time_in_force or 'day'
        self.timestamp = kwargs.get('timestamp', pd.Timestamp.now())
        self.order_id = kwargs.get('order_id', None)
        
        # Any other fields can be added via kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self) -> str:
        return f"Order({self.symbol}, {self.side} {self.qty} @ {self.type})"


class Fill:
    def __init__(self, symbol: str, qty: int, price: float, timestamp: pd.Timestamp, 
                 order_id: Optional[str] = None, commission: float = 0.0):
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.timestamp = timestamp
        self.order_id = order_id
        self.commission = commission

    def __repr__(self) -> str:
        return f"Fill({self.symbol}, {self.qty} @ ${self.price:.2f})"


class Position:
    def __init__(self, symbol: str, qty: int, avg_cost: float, **kwargs: Any):
        self.symbol = symbol
        self.qty = qty
        self.avg_cost = avg_cost
        self.market_value = kwargs.get('market_value', 0.0)
        self.unrealized_pnl = kwargs.get('unrealized_pnl', 0.0)
        
        # Any other useful fields
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L based on current price."""
        self.market_value = self.qty * current_price
        self.unrealized_pnl = self.market_value - (self.qty * self.avg_cost)

    def __repr__(self) -> str:
        return f"Position({self.symbol}, {self.qty} @ ${self.avg_cost:.2f})"


class Option:
    def __init__(self, symbol: str, expiration_date: pd.Timestamp, strike_price: float, 
                 type: str = "call", qty: int = 0, avg_cost: float = 0.0, **kwargs: Any):
        self.symbol = symbol
        self.expiration_date = expiration_date
        self.strike_price = strike_price
        self.type = type  # 'call' or 'put'
        self.qty = qty
        self.avg_cost = avg_cost
        self.underlying = kwargs.get('underlying', symbol[:3])  # Extract underlying from symbol
        self.market_value = kwargs.get('market_value', 0.0)
        self.unrealized_pnl = kwargs.get('unrealized_pnl', 0.0)
        
        # Greeks
        self.delta = kwargs.get('delta', 0.0)
        self.gamma = kwargs.get('gamma', 0.0)
        self.vega = kwargs.get('vega', 0.0)
        self.theta = kwargs.get('theta', 0.0)
        self.rho = kwargs.get('rho', 0.0)
        
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L based on current price."""
        self.market_value = self.qty * current_price * 100  # Options are typically 100 shares per contract
        self.unrealized_pnl = self.market_value - (self.qty * self.avg_cost * 100)

    def time_to_expiry(self, current_date: pd.Timestamp) -> float:
        """Calculate time to expiry in years."""
        days_to_expiry = (self.expiration_date - current_date).days
        return max(0, days_to_expiry / 365.0)

    def is_expired(self, current_date: pd.Timestamp) -> bool:
        """Check if option is expired."""
        return current_date >= self.expiration_date

    def __repr__(self) -> str:
        return f"Option({self.symbol}, {self.type} {self.strike_price} exp:{self.expiration_date.date()})"


class Stock:
    def __init__(self, symbol: str, qty: int = 0, avg_cost: float = 0.0, **kwargs: Any):
        self.symbol = symbol
        self.qty = qty
        self.avg_cost = avg_cost
        self.market_value = kwargs.get('market_value', 0.0)
        self.unrealized_pnl = kwargs.get('unrealized_pnl', 0.0)
        
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L based on current price."""
        self.market_value = self.qty * current_price
        self.unrealized_pnl = self.market_value - (self.qty * self.avg_cost)

    def __repr__(self) -> str:
        return f"Stock({self.symbol}, {self.qty} @ ${self.avg_cost:.2f})"