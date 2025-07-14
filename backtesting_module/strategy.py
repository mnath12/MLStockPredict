from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.parameters = {}
        self.state = {}
        
    @abstractmethod
    def generate_raw_target(self, ts: pd.Timestamp, i: int, market_data: Dict[str, Any], 
                           portfolio_view: Dict[str, Any]) -> Dict[str, int]:
        """
        Generate raw target positions before applying mixins.
        
        Args:
            ts: Current timestamp
            i: Current bar index
            market_data: Current market data
            portfolio_view: Current portfolio state
            
        Returns:
            Dictionary of symbol -> target quantity
        """
        pass
    
    def on_bar(self, big_df: pd.DataFrame, i: int, portfolio_view: Dict[str, Any], 
               market_data: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        Main strategy entry point called on each bar.
        
        Args:
            big_df: Combined market data DataFrame
            i: Current bar index
            portfolio_view: Current portfolio state
            market_data: Additional market data
            
        Returns:
            Dictionary of symbol -> target quantity
        """
        if i >= len(big_df):
            return {}
            
        current_ts = big_df.index[i]
        
        # Prepare market data
        if market_data is None:
            market_data = self._extract_market_data(big_df, i)
        
        # Generate raw targets
        raw_targets = self.generate_raw_target(current_ts, i, market_data, portfolio_view)
        
        # Apply any mixins or adjustments
        final_targets = self._apply_mixins(raw_targets, market_data, portfolio_view)
        
        return final_targets
    
    def _extract_market_data(self, big_df: pd.DataFrame, i: int) -> Dict[str, Any]:
        """Extract market data from big_df for current bar."""
        market_data = {}
        
        if i < len(big_df):
            current_row = big_df.iloc[i]
            
            # Extract prices
            for col in current_row.index:
                if col.endswith('_close'):
                    symbol = col.replace('_close', '')
                    market_data[symbol] = current_row[col]
                elif col.endswith('_volume'):
                    symbol = col.replace('_volume', '')
                    market_data[f"{symbol}_volume"] = current_row[col]
        
        return market_data
    
    def _apply_mixins(self, raw_targets: Dict[str, int], market_data: Dict[str, Any],
                     portfolio_view: Dict[str, Any]) -> Dict[str, int]:
        """Apply strategy mixins (override in subclasses)."""
        return raw_targets
    
    def set_parameters(self, **params: Any) -> None:
        """Set strategy parameters."""
        self.parameters.update(params)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        return {
            'name': self.name,
            'parameters': self.parameters.copy(),
            'state': self.state.copy()
        }


class DeltaHedgingMixin:
    """
    Mixin for delta hedging functionality.
    """
    
    def __init__(self, delta_tolerance: float = 0.1, hedge_symbol: str = "SPY"):
        self.delta_tolerance = delta_tolerance
        self.hedge_symbol = hedge_symbol
    
    def apply_delta_hedge(self, raw_targets: Dict[str, int], market_data: Dict[str, Any],
                         portfolio_view: Dict[str, Any]) -> Dict[str, int]:
        """
        Apply delta hedging to raw targets.
        
        Args:
            raw_targets: Original target positions
            market_data: Current market data including Greeks
            portfolio_view: Current portfolio state
            
        Returns:
            Adjusted targets with delta hedging
        """
        adjusted_targets = raw_targets.copy()
        
        # Calculate current portfolio delta
        current_delta = self._calculate_portfolio_delta(portfolio_view, market_data)
        
        # Calculate delta from new targets
        target_delta = self._calculate_targets_delta(raw_targets, market_data)
        
        # Total delta after implementing targets
        total_delta = current_delta + target_delta
        
        # Check if hedging is needed
        if abs(total_delta) > self.delta_tolerance:
            # Calculate hedge quantity (negative to offset delta)
            hedge_qty = -int(round(total_delta))
            
            # Add hedge to targets
            current_hedge_target = adjusted_targets.get(self.hedge_symbol, 0)
            adjusted_targets[self.hedge_symbol] = current_hedge_target + hedge_qty
        
        return adjusted_targets
    
    def _calculate_portfolio_delta(self, portfolio_view: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate current portfolio delta."""
        total_delta = 0.0
        
        # Stock positions (delta = 1.0 per share)
        for symbol, stock in portfolio_view.get('stocks', {}).items():
            if stock.qty != 0:
                total_delta += stock.qty * 1.0
        
        # Option positions
        for symbol, option in portfolio_view.get('options', {}).items():
            if option.qty != 0:
                delta = market_data.get(f"{symbol}_delta", getattr(option, 'delta', 0.0))
                total_delta += option.qty * delta * 100  # 100 shares per contract
        
        return total_delta
    
    def _calculate_targets_delta(self, targets: Dict[str, int], 
                                market_data: Dict[str, Any]) -> float:
        """Calculate delta from target positions."""
        targets_delta = 0.0
        
        for symbol, qty in targets.items():
            if qty != 0:
                if self._is_option_symbol(symbol):
                    delta = market_data.get(f"{symbol}_delta", 0.0)
                    targets_delta += qty * delta * 100
                else:
                    # Stock delta = 1.0
                    targets_delta += qty * 1.0
        
        return targets_delta
    
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol represents an option."""
        return len(symbol) > 8 and any(char.isdigit() for char in symbol[-8:])


class DeltaGammaHedgingMixin:
    """
    Mixin for delta-gamma hedging functionality.
    """
    
    def __init__(self, delta_tolerance: float = 0.1, gamma_tolerance: float = 0.05,
                 hedge_symbol: str = "SPY", gamma_hedge_symbol: Optional[str] = None):
        self.delta_tolerance = delta_tolerance
        self.gamma_tolerance = gamma_tolerance
        self.hedge_symbol = hedge_symbol
        self.gamma_hedge_symbol = gamma_hedge_symbol
    
    def apply_delta_gamma_hedge(self, raw_targets: Dict[str, int], market_data: Dict[str, Any],
                               portfolio_view: Dict[str, Any]) -> Dict[str, int]:
        """
        Apply delta-gamma hedging to raw targets.
        """
        adjusted_targets = raw_targets.copy()
        
        # Calculate current portfolio Greeks
        current_delta = self._calculate_portfolio_delta(portfolio_view, market_data)
        current_gamma = self._calculate_portfolio_gamma(portfolio_view, market_data)
        
        # Calculate Greeks from targets
        target_delta = self._calculate_targets_delta(raw_targets, market_data)
        target_gamma = self._calculate_targets_gamma(raw_targets, market_data)
        
        # Total Greeks after implementing targets
        total_delta = current_delta + target_delta
        total_gamma = current_gamma + target_gamma
        
        # Step 1: Hedge gamma with options (if available)
        if (abs(total_gamma) > self.gamma_tolerance and 
            self.gamma_hedge_symbol and 
            f"{self.gamma_hedge_symbol}_gamma" in market_data):
            
            option_gamma = market_data[f"{self.gamma_hedge_symbol}_gamma"]
            if abs(option_gamma) > 1e-6:
                gamma_hedge_qty = -int(round(total_gamma / option_gamma))
                
                current_option_target = adjusted_targets.get(self.gamma_hedge_symbol, 0)
                adjusted_targets[self.gamma_hedge_symbol] = current_option_target + gamma_hedge_qty
                
                # Update total delta after gamma hedge
                option_delta = market_data.get(f"{self.gamma_hedge_symbol}_delta", 0.0)
                total_delta += gamma_hedge_qty * option_delta * 100
        
        # Step 2: Hedge remaining delta with stock
        if abs(total_delta) > self.delta_tolerance:
            delta_hedge_qty = -int(round(total_delta))
            
            current_stock_target = adjusted_targets.get(self.hedge_symbol, 0)
            adjusted_targets[self.hedge_symbol] = current_stock_target + delta_hedge_qty
        
        return adjusted_targets
    
    def _calculate_portfolio_delta(self, portfolio_view: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate current portfolio delta."""
        total_delta = 0.0
        
        for symbol, stock in portfolio_view.get('stocks', {}).items():
            if stock.qty != 0:
                total_delta += stock.qty * 1.0
        
        for symbol, option in portfolio_view.get('options', {}).items():
            if option.qty != 0:
                delta = market_data.get(f"{symbol}_delta", getattr(option, 'delta', 0.0))
                total_delta += option.qty * delta * 100
        
        return total_delta
    
    def _calculate_portfolio_gamma(self, portfolio_view: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate current portfolio gamma."""
        total_gamma = 0.0
        
        for symbol, option in portfolio_view.get('options', {}).items():
            if option.qty != 0:
                gamma = market_data.get(f"{symbol}_gamma", getattr(option, 'gamma', 0.0))
                total_gamma += option.qty * gamma * 100
        
        return total_gamma
    
    def _calculate_targets_delta(self, targets: Dict[str, int], 
                                market_data: Dict[str, Any]) -> float:
        """Calculate delta from target positions."""
        targets_delta = 0.0
        
        for symbol, qty in targets.items():
            if qty != 0:
                if self._is_option_symbol(symbol):
                    delta = market_data.get(f"{symbol}_delta", 0.0)
                    targets_delta += qty * delta * 100
                else:
                    targets_delta += qty * 1.0
        
        return targets_delta
    
    def _calculate_targets_gamma(self, targets: Dict[str, int], 
                                market_data: Dict[str, Any]) -> float:
        """Calculate gamma from target positions."""
        targets_gamma = 0.0
        
        for symbol, qty in targets.items():
            if qty != 0 and self._is_option_symbol(symbol):
                gamma = market_data.get(f"{symbol}_gamma", 0.0)
                targets_gamma += qty * gamma * 100
        
        return targets_gamma
    
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol represents an option."""
        return len(symbol) > 8 and any(char.isdigit() for char in symbol[-8:])


class LSTMStrategy(BaseStrategy):
    """
    LSTM-based trading strategy.
    """
    
    def __init__(self, lookback_window: int = 20, prediction_horizon: int = 1):
        super().__init__("LSTM_Strategy")
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.model = None  # Placeholder for LSTM model
        
    def generate_raw_target(self, ts: pd.Timestamp, i: int, market_data: Dict[str, Any],
                           portfolio_view: Dict[str, Any]) -> Dict[str, int]:
        """
        Generate targets using LSTM predictions.
        """
        # Placeholder implementation
        # In practice, this would use a trained LSTM model
        pass


class BuyAndHoldStrategy(BaseStrategy):
    """
    Simple buy and hold strategy for testing and benchmarking.
    
    Strategy Logic:
    1. On first bar: Buy specified quantity of target symbol
    2. On subsequent bars: Maintain existing position (no new trades)
    3. Hold until end of backtest period
    
    This strategy is useful for:
    - Benchmarking other strategies
    - Testing the backtesting framework
    - Providing baseline performance comparison
    """
    
    def __init__(self, symbol: str = "SPY", quantity: int = 100):
        """
        Initialize Buy and Hold strategy.
        
        Args:
            symbol: Symbol to buy and hold (default: "SPY")
            quantity: Number of shares to buy (default: 100)
        """
        super().__init__("BuyAndHold")
        self.symbol = symbol
        self.quantity = quantity
        self.initialized = False
        
        # Store parameters for reference
        self.set_parameters(
            target_symbol=symbol,
            target_quantity=quantity
        )
    
    def generate_raw_target(self, ts: pd.Timestamp, i: int, market_data: Dict[str, Any],
                           portfolio_view: Dict[str, Any]) -> Dict[str, int]:
        """
        Generate buy and hold targets.
        
        Args:
            ts: Current timestamp
            i: Current bar index  
            market_data: Current market data including prices
            portfolio_view: Current portfolio state
            
        Returns:
            Dictionary with target positions
        """
        # On first call with valid market data, buy the target quantity
        if not self.initialized and self.symbol in market_data:
            self.initialized = True
            print(f"Buy and Hold: Initial purchase of {self.quantity} shares of {self.symbol}")
            return {self.symbol: self.quantity}
        
        # After initial purchase, maintain current position
        current_qty = self._get_current_position(portfolio_view)
        
        # Return current position to maintain it
        return {self.symbol: current_qty}
    
    def _get_current_position(self, portfolio_view: Dict[str, Any]) -> int:
        """
        Get current position quantity for the target symbol.
        
        Args:
            portfolio_view: Current portfolio state
            
        Returns:
            Current quantity held
        """
        # Check stock positions
        stocks = portfolio_view.get('stocks', {})
        if self.symbol in stocks:
            return stocks[self.symbol].qty
        
        # If no position exists, return 0
        return 0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current state.
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': self.name,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'initialized': self.initialized,
            'strategy_type': 'Buy and Hold',
            'description': f'Buy {self.quantity} shares of {self.symbol} and hold'
        }