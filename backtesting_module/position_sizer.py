from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional, Any
import uuid

from .entities import Order
from .portfolio import Portfolio


class PositionSizer:
    """
    Converts target positions into executable orders.
    """
    
    def __init__(self, max_position_size: float = 0.1, max_order_size: int = 1000,
                 min_order_size: int = 1):
        self.max_position_size = max_position_size  # Max position as fraction of portfolio
        self.max_order_size = max_order_size  # Max order size in shares/contracts
        self.min_order_size = min_order_size  # Min order size
        
    def get_orders(self, targets: Dict[str, int], portfolio: Portfolio, 
                   timestamp: Optional[pd.Timestamp] = None) -> List[Order]:
        """
        Convert target positions to orders.
        
        Args:
            targets: Dictionary of symbol -> target quantity
            portfolio: Current portfolio state
            timestamp: Timestamp for orders
            
        Returns:
            List of orders to execute
        """
        if not targets:
            return []
            
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        orders = []
        portfolio_view = portfolio.portfolio_view()
        
        for symbol, target_qty in targets.items():
            # Get current position
            current_qty = self._get_current_position(symbol, portfolio_view)
            
            # Calculate required order quantity
            order_qty = target_qty - current_qty
            
            if abs(order_qty) < self.min_order_size:
                continue  # Skip small orders
            
            # Apply size constraints
            order_qty = self._apply_size_constraints(order_qty, symbol, portfolio_view)
            
            if order_qty != 0:
                order = self._create_order(symbol, order_qty, timestamp)
                orders.append(order)
        
        return orders
    
    def _get_current_position(self, symbol: str, portfolio_view: Dict[str, Any]) -> int:
        """Get current position quantity for a symbol."""
        # Check stocks
        stocks = portfolio_view.get('stocks', {})
        if symbol in stocks:
            return stocks[symbol].qty
        
        # Check options
        options = portfolio_view.get('options', {})
        if symbol in options:
            return options[symbol].qty
        
        return 0
    
    def _apply_size_constraints(self, order_qty: int, symbol: str, 
                               portfolio_view: Dict[str, Any]) -> int:
        """Apply position sizing constraints."""
        # Apply maximum order size constraint
        if abs(order_qty) > self.max_order_size:
            order_qty = self.max_order_size if order_qty > 0 else -self.max_order_size
        
        # Apply maximum position size constraint (simplified)
        portfolio_value = portfolio_view.get('portfolio_value', 100000)
        max_position_value = portfolio_value * self.max_position_size
        
        # For now, just return the constrained order quantity
        # In practice, you'd calculate position value and check constraints
        
        return order_qty
    
    def _create_order(self, symbol: str, qty: int, timestamp: pd.Timestamp) -> Order:
        """Create an order object."""
        side = 'buy' if qty > 0 else 'sell'
        
        return Order(
            symbol=symbol,
            qty=abs(qty),
            side=side,
            type='market',  # Default to market orders
            time_in_force='day',
            timestamp=timestamp,
            order_id=str(uuid.uuid4())
        )
    
    def calculate_target_weights(self, symbols: List[str], 
                                strategy: str = "equal_weight") -> Dict[str, float]:
        """
        Calculate target weights for a list of symbols.
        
        Args:
            symbols: List of symbols
            strategy: Weighting strategy ('equal_weight', 'market_cap', etc.)
            
        Returns:
            Dictionary of symbol -> weight
        """
        if not symbols:
            return {}
        
        if strategy == "equal_weight":
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols}
        
        elif strategy == "market_cap":
            # Placeholder - in practice would use actual market cap data
            # For now, assign random weights that sum to 1
            import numpy as np
            weights = np.random.dirichlet(np.ones(len(symbols)))
            return {symbol: weight for symbol, weight in zip(symbols, weights)}
        
        else:
            # Default to equal weight
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols}
    
    def calculate_position_sizes(self, target_weights: Dict[str, float],
                                portfolio_value: float,
                                current_prices: Dict[str, float]) -> Dict[str, int]:
        """
        Convert target weights to position sizes.
        
        Args:
            target_weights: Dictionary of symbol -> target weight
            portfolio_value: Total portfolio value
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary of symbol -> target quantity
        """
        position_sizes = {}
        
        for symbol, weight in target_weights.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                target_value = portfolio_value * weight
                target_qty = int(target_value / current_prices[symbol])
                position_sizes[symbol] = target_qty
            else:
                position_sizes[symbol] = 0
        
        return position_sizes
    
    def apply_risk_constraints(self, targets: Dict[str, int], 
                              portfolio: Portfolio,
                              max_portfolio_delta: float = 1.0,
                              max_single_position: float = 0.2) -> Dict[str, int]:
        """
        Apply risk-based constraints to target positions.
        
        Args:
            targets: Original target positions
            portfolio: Current portfolio
            max_portfolio_delta: Maximum allowed portfolio delta
            max_single_position: Maximum position size as fraction of portfolio
            
        Returns:
            Adjusted target positions
        """
        adjusted_targets = targets.copy()
        portfolio_view = portfolio.portfolio_view()
        portfolio_value = portfolio_view.get('portfolio_value', 100000)
        
        # Apply single position size constraint
        for symbol, qty in list(adjusted_targets.items()):
            # Estimate position value (simplified)
            estimated_price = 100.0  # Default price assumption
            position_value = abs(qty) * estimated_price
            
            if position_value > max_single_position * portfolio_value:
                # Scale down position
                max_qty = int(max_single_position * portfolio_value / estimated_price)
                adjusted_targets[symbol] = max_qty if qty > 0 else -max_qty
        
        # Additional risk constraints could be added here
        # - Sector concentration limits
        # - Correlation limits
        # - Volatility-based sizing
        
        return adjusted_targets
    
    def get_rebalancing_orders(self, target_weights: Dict[str, float],
                              portfolio: Portfolio,
                              current_prices: Dict[str, float],
                              rebalance_threshold: float = 0.05) -> List[Order]:
        """
        Generate rebalancing orders based on target weights.
        
        Args:
            target_weights: Target allocation weights
            portfolio: Current portfolio
            current_prices: Current market prices
            rebalance_threshold: Minimum deviation to trigger rebalancing
            
        Returns:
            List of rebalancing orders
        """
        portfolio_view = portfolio.portfolio_view()
        portfolio_value = portfolio_view.get('portfolio_value', 100000)
        
        # Calculate target quantities
        target_quantities = self.calculate_position_sizes(
            target_weights, portfolio_value, current_prices
        )
        
        # Only rebalance if deviation exceeds threshold
        rebalance_targets = {}
        
        for symbol, target_qty in target_quantities.items():
            current_qty = self._get_current_position(symbol, portfolio_view)
            
            if symbol in current_prices and current_prices[symbol] > 0:
                current_value = current_qty * current_prices[symbol]
                target_value = target_qty * current_prices[symbol]
                
                # Check if deviation exceeds threshold
                if portfolio_value > 0:
                    current_weight = current_value / portfolio_value
                    target_weight = target_weights.get(symbol, 0)
                    
                    if abs(current_weight - target_weight) > rebalance_threshold:
                        rebalance_targets[symbol] = target_qty
        
        return self.get_orders(rebalance_targets, portfolio)