from __future__ import annotations

import pandas as pd
import numpy as np
import uuid
from typing import List, Dict, Optional

from .entities import Order, Fill


class ExecutionHandler:
    """
    Simulates order execution with realistic market effects.
    """
    
    def __init__(self, commission_per_share: float = 0.005, slippage_std: float = 0.001):
        self.commission_per_share = commission_per_share
        self.slippage_std = slippage_std  # Standard deviation of slippage as fraction of price
        
    def get_fills(self, orders: List[Order], market_data: Optional[Dict[str, float]] = None) -> List[Fill]:
        """
        Execute a list of orders and return fills.
        
        Args:
            orders: List of orders to execute
            market_data: Dictionary of symbol -> current price
            
        Returns:
            List of Fill objects
        """
        if not orders:
            return []
            
        fills = []
        current_time = pd.Timestamp.now()
        
        for order in orders:
            fill = self._execute_single_order(order, market_data, current_time)
            if fill:
                fills.append(fill)
                
        return fills
    
    def _execute_single_order(self, order: Order, market_data: Optional[Dict[str, float]], 
                             timestamp: pd.Timestamp) -> Optional[Fill]:
        """
        Execute a single order and return a fill.
        """
        # Get market price
        if market_data and order.symbol in market_data:
            market_price = market_data[order.symbol]
        else:
            # Use limit price if available, otherwise assume $100 default
            market_price = order.limit if order.limit else 100.0
            
        # Calculate execution price with slippage
        execution_price = self._apply_slippage(market_price, order)
        
        # Calculate commission
        commission = abs(order.qty) * self.commission_per_share
        
        # Create fill
        fill = Fill(
            symbol=order.symbol,
            qty=order.qty,
            price=execution_price,
            timestamp=timestamp,
            order_id=order.order_id or str(uuid.uuid4()),
            commission=commission
        )
        
        return fill
    
    def _apply_slippage(self, market_price: float, order: Order) -> float:
        """
        Apply slippage to market price based on order characteristics.
        """
        if order.type == 'limit':
            # Limit orders execute at limit price (simplified)
            return order.limit if order.limit else market_price
        
        # Market orders get slippage
        # Slippage is worse for larger orders and for sells vs buys
        base_slippage = np.random.normal(0, self.slippage_std)
        
        # Add size impact (larger orders get more slippage)
        size_impact = min(0.001, abs(order.qty) / 10000 * 0.0005)
        
        # Direction impact (buys pay more, sells get less)
        if order.side.lower() == 'buy':
            slippage = abs(base_slippage) + size_impact
        else:  # sell
            slippage = -abs(base_slippage) - size_impact
            
        execution_price = market_price * (1 + slippage)
        
        # Ensure positive price
        return max(0.01, execution_price)
    
    def get_market_impact(self, order: Order, avg_volume: float = 1000000) -> float:
        """
        Estimate market impact of an order.
        
        Args:
            order: The order to analyze
            avg_volume: Average daily volume for the symbol
            
        Returns:
            Estimated market impact as fraction of price
        """
        if avg_volume <= 0:
            return 0.0
            
        participation_rate = abs(order.qty) / avg_volume
        
        # Square root impact law (simplified)
        impact = 0.1 * np.sqrt(participation_rate)
        
        return min(impact, 0.05)  # Cap at 5%