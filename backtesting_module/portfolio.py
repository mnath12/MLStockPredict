from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .entities import Fill, Stock, Option, Position


class Portfolio:
    """
    Tracks portfolio positions, cash, and performance metrics.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        
        # Position tracking
        self.stocks: Dict[str, Stock] = {}
        self.options: Dict[str, Option] = {}
        
        # Performance tracking
        self.portfolio_values: List[float] = [initial_cash]
        self.timestamps: List[pd.Timestamp] = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        
        # Trade tracking
        self.fills_history: List[Fill] = []
        self.daily_pnl: List[float] = []
        
    def update_with_fill(self, fill: Fill) -> None:
        """
        Update portfolio with a new fill.
        
        Args:
            fill: Fill object representing executed trade
        """
        self.fills_history.append(fill)
        
        # Determine if this is a stock or option
        if self._is_option_symbol(fill.symbol):
            self._update_option_position(fill)
        else:
            self._update_stock_position(fill)
            
        # Update cash (subtract cost for buys, add proceeds for sells)
        trade_value = fill.qty * fill.price + fill.commission
        if fill.qty > 0:  # Buy
            self.cash -= trade_value
        else:  # Sell
            self.cash += abs(trade_value)
    
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol represents an option contract."""
        # Simple heuristic: options typically have longer symbols with specific patterns
        return len(symbol) > 8 and any(char.isdigit() for char in symbol[-8:])
    
    def _update_stock_position(self, fill: Fill) -> None:
        """Update stock position with fill."""
        symbol = fill.symbol
        
        if symbol not in self.stocks:
            self.stocks[symbol] = Stock(symbol=symbol, qty=0, avg_cost=0.0)
        
        stock = self.stocks[symbol]
        old_qty = stock.qty
        old_cost = stock.avg_cost
        
        if fill.qty > 0:  # Buy
            # Calculate new weighted average cost
            total_cost = (old_qty * old_cost) + (fill.qty * fill.price)
            new_qty = old_qty + fill.qty
            
            if new_qty != 0:
                stock.avg_cost = total_cost / new_qty
            stock.qty = new_qty
            
        else:  # Sell
            # Calculate realized P&L
            sell_qty = abs(fill.qty)
            realized_pnl = sell_qty * (fill.price - old_cost)
            self.realized_pnl += realized_pnl
            
            # Update position
            stock.qty += fill.qty  # fill.qty is negative for sells
            
            # If position closed, reset avg_cost
            if stock.qty == 0:
                stock.avg_cost = 0.0
    
    def _update_option_position(self, fill: Fill) -> None:
        """Update option position with fill."""
        symbol = fill.symbol
        
        if symbol not in self.options:
            # Parse option details from symbol (simplified)
            underlying, strike, expiry, option_type = self._parse_option_symbol(symbol)
            self.options[symbol] = Option(
                symbol=symbol,
                expiration_date=expiry,
                strike_price=strike,
                type=option_type,
                qty=0,
                avg_cost=0.0
            )
        
        option = self.options[symbol]
        old_qty = option.qty
        old_cost = option.avg_cost
        
        if fill.qty > 0:  # Buy
            total_cost = (old_qty * old_cost) + (fill.qty * fill.price)
            new_qty = old_qty + fill.qty
            
            if new_qty != 0:
                option.avg_cost = total_cost / new_qty
            option.qty = new_qty
            
        else:  # Sell
            sell_qty = abs(fill.qty)
            realized_pnl = sell_qty * (fill.price - old_cost) * 100  # Options are per 100 shares
            self.realized_pnl += realized_pnl
            
            option.qty += fill.qty
            
            if option.qty == 0:
                option.avg_cost = 0.0
    
    def _parse_option_symbol(self, symbol: str) -> tuple:
        """Parse option symbol to extract underlying, strike, expiry, and type."""
        # Simplified parser - in practice this would be more sophisticated
        try:
            # Example: AAPL240315C00150000
            underlying = symbol[:4]  # First 4 characters
            
            # Extract expiry (next 6 digits)
            date_part = symbol[4:10]
            expiry = pd.to_datetime(f"20{date_part}", format='%Y%m%d')
            
            # Extract type
            option_type = "call" if symbol[10] == 'C' else "put"
            
            # Extract strike (remaining digits divided by 1000)
            strike_part = symbol[11:]
            strike = float(strike_part) / 1000.0
            
            return underlying, strike, expiry, option_type
            
        except:
            # Default fallback
            return "SPY", 100.0, pd.Timestamp('2024-12-31'), "call"
    
    def portfolio_view(self) -> Dict[str, Any]:
        """
        Return current portfolio state.
        
        Returns:
            Dictionary with portfolio information
        """
        return {
            'cash': self.cash,
            'stocks': dict(self.stocks),
            'options': dict(self.options),
            'total_positions': len(self.stocks) + len(self.options),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'portfolio_value': self.portfolio_values[-1] if self.portfolio_values else self.initial_cash
        }
    
    def update_portfolio_value(self, current_prices: Dict[str, float], 
                              timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        Update portfolio value based on current market prices.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
            
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        
        # Stock positions
        for symbol, stock in self.stocks.items():
            if stock.qty != 0 and symbol in current_prices:
                current_value = stock.qty * current_prices[symbol]
                cost_basis = stock.qty * stock.avg_cost
                unrealized_pnl += current_value - cost_basis
                stock.update_market_value(current_prices[symbol])
        
        # Option positions
        for symbol, option in self.options.items():
            if option.qty != 0 and symbol in current_prices:
                current_value = option.qty * current_prices[symbol] * 100  # Options per 100 shares
                cost_basis = option.qty * option.avg_cost * 100
                unrealized_pnl += current_value - cost_basis
                option.update_market_value(current_prices[symbol])
        
        self.unrealized_pnl = unrealized_pnl
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Calculate total portfolio value
        portfolio_value = self.cash + self._calculate_positions_value(current_prices)
        
        # Track history
        self.portfolio_values.append(portfolio_value)
        self.timestamps.append(timestamp)
        
        # Calculate daily P&L
        if len(self.portfolio_values) > 1:
            daily_change = portfolio_value - self.portfolio_values[-2]
            self.daily_pnl.append(daily_change)
    
    def _calculate_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total value of all positions."""
        total_value = 0.0
        
        # Stock positions
        for symbol, stock in self.stocks.items():
            if stock.qty != 0 and symbol in current_prices:
                total_value += stock.qty * current_prices[symbol]
        
        # Option positions
        for symbol, option in self.options.items():
            if option.qty != 0 and symbol in current_prices:
                total_value += option.qty * current_prices[symbol] * 100
        
        return total_value
    
    def get_portfolio_delta(self, greeks_data: Dict[str, float]) -> float:
        """
        Calculate total portfolio delta.
        
        Args:
            greeks_data: Dictionary with symbol -> delta values
            
        Returns:
            Total portfolio delta
        """
        total_delta = 0.0
        
        # Stock positions (delta = 1.0 per share)
        for symbol, stock in self.stocks.items():
            if stock.qty != 0:
                total_delta += stock.qty * 1.0
        
        # Option positions
        for symbol, option in self.options.items():
            if option.qty != 0:
                delta_key = f"{symbol}_delta"
                if delta_key in greeks_data:
                    total_delta += option.qty * greeks_data[delta_key] * 100
                elif hasattr(option, 'delta'):
                    total_delta += option.qty * option.delta * 100
        
        return total_delta
    
    def get_portfolio_gamma(self, greeks_data: Dict[str, float]) -> float:
        """Calculate total portfolio gamma."""
        total_gamma = 0.0
        
        # Only options have gamma
        for symbol, option in self.options.items():
            if option.qty != 0:
                gamma_key = f"{symbol}_gamma"
                if gamma_key in greeks_data:
                    total_gamma += option.qty * greeks_data[gamma_key] * 100
                elif hasattr(option, 'gamma'):
                    total_gamma += option.qty * option.gamma * 100
        
        return total_gamma
    
    def get_performance_metrics(self, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with performance metrics
        """
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'num_trades': len(self.fills_history)
            }
        
        # Calculate returns
        current_value = self.portfolio_values[-1]
        total_return = (current_value - self.initial_cash) / self.initial_cash
        
        # Calculate Sharpe ratio
        if len(self.daily_pnl) > 1:
            daily_returns = np.array(self.daily_pnl) / np.array(self.portfolio_values[:-1])
            excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) > 0:
                sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
            else:
                sharpe_ratio = 0.0
                
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
            volatility = 0.0
        
        # Calculate maximum drawdown
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len(self.fills_history),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """Return summary of all positions as DataFrame."""
        positions = []
        
        # Stock positions
        for symbol, stock in self.stocks.items():
            if stock.qty != 0:
                positions.append({
                    'symbol': symbol,
                    'type': 'stock',
                    'quantity': stock.qty,
                    'avg_cost': stock.avg_cost,
                    'market_value': getattr(stock, 'market_value', 0),
                    'unrealized_pnl': getattr(stock, 'unrealized_pnl', 0)
                })
        
        # Option positions
        for symbol, option in self.options.items():
            if option.qty != 0:
                positions.append({
                    'symbol': symbol,
                    'type': 'option',
                    'quantity': option.qty,
                    'avg_cost': option.avg_cost,
                    'market_value': getattr(option, 'market_value', 0),
                    'unrealized_pnl': getattr(option, 'unrealized_pnl', 0),
                    'strike': option.strike_price,
                    'expiry': option.expiration_date,
                    'option_type': option.type
                })
        
        return pd.DataFrame(positions)
    
    def close_expired_options(self, current_date: pd.Timestamp) -> List[Fill]:
        """
        Close expired options and return synthetic fills.
        
        Args:
            current_date: Current date to check against
            
        Returns:
            List of synthetic fills for expired options
        """
        expired_fills = []
        
        for symbol, option in list(self.options.items()):
            if option.qty != 0 and option.is_expired(current_date):
                # Calculate intrinsic value
                # This would need current underlying price in practice
                intrinsic_value = 0.0  # Simplified - assume expired worthless
                
                # Create synthetic fill to close position
                fill = Fill(
                    symbol=symbol,
                    qty=-option.qty,  # Close entire position
                    price=intrinsic_value,
                    timestamp=current_date,
                    commission=0.0
                )
                
                self.update_with_fill(fill)
                expired_fills.append(fill)
        
        return expired_fills