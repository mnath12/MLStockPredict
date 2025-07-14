"""
Fixed LSTM Backtesting Framework
================================

A fully functional backtesting framework with optimized LSTM model
that runs automatically without user input issues.
"""

import os
# Fix Intel MKL warnings and TensorFlow issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Literal, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    print("✅ TensorFlow loaded successfully")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("📦 Please install TensorFlow: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

# =============================================================================
# Core Data Structures
# =============================================================================

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"

class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Signal:
    symbol: str
    datetime: datetime
    signal_type: SignalType
    strength: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Order:
    symbol: str
    datetime: datetime
    direction: OrderDirection
    quantity: int
    order_id: str = None
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.datetime.isoformat()}_{id(self)}"

@dataclass
class Fill:
    order_id: str
    symbol: str
    datetime: datetime
    direction: OrderDirection
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def total_cost(self) -> float:
        gross_cost = self.quantity * self.price
        if self.direction == OrderDirection.BUY:
            return gross_cost + self.commission + (self.slippage * self.quantity)
        else:
            return gross_cost - self.commission - (self.slippage * self.quantity)

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

# =============================================================================
# Data Handler
# =============================================================================

class YahooDataHandler:
    def __init__(self, symbols: List[str], start: datetime, end: datetime):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.current_index = 0
        self.data = {}
        self.all_dates = []
        self._fetch_data()
    
    def _fetch_data(self):
        print("📊 Fetching data from Yahoo Finance...")
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start, end=self.end)
                if df.empty:
                    raise ValueError(f"No data found for symbol {symbol}")
                
                df.columns = df.columns.str.lower()
                df.index.name = 'datetime'
                self.data[symbol] = df
                print(f"✅ Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                print(f"❌ Error loading data for {symbol}: {e}")
                raise
        
        all_dates_set = set()
        for df in self.data.values():
            all_dates_set.update(df.index)
        self.all_dates = sorted(list(all_dates_set))
        print(f"📅 Total trading days: {len(self.all_dates)}")
    
    def update_bars(self) -> None:
        if self.has_next():
            self.current_index += 1
    
    def get_latest_bars(self, symbol: str, N: int = 1) -> pd.DataFrame:
        if symbol not in self.data:
            raise ValueError(f"Symbol {symbol} not found in data")
        
        current_date = self.get_current_datetime()
        symbol_data = self.data[symbol]
        available_data = symbol_data[symbol_data.index <= current_date]
        
        if len(available_data) == 0:
            return pd.DataFrame()
        
        return available_data.tail(N)
    
    def get_current_datetime(self) -> datetime:
        if self.current_index < len(self.all_dates):
            return self.all_dates[self.current_index]
        return self.all_dates[-1]
    
    def has_next(self) -> bool:
        return self.current_index < len(self.all_dates) - 1

# =============================================================================
# Strategy Base Class
# =============================================================================

class Strategy(ABC):
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.signals = []
    
    @abstractmethod
    def on_bar(self, symbol: str, datetime: datetime, bar: pd.Series) -> None:
        pass
    
    def generate_signals(self) -> List[Signal]:
        signals = self.signals.copy()
        self.signals.clear()
        return signals
    
    def add_signal(self, signal: Signal):
        self.signals.append(signal)

# =============================================================================
# LSTM Strategy (Only if TensorFlow is available)
# =============================================================================

if TENSORFLOW_AVAILABLE:
    class OptimizedLSTMStrategy(Strategy):
        def __init__(self, symbols: List[str], 
                     lookback_window: int = 60,
                     prediction_threshold: float = 0.015,
                     retrain_frequency: int = 63):
            super().__init__(symbols)
            self.lookback_window = lookback_window
            self.prediction_threshold = prediction_threshold
            self.retrain_frequency = retrain_frequency
            
            # Data storage
            self.price_history = {symbol: [] for symbol in symbols}
            self.volume_history = {symbol: [] for symbol in symbols}
            self.returns_history = {symbol: [] for symbol in symbols}
            
            # Models and scalers
            self.models = {}
            self.scalers = {}
            self.bars_processed = 0
            
            # For analysis
            self.prediction_history = {symbol: [] for symbol in symbols}
            self.actual_prices = {symbol: [] for symbol in symbols}
            self.prediction_dates = {symbol: [] for symbol in symbols}
        
        def _calculate_technical_features(self, prices: np.array) -> np.array:
            """Calculate technical indicators"""
            features = []
            for i in range(len(prices)):
                row = [prices[i]]
                
                # Moving averages
                for window in [5, 10, 20]:
                    if i >= window - 1:
                        ma = np.mean(prices[max(0, i-window+1):i+1])
                        row.append(ma)
                        row.append(prices[i] / ma - 1 if ma != 0 else 0)
                    else:
                        row.extend([prices[i], 0])
                
                # Price momentum
                for lag in [1, 3, 5]:
                    if i >= lag:
                        momentum = (prices[i] - prices[i-lag]) / prices[i-lag] if prices[i-lag] != 0 else 0
                        row.append(momentum)
                    else:
                        row.append(0)
                
                # Volatility
                if i >= 9:
                    vol = np.std(prices[max(0, i-9):i+1])
                    row.append(vol)
                else:
                    row.append(0)
                
                features.append(row)
            
            return np.array(features)
        
        def _prepare_features(self, symbol: str) -> np.array:
            """Prepare feature matrix"""
            if len(self.price_history[symbol]) < 30:
                return None
            
            prices = np.array(self.price_history[symbol])
            volumes = np.array(self.volume_history[symbol])
            returns = np.array(self.returns_history[symbol])
            
            # Technical features
            tech_features = self._calculate_technical_features(prices)
            
            # Combine with volume and returns
            features = []
            for i in range(len(tech_features)):
                row = list(tech_features[i])
                if i < len(volumes):
                    row.append(volumes[i])
                else:
                    row.append(0)
                if i < len(returns):
                    row.append(returns[i])
                else:
                    row.append(0)
                features.append(row)
            
            return np.array(features)
        
        def _create_sequences(self, data: np.array, target: np.array) -> Tuple[np.array, np.array]:
            """Create LSTM sequences"""
            X, y = [], []
            for i in range(self.lookback_window, len(data)):
                X.append(data[i-self.lookback_window:i])
                y.append(target[i])
            return np.array(X), np.array(y)
        
        def _build_model(self, input_shape: Tuple) -> Sequential:
            """Build optimized LSTM model"""
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.2),
                
                LSTM(32, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),
                
                LSTM(16),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(8, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
            return model
        
        def _train_model(self, symbol: str) -> None:
            """Train LSTM model"""
            if len(self.price_history[symbol]) < 200:
                return
            
            print(f"🚀 Training LSTM model for {symbol}...")
            
            try:
                features = self._prepare_features(symbol)
                if features is None:
                    return
                
                prices = np.array(self.price_history[symbol])
                
                # Scale features
                if symbol not in self.scalers:
                    self.scalers[symbol] = MinMaxScaler()
                
                scaled_features = self.scalers[symbol].fit_transform(features)
                
                # Create sequences
                X, y = self._create_sequences(scaled_features, prices[self.lookback_window:])
                
                if len(X) < 50:
                    return
                
                # Train/validation split
                split_idx = int(0.8 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Build and train model
                self.models[symbol] = self._build_model((X.shape[1], X.shape[2]))
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0)
                ]
                
                history = self.models[symbol].fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=16,
                    callbacks=callbacks,
                    verbose=0
                )
                
                val_loss = min(history.history['val_loss'])
                print(f"✅ Model trained for {symbol} - Val Loss: {val_loss:.4f}")
                
            except Exception as e:
                print(f"❌ Error training model for {symbol}: {e}")
        
        def _make_prediction(self, symbol: str) -> Optional[float]:
            """Make price prediction"""
            if symbol not in self.models or len(self.price_history[symbol]) < self.lookback_window:
                return None
            
            try:
                features = self._prepare_features(symbol)
                if features is None:
                    return None
                
                scaled_features = self.scalers[symbol].transform(features)
                last_sequence = scaled_features[-self.lookback_window:].reshape(1, self.lookback_window, -1)
                
                prediction = self.models[symbol].predict(last_sequence, verbose=0)[0][0]
                return prediction
            except Exception as e:
                print(f"❌ Error making prediction for {symbol}: {e}")
                return None
        
        def on_bar(self, symbol: str, datetime: datetime, bar: pd.Series) -> None:
            """Process new bar and generate signals"""
            close_price = bar['close']
            volume = bar.get('volume', 0)
            
            self.price_history[symbol].append(close_price)
            self.volume_history[symbol].append(volume)
            
            # Calculate returns
            if len(self.price_history[symbol]) > 1:
                ret = (close_price - self.price_history[symbol][-2]) / self.price_history[symbol][-2]
                self.returns_history[symbol].append(ret)
            else:
                self.returns_history[symbol].append(0.0)
            
            # Limit history
            max_history = 1000
            for hist in [self.price_history, self.volume_history, self.returns_history]:
                if len(hist[symbol]) > max_history:
                    hist[symbol] = hist[symbol][-max_history:]
            
            self.bars_processed += 1
            
            # Retrain model periodically
            if self.bars_processed % self.retrain_frequency == 0 or symbol not in self.models:
                self._train_model(symbol)
            
            # Make prediction
            if symbol in self.models:
                prediction = self._make_prediction(symbol)
                
                if prediction is not None:
                    # Store for analysis
                    self.prediction_history[symbol].append(prediction)
                    self.actual_prices[symbol].append(close_price)
                    self.prediction_dates[symbol].append(datetime)
                    
                    # Calculate predicted return
                    predicted_return = (prediction - close_price) / close_price
                    
                    # Generate signals
                    if predicted_return > self.prediction_threshold:
                        signal = Signal(
                            symbol=symbol,
                            datetime=datetime,
                            signal_type=SignalType.LONG,
                            strength=min(1.0, abs(predicted_return) / self.prediction_threshold),
                            metadata={'predicted_price': prediction, 'predicted_return': predicted_return}
                        )
                        self.add_signal(signal)
                    
                    elif predicted_return < -self.prediction_threshold:
                        signal = Signal(
                            symbol=symbol,
                            datetime=datetime,
                            signal_type=SignalType.EXIT,
                            metadata={'predicted_price': prediction, 'predicted_return': predicted_return}
                        )
                        self.add_signal(signal)

# =============================================================================
# Fallback Strategy (if TensorFlow not available)
# =============================================================================

class SimpleMovingAverageStrategy(Strategy):
    """Simple moving average strategy as fallback"""
    
    def __init__(self, symbols: List[str], short_window: int = 10, long_window: int = 30):
        super().__init__(symbols)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = {symbol: [] for symbol in symbols}
        
        # For analysis compatibility
        self.prediction_history = {symbol: [] for symbol in symbols}
        self.actual_prices = {symbol: [] for symbol in symbols}
        self.prediction_dates = {symbol: [] for symbol in symbols}
    
    def on_bar(self, symbol: str, datetime: datetime, bar: pd.Series) -> None:
        """Process new bar and generate signals"""
        price = bar['close']
        self.price_history[symbol].append(price)
        
        # Store for analysis
        self.actual_prices[symbol].append(price)
        self.prediction_dates[symbol].append(datetime)
        
        if len(self.price_history[symbol]) >= self.long_window:
            recent_prices = self.price_history[symbol][-self.long_window:]
            short_ma = np.mean(recent_prices[-self.short_window:])
            long_ma = np.mean(recent_prices[-self.long_window:])
            
            # Store "prediction" (just the short MA for visualization)
            self.prediction_history[symbol].append(short_ma)
            
            # Check for crossover
            if len(self.price_history[symbol]) > self.long_window:
                prev_short = np.mean(self.price_history[symbol][-(self.short_window+1):-1])
                prev_long = np.mean(self.price_history[symbol][-(self.long_window+1):-1])
                
                # Golden cross
                if short_ma > long_ma and prev_short <= prev_long:
                    signal = Signal(symbol, datetime, SignalType.LONG, strength=0.8)
                    self.add_signal(signal)
                
                # Death cross
                elif short_ma < long_ma and prev_short >= prev_long:
                    signal = Signal(symbol, datetime, SignalType.EXIT)
                    self.add_signal(signal)
        else:
            # Add dummy prediction for early bars
            self.prediction_history[symbol].append(price)
        
        # Keep history manageable
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]

# =============================================================================
# Position Sizer
# =============================================================================

class AdvancedVolatilitySizer:
    def __init__(self, target_volatility: float = 0.15, 
                 max_risk_per_trade: float = 0.02,
                 stop_loss_pct: float = 0.05):
        self.target_volatility = target_volatility
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_pct = stop_loss_pct
    
    def size_order(self, signal: Signal, portfolio_value: float, 
                   current_price: float, current_volatility: float = None) -> Order:
        """Size order with volatility and risk controls"""
        
        # Volatility-based sizing
        if current_volatility and current_volatility > 0:
            vol_factor = self.target_volatility / current_volatility
            vol_allocation = portfolio_value * signal.strength * vol_factor * 0.1
        else:
            vol_allocation = 0.02 * portfolio_value * signal.strength
        
        # Risk-based sizing
        max_risk = self.max_risk_per_trade * portfolio_value
        risk_per_share = current_price * self.stop_loss_pct
        risk_allocation = max_risk / risk_per_share * current_price if risk_per_share > 0 else vol_allocation
        
        # Use conservative approach
        allocation = min(vol_allocation, risk_allocation, 0.2 * portfolio_value)
        quantity = max(1, int(allocation / current_price))
        
        if quantity <= 0:
            return None
        
        direction = OrderDirection.BUY if signal.signal_type == SignalType.LONG else OrderDirection.SELL
        
        return Order(
            symbol=signal.symbol,
            datetime=signal.datetime,
            direction=direction,
            quantity=quantity
        )

# =============================================================================
# Execution Handler
# =============================================================================

class SimulatedExecutionHandler:
    def __init__(self, commission: float = 1.0, slippage_bps: float = 2.0):
        self.commission = commission
        self.slippage_bps = slippage_bps
    
    def execute_order(self, order: Order, current_price: float) -> Optional[Fill]:
        if order is None or order.quantity == 0:
            return None
        
        slippage_factor = self.slippage_bps / 10000
        if order.direction == OrderDirection.BUY:
            fill_price = current_price * (1 + slippage_factor)
        else:
            fill_price = current_price * (1 - slippage_factor)
        
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            datetime=order.datetime,
            direction=order.direction,
            quantity=order.quantity,
            price=fill_price,
            commission=self.commission
        )

# =============================================================================
# Portfolio
# =============================================================================

class Portfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []
        self.equity_curve: List[Dict] = []
    
    def on_fill(self, fill: Fill) -> None:
        if fill is None:
            return
        
        self.fills.append(fill)
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0.0)
        
        position = self.positions[symbol]
        
        if fill.direction == OrderDirection.BUY:
            total_quantity = position.quantity + fill.quantity
            if total_quantity != 0:
                total_cost = (position.quantity * position.avg_price) + fill.total_cost
                position.avg_price = total_cost / total_quantity
            position.quantity += fill.quantity
            self.cash -= fill.total_cost
        
        else:  # SELL
            realized_pnl = fill.quantity * (fill.price - position.avg_price) - fill.commission
            position.realized_pnl += realized_pnl
            position.quantity -= fill.quantity
            self.cash += fill.total_cost
        
        if position.quantity == 0:
            position.avg_price = 0.0
    
    def update_market_value(self, data_handler: YahooDataHandler) -> None:
        current_datetime = data_handler.get_current_datetime()
        total_market_value = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                try:
                    latest_bar = data_handler.get_latest_bars(symbol, 1)
                    if not latest_bar.empty:
                        current_price = latest_bar.iloc[-1]['close']
                        market_value = position.quantity * current_price
                        position.unrealized_pnl = market_value - (position.quantity * position.avg_price)
                        total_market_value += market_value
                except:
                    total_market_value += position.quantity * position.avg_price
        
        self.equity_curve.append({
            'datetime': current_datetime,
            'equity': total_market_value,
            'cash': self.cash,
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values())
        })
    
    @property
    def equity(self) -> float:
        if not self.equity_curve:
            return self.initial_capital
        return self.equity_curve[-1]['equity']
    
    def get_position(self, symbol: str) -> Position:
        return self.positions.get(symbol, Position(symbol=symbol, quantity=0, avg_price=0.0))

# =============================================================================
# Performance Analysis
# =============================================================================

class PerformanceAnalyzer:
    @staticmethod
    def calculate_performance(equity_curve: List[Dict]) -> Dict[str, float]:
        if not equity_curve:
            return {}
        
        df = pd.DataFrame(equity_curve)
        df.set_index('datetime', inplace=True)
        df['returns'] = df['equity'].pct_change().dropna()
        
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        num_days = (df.index[-1] - df.index[0]).days
        annualized_return = (1 + total_return) ** (365.25 / num_days) - 1 if num_days > 0 else 0
        
        volatility = df['returns'].std() * np.sqrt(252) if len(df['returns']) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        running_max = df['equity'].expanding().max()
        drawdown = (df['equity'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        winning_trades = (df['returns'] > 0).sum()
        total_trades = len(df['returns'].dropna())
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_equity': df['equity'].iloc[-1],
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }

# =============================================================================
# Backtest Runner
# =============================================================================

class BacktestRunner:
    def __init__(self, data_handler: YahooDataHandler, strategy: Strategy,
                 position_sizer: AdvancedVolatilitySizer, 
                 execution_handler: SimulatedExecutionHandler,
                 portfolio: Portfolio):
        self.data_handler = data_handler
        self.strategy = strategy
        self.position_sizer = position_sizer
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.performance_analyzer = PerformanceAnalyzer()
    
    def calculate_rolling_volatility(self, prices: np.array, window: int = 20) -> float:
        if len(prices) < window:
            return None
        returns = np.diff(np.log(prices[-window:]))
        return np.std(returns) * np.sqrt(252)
    
    def run_backtest(self) -> Dict[str, Any]:
        print("🚀 Starting backtest...")
        bar_count = 0
        
        while self.data_handler.has_next():
            self.data_handler.update_bars()
            current_datetime = self.data_handler.get_current_datetime()
            
            # Process each symbol
            for symbol in self.data_handler.symbols:
                latest_bars = self.data_handler.get_latest_bars(symbol, 1)
                if not latest_bars.empty:
                    bar = latest_bars.iloc[-1]
                    self.strategy.on_bar(symbol, current_datetime, bar)
            
            # Process signals
            signals = self.strategy.generate_signals()
            for signal in signals:
                latest_bars = self.data_handler.get_latest_bars(signal.symbol, 21)
                if not latest_bars.empty:
                    current_price = latest_bars.iloc[-1]['close']
                    
                    # Calculate volatility
                    prices = latest_bars['close'].values
                    current_volatility = self.calculate_rolling_volatility(prices)
                    
                    if signal.signal_type == SignalType.EXIT:
                        position = self.portfolio.get_position(signal.symbol)
                        if position.quantity != 0:
                            order = Order(
                                symbol=signal.symbol,
                                datetime=signal.datetime,
                                direction=OrderDirection.SELL if position.quantity > 0 else OrderDirection.BUY,
                                quantity=abs(position.quantity)
                            )
                            fill = self.execution_handler.execute_order(order, current_price)
                            self.portfolio.on_fill(fill)
                    else:
                        order = self.position_sizer.size_order(
                            signal, self.portfolio.equity, current_price, current_volatility
                        )
                        if order:
                            fill = self.execution_handler.execute_order(order, current_price)
                            self.portfolio.on_fill(fill)
            
            # Update portfolio
            self.portfolio.update_market_value(self.data_handler)
            
            bar_count += 1
            if bar_count % 250 == 0:
                print(f"📊 Processed {bar_count} bars, Equity: ${self.portfolio.equity:,.2f}")
        
        performance = self.performance_analyzer.calculate_performance(self.portfolio.equity_curve)
        print(f"✅ Backtest completed. Final equity: ${performance.get('final_equity', 0):,.2f}")
        
        return {
            'performance': performance,
            'equity_curve': self.portfolio.equity_curve,
            'positions': self.portfolio.positions,
            'fills': self.portfolio.fills
        }

# =============================================================================
# Plotting and Analysis
# =============================================================================

def plot_results(results: Dict, strategy):
    """Plot comprehensive backtest results"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Equity Curve
        equity_df = pd.DataFrame(results['equity_curve'])
        if not equity_df.empty:
            equity_df.set_index('datetime', inplace=True)
            ax1.plot(equity_df.index, equity_df['equity'], 'b-', linewidth=2, label='Portfolio Value')
            ax1.axhline(y=equity_df['equity'].iloc[0], color='r', linestyle='--', alpha=0.7, label='Initial Capital')
            ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Returns Distribution
        if not equity_df.empty:
            returns = equity_df['equity'].pct_change().dropna()
            if len(returns) > 0:
                ax2.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Daily Return')
                ax2.set_ylabel('Frequency')
                ax2.axvline(x=returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean():.4f}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Predictions vs Actual
        if hasattr(strategy, 'symbols') and strategy.symbols and strategy.prediction_history[strategy.symbols[0]]:
            symbol = strategy.symbols[0]
            if len(strategy.prediction_history[symbol]) > 0:
                # Get last 100 points for clarity
                n_points = min(100, len(strategy.prediction_history[symbol]))
                dates = strategy.prediction_dates[symbol][-n_points:]
                actual = strategy.actual_prices[symbol][-n_points:]
                predicted = strategy.prediction_history[symbol][-n_points:]
                
                ax3.plot(dates, actual, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
                ax3.plot(dates, predicted, 'r--', linewidth=2, label='Prediction/MA', alpha=0.8)
                ax3.set_title(f'Predictions vs Actual - {symbol}', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Price ($)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Drawdown
        if not equity_df.empty:
            running_max = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - running_max) / running_max * 100
            
            ax4.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
            ax4.plot(equity_df.index, drawdown, 'r-', linewidth=1)
            ax4.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Drawdown (%)')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error plotting results: {e}")
        print("📊 Backtest completed but plots could not be generated")

def print_detailed_results(results: Dict, initial_capital: float):
    """Print comprehensive results"""
    performance = results['performance']
    
    print("\n" + "="*60)
    print("           BACKTESTING RESULTS")
    print("="*60)
    
    print(f"\n📊 ACCOUNT SUMMARY:")
    print(f"   Initial Capital:      ${initial_capital:,.2f}")
    print(f"   Final Equity:         ${performance['final_equity']:,.2f}")
    print(f"   Total P&L:            ${performance['final_equity'] - initial_capital:,.2f}")
    print(f"   Total Return:         {performance['total_return']:.2%}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Annualized Return:    {performance['annualized_return']:.2%}")
    print(f"   Volatility:           {performance['volatility']:.2%}")
    print(f"   Sharpe Ratio:         {performance['sharpe_ratio']:.3f}")
    print(f"   Calmar Ratio:         {performance['calmar_ratio']:.3f}")
    print(f"   Maximum Drawdown:     {performance['max_drawdown']:.2%}")
    
    print(f"\n🔄 TRADING ACTIVITY:")
    print(f"   Total Trades:         {performance['total_trades']:,}")
    print(f"   Win Rate:             {performance['win_rate']:.2%}")
    print("="*60)

# =============================================================================
# Main Execution Functions
# =============================================================================

def run_lstm_backtest():
    """Run LSTM strategy backtest"""
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Running Moving Average strategy instead.")
        return run_simple_backtest()
    
    # Configuration
    symbols = ['AAPL']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    initial_capital = 100000.0
    
    print(f"🚀 Starting LSTM Backtest for {symbols[0]}")
    print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print("-" * 60)
    
    try:
        # Initialize components
        data_handler = YahooDataHandler(symbols, start_date, end_date)
        
        strategy = OptimizedLSTMStrategy(
            symbols=symbols,
            lookback_window=60,
            prediction_threshold=0.015,
            retrain_frequency=63
        )
        
        position_sizer = AdvancedVolatilitySizer(
            target_volatility=0.15,
            max_risk_per_trade=0.02,
            stop_loss_pct=0.05
        )
        
        execution_handler = SimulatedExecutionHandler(commission=1.0, slippage_bps=2.0)
        portfolio = Portfolio(initial_capital)
        
        # Run backtest
        runner = BacktestRunner(data_handler, strategy, position_sizer, execution_handler, portfolio)
        results = runner.run_backtest()
        
        # Print results
        print_detailed_results(results, initial_capital)
        
        # Plot results
        plot_results(results, strategy)
        
        return results, strategy
        
    except Exception as e:
        print(f"❌ Error during LSTM backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def run_simple_backtest():
    """Run simple moving average strategy backtest"""
    symbols = ['AAPL']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 1, 1)
    initial_capital = 100000.0
    
    print(f"🚀 Starting Simple MA Backtest for {symbols[0]}")
    print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"💰 Initial Capital: ${initial_capital:,.2f}")
    print("-" * 60)
    
    try:
        # Initialize components
        data_handler = YahooDataHandler(symbols, start_date, end_date)
        
        strategy = SimpleMovingAverageStrategy(symbols, short_window=10, long_window=30)
        position_sizer = AdvancedVolatilitySizer()
        execution_handler = SimulatedExecutionHandler(commission=1.0, slippage_bps=2.0)
        portfolio = Portfolio(initial_capital)
        
        # Run backtest
        runner = BacktestRunner(data_handler, strategy, position_sizer, execution_handler, portfolio)
        results = runner.run_backtest()
        
        # Print results
        print_detailed_results(results, initial_capital)
        
        # Plot results
        plot_results(results, strategy)
        
        return results, strategy
        
    except Exception as e:
        print(f"❌ Error during simple backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function that runs automatically"""
    print("🤖 Advanced Backtesting Framework")
    print("="*50)
    
    # Check TensorFlow availability and run appropriate strategy
    if TENSORFLOW_AVAILABLE:
        print("✅ TensorFlow available - Running LSTM strategy")
        results, strategy = run_lstm_backtest()
    else:
        print("⚠️  TensorFlow not available - Running Moving Average strategy")
        results, strategy = run_simple_backtest()
    
    if results:
        print("\n🎯 Backtest completed successfully!")
        print("📈 Check the plots above for detailed analysis")
        
        # Additional summary
        performance = results['performance']
        print(f"\n💡 QUICK SUMMARY:")
        print(f"   🎯 Final Return: {performance['total_return']:.2%}")
        print(f"   📊 Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   📉 Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   🎲 Win Rate: {performance['win_rate']:.1%}")
    else:
        print("\n❌ Backtest failed. Please check the error messages above.")

if __name__ == "__main__":
    main()