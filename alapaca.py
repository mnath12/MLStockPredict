import trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Optional, List, Dict
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaIntradayDataFetcher:
    """
    Fetches historical intraday data from Alpaca API for volatility modeling.
    Handles rate limiting, data cleaning, and formats data for volatility calculations.
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Initialize Alpaca API connection.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key  
            base_url: Alpaca base URL (paper or live)
        """
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.market_tz = pytz.timezone('America/New_York')
        
    def get_intraday_bars(self, 
                         symbol: str,
                         start_date: datetime,
                         end_date: datetime,
                         timeframe: str = '1Min',
                         adjustment: str = 'split') -> pd.DataFrame:
        """
        Fetch intraday bars for a symbol over a date range.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour')
            adjustment: Price adjustment type ('raw', 'split', 'dividend', 'all')
            
        Returns:
            DataFrame with OHLCV data and timestamp index
        """
        try:
            # Convert timeframe to Alpaca format
            timeframe_map = {
                '1Min': tradeapi.TimeFrame.Minute,
                '5Min': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
                '15Min': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
                '1Hour': tradeapi.TimeFrame.Hour,
                '1Day': tradeapi.TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, tradeapi.TimeFrame.Minute)
            
            logger.info(f"Fetching {timeframe} bars for {symbol} from {start_date} to {end_date}")
            
            # Fetch bars with rate limiting
            bars = self.api.get_bars(
                symbol,
                tf,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                adjustment=adjustment,
                limit=10000  # Max per request
            ).df
            
            if bars.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean and format data
            bars = self._clean_bars_data(bars, symbol)
            
            logger.info(f"Retrieved {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_multiple_symbols_data(self,
                                 symbols: List[str],
                                 start_date: datetime,
                                 end_date: datetime,
                                 timeframe: str = '1Min',
                                 delay_between_requests: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for multiple symbols with rate limiting.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            timeframe: Bar timeframe
            delay_between_requests: Delay between API calls (seconds)
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data = {}
        
        for symbol in symbols:
            try:
                df = self.get_intraday_bars(symbol, start_date, end_date, timeframe)
                if not df.empty:
                    data[symbol] = df
                    
                # Rate limiting
                time.sleep(delay_between_requests)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
                
        return data
    
    def _clean_bars_data(self, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and format bars data for volatility calculations.
        
        Args:
            bars: Raw bars DataFrame from Alpaca
            symbol: Stock symbol for reference
            
        Returns:
            Cleaned DataFrame
        """
        # Remove any rows with missing data
        bars = bars.dropna()
        
        # Ensure timestamp is timezone-aware and convert to market timezone
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize('UTC')
        bars.index = bars.index.tz_convert(self.market_tz)
        
        # Add symbol column
        bars['symbol'] = symbol
        
        # Calculate returns for volatility computation
        bars['log_return'] = np.log(bars['close'] / bars['close'].shift(1))
        bars['simple_return'] = bars['close'].pct_change()
        
        # Filter out extreme outliers (likely data errors)
        # Remove returns > 50% in a single bar (adjust threshold as needed)
        bars = bars[abs(bars['log_return']) < 0.5]
        
        # Add trading session indicators
        bars['hour'] = bars.index.hour
        bars['minute'] = bars.index.minute
        bars['is_market_hours'] = (
            (bars['hour'] >= 9) & 
            ((bars['hour'] < 16) | ((bars['hour'] == 9) & (bars['minute'] >= 30)))
        )
        
        return bars
    
    def calculate_daily_realized_volatility(self, 
                                          bars_df: pd.DataFrame,
                                          method: str = 'close_to_close',
                                          annualize: bool = True) -> pd.DataFrame:
        """
        Calculate daily realized volatility from intraday data.
        
        Args:
            bars_df: DataFrame with intraday bars
            method: Method for RV calculation ('close_to_close', 'high_low', 'rogers_satchell')
            annualize: Whether to annualize the volatility
            
        Returns:
            DataFrame with daily realized volatility
        """
        if bars_df.empty:
            return pd.DataFrame()
        
        # Group by date
        daily_data = []
        
        for date, day_data in bars_df.groupby(bars_df.index.date):
            # Filter to market hours only
            market_data = day_data[day_data['is_market_hours']]
            
            if len(market_data) < 10:  # Need minimum bars for reliable RV
                continue
            
            if method == 'close_to_close':
                # Sum of squared log returns
                rv = (market_data['log_return'] ** 2).sum()
                
            elif method == 'high_low':
                # Parkinson estimator using high-low
                hl_data = market_data.groupby(market_data.index.date).agg({
                    'high': 'max',
                    'low': 'min',
                    'open': 'first',
                    'close': 'last'
                })
                rv = (np.log(hl_data['high'] / hl_data['low']) ** 2).iloc[0]
                
            elif method == 'rogers_satchell':
                # Rogers-Satchell estimator (drift-independent)
                rs_terms = (
                    np.log(market_data['high'] / market_data['close']) * 
                    np.log(market_data['high'] / market_data['open']) +
                    np.log(market_data['low'] / market_data['close']) * 
                    np.log(market_data['low'] / market_data['open'])
                )
                rv = rs_terms.sum()
            
            # Annualize if requested (252 trading days)
            if annualize:
                rv = rv * 252
            
            daily_data.append({
                'date': date,
                'realized_vol': np.sqrt(rv),
                'realized_var': rv,
                'n_bars': len(market_data)
            })
        
        return pd.DataFrame(daily_data).set_index('date')
    
    def get_data_for_volatility_modeling(self,
                                       symbol: str,
                                       lookback_days: int = 252,
                                       timeframe: str = '1Min') -> Dict:
        """
        Get complete dataset for volatility modeling (HAR-RV).
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days of historical data
            timeframe: Intraday timeframe
            
        Returns:
            Dictionary with intraday bars and daily realized volatility
        """
        # Calculate date range
        end_date = datetime.now(self.market_tz)
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))  # Buffer for weekends/holidays
        
        # Get intraday data
        bars = self.get_intraday_bars(symbol, start_date, end_date, timeframe)
        
        if bars.empty:
            return {'bars': pd.DataFrame(), 'daily_rv': pd.DataFrame()}
        
        # Calculate daily realized volatility
        daily_rv = self.calculate_daily_realized_volatility(bars)
        
        # Keep only the requested number of days
        daily_rv = daily_rv.tail(lookback_days)
        
        logger.info(f"Prepared {len(daily_rv)} days of RV data for {symbol}")
        
        return {
            'bars': bars,
            'daily_rv': daily_rv,
            'symbol': symbol,
            'timeframe': timeframe,
            'data_range': (daily_rv.index.min(), daily_rv.index.max()) if not daily_rv.empty else None
        }


# Example usage and testing
if __name__ == "__main__":
    # Example usage - replace with your actual API keys
    API_KEY = "6a3ea255-7f31-455c-8e41-6e444b1c4fc6"
    SECRET_KEY = "ig5CGnl3c1jXEepU6VK5DPXgsV5WSOBYrIJGk70T"
    
    # Initialize fetcher
    fetcher = AlpacaIntradayDataFetcher(API_KEY, SECRET_KEY)
    
    # Example 1: Get recent SPY data for volatility modeling
    spy_data = fetcher.get_data_for_volatility_modeling('SPY', lookback_days=60)
    
    if not spy_data['daily_rv'].empty:
        print(f"SPY Realized Volatility Summary:")
        print(spy_data['daily_rv']['realized_vol'].describe())
        print(f"\nLatest RV: {spy_data['daily_rv']['realized_vol'].iloc[-1]:.4f}")
    
    # Example 2: Get data for multiple symbols
    symbols = ['SPY', 'QQQ', 'IWM']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    multi_data = fetcher.get_multiple_symbols_data(
        symbols, start_date, end_date, timeframe='5Min'
    )
    
    for symbol, df in multi_data.items():
        print(f"\n{symbol}: {len(df)} bars retrieved")
        if not df.empty:
            print(f"Date range: {df.index.min()} to {df.index.max()}")