import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class Strategy(ABC):
    """Base strategy class that all strategies should inherit from"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, date) -> Tuple[List[str], List[str]]:
        """
        Generate buy/sell signals for a given date
        
        Args:
            data: DataFrame with OHLCV and other required data
            date: Current date
            
        Returns:
            Tuple of (buy_list, sell_list) containing stock codes to buy and sell
        """
        pass


class SignalBoundStrategy(Strategy):
    """Strategy based on signal crossing upper/lower bounds"""
    
    def __init__(self):
        super().__init__(name="Signal Bound Strategy")
    
    def generate_signals(self, data: pd.DataFrame, date) -> Tuple[List[str], List[str]]:
        """
        Generate buy/sell signals based on signal crossing upper/lower bounds
        
        Args:
            data: DataFrame with columns [date, code, open, high, low, close, volume, signal_prev, upperbound_prev, lowerbound_prev]
            date: Current date
            
        Returns:
            Tuple of (buy_list, sell_list) containing stock codes to buy and sell
        """
        # Get data for the current date
        current_data = data[data['date'] == date]
        
        # Skip if no data for current date or if signals are not available yet
        if len(current_data) == 0 or current_data['signal_prev'].isna().all():
            return [], []
        
        # Determine buy signals using the previous day's signals
        buy_signals = current_data[current_data['signal_prev'] > current_data['upperbound_prev']]
        buy_list = buy_signals['code'].tolist()
        
        # Determine sell signals using the previous day's signals
        sell_signals = current_data[current_data['signal_prev'] < current_data['lowerbound_prev']]
        sell_list = sell_signals['code'].tolist()
        
        return buy_list, sell_list

class DynamicPositionStrategy(Strategy):

    def __init__(self, position_signals_df: pd.DataFrame, benchmark_df: pd.DataFrame, num_stocks: int = 10, 
                 lookback_period: int = 20, volatility_threshold: float = 0.5, momentum_threshold: float = 0.1):

        super().__init__(name="Dynamic Position Strategy")
        self.position_signals_df = position_signals_df
        self.benchmark_df = benchmark_df
        self.num_stocks = num_stocks
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.momentum_threshold = momentum_threshold
        self.last_rebalance_date = None
        self.current_stocks = []
        
    def get_position_signal(self, date) -> float:
        """Get position signal for a given date"""
        if date not in self.position_signals_df.index:
            # Use last available signal if date not found
            if not self.position_signals_df.empty:
                return self.position_signals_df['position_signal'].iloc[-1]
            return 1.0  # Default to full position
            
        position_signal = self.position_signals_df.loc[date, 'position_signal']
        # Ensure position_signal is in [0,1] range
        position_signal = max(0, min(1, position_signal))
        return position_signal
    
    def check_seasonal_trading(self, date) -> str:
        """
        Check if date falls in seasonal trading rules
        
        Args:
            date: Current date
            
        Returns:
            String indicating action: "sell_all", "resume", or None
        """
        # Convert date to pandas Timestamp if needed
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Seasonal empty position periods
        is_april_end = date.month == 4 and date.day >= 15
        is_may_start = date.month == 5 and date.day >= 10
        is_august_end = date.month == 8 and date.day >= 15
        is_september_start = date.month == 9 and date.day >= 10
        is_jan_end = date.month == 1 and date.day >= 15
        is_feb_start = date.month == 2 and date.day >= 10
        is_oct_end = date.month == 10 and date.day >= 15
        is_nov_start = date.month == 11 and date.day >= 15
        
        # Rules for emptying positions
        if is_april_end or is_august_end or is_jan_end or is_oct_end:
            return "sell_all"
        
        # Rules for resuming positions
        elif (is_may_start and date.day >= 15) or \
             (is_september_start and date.day >= 15) or \
             (is_feb_start and date.day >= 15) or \
             (is_nov_start and date.day >= 15):
            return "resume"
            
        return None
    
    def calculate_momentum_signal(self, date) -> bool:
        """
        Calculate benchmark momentum signal
        
        Args:
            date: Current date
            
        Returns:
            Boolean indicating whether to use volatility factor (True) or just market cap (False)
        """
        # Get historical benchmark data
        historical_df = self.benchmark_df.loc[:date].tail(self.lookback_period + 1)
        
        if len(historical_df) >= self.lookback_period + 1:
            momentum = historical_df['close'].pct_change(periods=self.lookback_period).iloc[-1]
            # If momentum is high, use market cap only
            if momentum > self.momentum_threshold:
                return False
        
        # Default to using volatility factor
        return True
    
    def select_stocks(self, date, stock_universe: List[str], current_data: pd.DataFrame, lookback_data: pd.DataFrame) -> List[str]:
        """
        Select stocks based on market cap and optionally volatility
        
        Args:
            date: Current date
            stock_universe: List of available stock codes
            current_data: Current market data
            lookback_data: Historical data for volatility calculation
            
        Returns:
            List of selected stock codes
        """
        # Filter available stocks from the universe
        available_stocks = current_data[current_data['code'].isin(stock_universe)]
        
        # Check if we should use volatility factor
        use_volatility = self.calculate_momentum_signal(date)
        
        # If market is in strong uptrend, use market cap only
        if not use_volatility:
            print(f"{date}: 市场强势上涨，仅使用市值因子选股")
            smallest_cap_stocks = available_stocks.sort_values('market_cap').head(self.num_stocks)
            return smallest_cap_stocks['code'].tolist()
        
        print(f"{date}: 使用波动率和市值混合因子选股")
            
        # Calculate volatility for each stock
        volatilities = {}
        for stock in available_stocks['code'].unique():
            # Get historical data for this stock
            historical_prices = lookback_data[lookback_data['code'] == stock]['close'].values
            
            if len(historical_prices) < int(self.lookback_period * 0.8):
                continue
                
            # Calculate log returns and volatility
            log_returns = np.diff(np.log(historical_prices))
            if len(log_returns) > 0:
                volatility = np.std(log_returns)
                volatilities[stock] = volatility
        
        # If not enough stocks with volatility data, use market cap only
        if len(volatilities) < self.num_stocks:
            smallest_cap_stocks = available_stocks.sort_values('market_cap').head(self.num_stocks)
            return smallest_cap_stocks['code'].tolist()
            
        # Calculate volatility percentile threshold
        volatility_series = pd.Series(volatilities)
        volatility_threshold_value = volatility_series.quantile(self.volatility_threshold)
        
        # Select low volatility stocks
        low_volatility_stocks = volatility_series[volatility_series <= volatility_threshold_value].index.tolist()
        
        # Get market caps for these low volatility stocks
        low_vol_stocks_data = available_stocks[available_stocks['code'].isin(low_volatility_stocks)]
        
        if len(low_vol_stocks_data) < self.num_stocks:
            # Supplement with remaining stocks by market cap
            remaining_stocks = available_stocks[~available_stocks['code'].isin(low_volatility_stocks)]
            remaining_by_cap = remaining_stocks.sort_values('market_cap')
            needed = self.num_stocks - len(low_vol_stocks_data)
            supplement = remaining_by_cap.head(needed)['code'].tolist()
            selected_stocks = low_vol_stocks_data['code'].tolist() + supplement
        else:
            # Select smallest market cap among low volatility stocks
            selected_stocks = low_vol_stocks_data.sort_values('market_cap').head(self.num_stocks)['code'].tolist()
        
        return selected_stocks
    
    def generate_signals(self, data: pd.DataFrame, date, is_last_day_of_week) -> Tuple[List[str], List[str]]:
        """
        Generate buy/sell signals for the given date
        
        Args:
            data: DataFrame with OHLCV and other data
            date: Current date
            
        Returns:
            Tuple of (buy_list, sell_list) containing stock codes to buy and sell
        """
        # Check for seasonal trading rules
        seasonal_action = self.check_seasonal_trading(date)
        if seasonal_action == "sell_all":
            # If we need to sell everything, return empty buy list and all current stocks in sell list
            return [], self.current_stocks
        
        # Prepare stock universe
        current_data = data[data['date'] == date]
        if current_data.empty:
            return [], []
            

        position_signal_changed = False
        
        if not is_last_day_of_week and self.last_rebalance_date is not None:
            # Check if position signal changed significantly since last rebalance
            last_signal = self.get_position_signal(self.last_rebalance_date)
            current_signal = self.get_position_signal(date)
            position_signal_changed = abs(current_signal - last_signal) > 0.01
        
        # Generate signals based on conditions
        if seasonal_action == "resume" or is_last_day_of_week or position_signal_changed:
            # Get all available stocks from universe
            available_stocks = current_data['code'].unique().tolist()
            
            # Get historical data for lookback period
            lookback_start_date = date - pd.Timedelta(days=self.lookback_period * 2)  # Extra days for potential market holidays
            lookback_data = data[(data['date'] >= lookback_start_date) & (data['date'] <= date)]
            
            # Select new stocks
            selected_stocks = self.select_stocks(date, available_stocks, current_data, lookback_data)
            
            if is_last_day_of_week:
                self.last_rebalance_date = date
                
            # Update current stocks and return signals
            buy_list = selected_stocks
            sell_list = [stock for stock in self.current_stocks if stock not in selected_stocks]
            self.current_stocks = selected_stocks
            
            return buy_list, sell_list
            
        # No changes if no rebalance needed
        return [], []
        
    def is_rebalance_day(self, date) -> bool:
        """
        Check if the given date is a rebalance day (weekly)
        
        Args:
            date: Current date
            
        Returns:
            Boolean indicating if rebalance should happen
        """
        # Simple weekly rebalance check - every Friday
        if isinstance(date, pd.Timestamp):
            return date.weekday() == 4  # 4 is Friday
        return False

# class MomentumStrategy(Strategy):
#     """Simple momentum strategy based on price changes"""
    
#     def __init__(self, lookback_period: int = 20):
#         super().__init__(name="Momentum Strategy")
#         self.lookback_period = lookback_period
    
#     def generate_signals(self, data: pd.DataFrame, date) -> Tuple[List[str], List[str]]:
#         """
#         Generate buy/sell signals based on momentum
        
#         Args:
#             data: DataFrame with OHLCV data
#             date: Current date
            
#         Returns:
#             Tuple of (buy_list, sell_list) containing stock codes to buy and sell
#         """
#         # Get data up to the current date
#         historical_data = data[data['date'] <= date].copy()
        
#         # Calculate momentum (percentage change over lookback period)
#         historical_data['momentum'] = historical_data.groupby('code')['close'].pct_change(self.lookback_period)
        
#         # Get data for the current date
#         current_data = historical_data[historical_data['date'] == date].copy()
        
#         # Skip if not enough data
#         if len(current_data) == 0 or current_data['momentum'].isna().all():
#             return [], []
        
#         # Determine buy signals (top 20% momentum)
#         threshold = current_data['momentum'].quantile(0.8)
#         buy_signals = current_data[current_data['momentum'] > threshold]
#         buy_list = buy_signals['code'].tolist()
        
#         # Determine sell signals (bottom 20% momentum)
#         threshold = current_data['momentum'].quantile(0.2)
#         sell_signals = current_data[current_data['momentum'] < threshold]
#         sell_list = sell_signals['code'].tolist()
        
#         return buy_list, sell_list