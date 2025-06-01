import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class Portfolio:
    def __init__(self, initial_capital: float = 1000000):
        """
        Initialize the portfolio
        
        Args:
            initial_capital: Initial capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # code -> shares
        self.total_value_history = []
        self.cash_history = []
        self.dates = []
        self.transactions = []  # List to store transaction history
        
    def update(self, date, current_data, buy_signals, sell_signals):
        """
        Update the portfolio for a given date
        
        Args:
            date: Current date
            current_data: Current market data
            buy_signals: DataFrame of buy signals
            sell_signals: DataFrame of sell signals
        
        Returns:
            Updated portfolio value
        """
        # Add date to history
        self.dates.append(date)
        
        # Log current positions before trading
        positions_before = self.positions.copy()
        
        # First sell
        for _, row in sell_signals.iterrows():
            code = row['code']
            if code in self.positions and self.positions[code] > 0:
                # Calculate sale proceeds using today's open price
                price = row['open']
                shares = self.positions[code]
                proceeds = shares * price
                
                # Add to cash
                self.cash += proceeds
                
                # Log transaction
                self.transactions.append({
                    'date': date,
                    'code': code,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': proceeds
                })
                
                # Remove from positions
                del self.positions[code]
        
        # Then buy with equal weights
        if not buy_signals.empty:
            # Calculate how much to allocate to each stock
            allocation_per_stock = self.cash / len(buy_signals)
            
            for _, row in buy_signals.iterrows():
                code = row['code']
                price = row['open']
                
                # Calculate number of shares to buy
                shares = allocation_per_stock / price
                
                # Remove from cash
                self.cash -= allocation_per_stock
                
                # Add to positions
                self.positions[code] = self.positions.get(code, 0) + shares
                
                # Log transaction
                self.transactions.append({
                    'date': date,
                    'code': code,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': allocation_per_stock
                })
        
        # Calculate portfolio value
        portfolio_value = self.calculate_value(current_data)
        
        # Update history
        self.total_value_history.append(portfolio_value)
        self.cash_history.append(self.cash)
        
        # Log positions that changed
        positions_after = self.positions.copy()
        
        changes = {}
        for code in set(list(positions_before.keys()) + list(positions_after.keys())):
            before = positions_before.get(code, 0)
            after = positions_after.get(code, 0)
            if before != after:
                changes[code] = (before, after)
        
        
        return portfolio_value
    
    def calculate_value(self, current_data):
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        
        # For each position, add the current value
        for code, shares in self.positions.items():
            # Find the current price (use close price)
            current_price = current_data[current_data['code'] == code]['open'].values
            if len(current_price) > 0:
                portfolio_value += shares * current_price[0]
        
        return portfolio_value
    
    def get_returns(self):
        """Calculate daily returns"""
        values = pd.Series(self.total_value_history, index=self.dates)
        daily_returns = values.pct_change().fillna(0)
        return daily_returns
    
    def get_holdings(self):
        """Get current holdings"""
        return self.positions
    
    def get_transactions(self):
        """Get transaction history"""
        return pd.DataFrame(self.transactions)