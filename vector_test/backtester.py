import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from vector_test.strategy import Strategy
from vector_test.portfolio import Portfolio

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 1000000):
        """
        Initialize the backtester
        
        Args:
            data: DataFrame with columns [date, code, open, high, low, close, volume, etc.]
            strategy: Strategy instance
            initial_capital: Initial capital for the backtest
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.dates = sorted(data['date'].unique())
        self.daily_returns = []
        self.equity_curve = []
        self.benchmark_returns = []
        
    def run(self):
        """Run the backtest"""
        # Calculate benchmark returns (average of all stocks)
        self.benchmark_returns = self.calculate_benchmark_returns()
        
        print(f"Running backtest for strategy: {self.strategy.name}")
        
        # For each date
        for i, date in enumerate(self.dates):
            # Get data for the current date
            current_data = self.data[self.data['date'] == date]
            
            # Skip if no data for current date
            if len(current_data) == 0:
                continue
            
            # Get buy and sell signals for this date
            buy_list, sell_list = self.strategy.generate_signals(self.data, date)

            
            # Extract buy and sell DataFrames
            buy_signals = current_data[current_data['code'].isin(buy_list)]
            sell_signals = current_data[current_data['code'].isin(sell_list)]
            
            # Update portfolio
            portfolio_value = self.portfolio.update(date, current_data, buy_signals, sell_signals)
            self.equity_curve.append(portfolio_value)
            
            # Calculate daily return
            if i > 0:
                daily_return = (portfolio_value / self.equity_curve[i-1]) - 1
                self.daily_returns.append(daily_return)
            else:
                self.daily_returns.append(0)
                
        # Convert lists to pandas Series
        self.daily_returns = pd.Series(self.daily_returns, index=self.dates)
        self.equity_curve = pd.Series(self.equity_curve, index=self.dates)
        
        # Calculate cumulative returns
        self.cumulative_returns = (1 + self.daily_returns).cumprod() - 1
        self.benchmark_cumulative_returns = (1 + self.benchmark_returns).cumprod() - 1
        
        # Print summary of positions at the end

        for code, shares in self.portfolio.positions.items():
            last_data = self.data[(self.data['date'] == self.dates[-1]) & (self.data['code'] == code)]
            if not last_data.empty:
                price = last_data['close'].values[0]
                value = shares * price


        
        return {
            'daily_returns': self.daily_returns,
            'cumulative_returns': self.cumulative_returns,
            'equity_curve': self.equity_curve,
            'benchmark_returns': self.benchmark_returns,
            'benchmark_cumulative_returns': self.benchmark_cumulative_returns
        }
    
    def calculate_benchmark_returns(self):
        """Calculate benchmark returns using average return of all stocks each day"""
        benchmark_returns = []
        
        # Calculate previous day's close for each stock
        self.data['prev_close'] = self.data.groupby('code')['open'].shift(1)
        
        # Calculate daily returns for each stock
        self.data['daily_return'] = (self.data['open'] / self.data['prev_close']) - 1
        
        # For each date, calculate average return across all stocks
        for date in self.dates:
            group = self.data[self.data['date'] == date]
            avg_return = group['daily_return'].mean()
            benchmark_returns.append(avg_return)
            
        return pd.Series(benchmark_returns, index=self.dates)
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        # Annual return
        annual_return = (1 + self.cumulative_returns.iloc[-1]) ** (252 / len(self.cumulative_returns)) - 1
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = np.sqrt(252) * self.daily_returns.mean() / self.daily_returns.std()
        
        # Maximum drawdown
        cum_returns = (1 + self.daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Benchmark metrics
        benchmark_annual_return = (1 + self.benchmark_cumulative_returns.iloc[-1]) ** (252 / len(self.benchmark_cumulative_returns)) - 1
        benchmark_sharpe_ratio = np.sqrt(252) * self.benchmark_returns.mean() / self.benchmark_returns.std()
        
        # Additional metrics
        win_days = (self.daily_returns > 0).sum()
        lose_days = (self.daily_returns < 0).sum()
        win_rate = win_days / (win_days + lose_days) if (win_days + lose_days) > 0 else 0
        
        return {
            'Annual Return': annual_return * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown * 100,
            'Win Rate': win_rate * 100,
            'Benchmark Annual Return': benchmark_annual_return * 100,
            'Benchmark Sharpe Ratio': benchmark_sharpe_ratio,
            'Alpha': (annual_return - benchmark_annual_return) * 100
        }
    
    def plot_results(self):
        """Plot backtest results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot cumulative returns
        ax1.plot(self.cumulative_returns.index, self.cumulative_returns * 100, label='Strategy')
        ax1.plot(self.benchmark_cumulative_returns.index, self.benchmark_cumulative_returns * 100, label='Benchmark')
        ax1.set_title('Cumulative Returns (%)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        cum_returns = (1 + self.daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = ((cum_returns / running_max) - 1) * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def get_transaction_history(self):
        """Get transaction history from portfolio"""
        return self.portfolio.get_transactions()