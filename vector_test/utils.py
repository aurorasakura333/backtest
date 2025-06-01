import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

import pandas as pd
import numpy as np

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for backtesting
    
    Args:
        data: Raw DataFrame
        
    Returns:
        Processed DataFrame ready for backtesting
    """
    # Make a copy
    df = data.copy()
    
    # Convert date to datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and code
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    
    # Calculate returns
    df['returns'] = df.groupby('code')['open'].pct_change()
    
    return df

def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
    """
    Calculate performance metrics
    
    Args:
        returns: Daily returns
        benchmark_returns: Benchmark daily returns
        
    Returns:
        Dictionary of performance metrics
    """
    # Annualization factor
    annualization_factor = 252
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod() - 1
    
    # Calculate annual return
    annual_return = (1 + cum_returns.iloc[-1]) ** (annualization_factor / len(returns)) - 1
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(annualization_factor)
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Calculate maximum drawdown
    running_max = (1 + returns).cumprod().cummax()
    drawdown = (1 + returns).cumprod() / running_max - 1
    max_drawdown = drawdown.min()
    
    metrics = {
        'Annual Return': annual_return * 100,
        'Volatility': volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown * 100,
        'Total Return': cum_returns.iloc[-1] * 100
    }
    
    # Add benchmark metrics if provided
    if benchmark_returns is not None:
        bench_cum_returns = (1 + benchmark_returns).cumprod() - 1
        bench_annual_return = (1 + bench_cum_returns.iloc[-1]) ** (annualization_factor / len(benchmark_returns)) - 1
        bench_volatility = benchmark_returns.std() * np.sqrt(annualization_factor)
        bench_sharpe_ratio = bench_annual_return / bench_volatility if bench_volatility != 0 else 0
        
        metrics.update({
            'Benchmark Annual Return': bench_annual_return * 100,
            'Benchmark Volatility': bench_volatility * 100,
            'Benchmark Sharpe Ratio': bench_sharpe_ratio,
            'Alpha': (annual_return - bench_annual_return) * 100,
            'Beta': np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) != 0 else 1
        })
    
    return metrics

def plot_cumulative_returns(returns: pd.Series, benchmark_returns: pd.Series = None, title: str = 'Cumulative Returns'):
    """
    Plot cumulative returns
    
    Args:
        returns: Daily returns
        benchmark_returns: Benchmark daily returns
        title: Plot title
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod() - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot strategy returns
    ax.plot(cum_returns.index, cum_returns * 100, label='Strategy')
    
    # Plot benchmark returns if provided
    if benchmark_returns is not None:
        bench_cum_returns = (1 + benchmark_returns).cumprod() - 1
        ax.plot(bench_cum_returns.index, bench_cum_returns * 100, label='Benchmark')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.grid(True)
    
    return fig