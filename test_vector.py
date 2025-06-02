import pandas as pd
import matplotlib.pyplot as plt
from vector_test.backtester import Backtester
from vector_test.strategy import SignalBoundStrategy
from vector_test.utils import prepare_data, calculate_metrics, plot_cumulative_returns


def main():
    # Load data
    print("Loading and preparing data...")
    data = pd.read_csv(r'C:\Users\sirui\Desktop\研报\my_back\ohlc_and_signal.csv')
    data = data[data['code'] == '000985.XSHG']
    # Prepare data (this shifts the signals)
    df = prepare_data(data)
    df['signal_prev'] = df.groupby('code')['signal'].shift(1)
    df['upperbound_prev'] = df.groupby('code')['upperbound'].shift(1)
    df['lowerbound_prev'] = df.groupby('code')['lowerbound'].shift(1)
    

    # Initialize strategy
    print("Initializing Signal Bound Strategy...")
    signal_strategy = SignalBoundStrategy()
    
    # Initialize backtester
    backtester = Backtester(df, strategy=signal_strategy)
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run()
    
    # Calculate metrics
    metrics = backtester.calculate_metrics()
    
    # Print metrics
    print("\nStrategy Performance Summary:")
    print(f"Total Return: {results['cumulative_returns'].iloc[-1]*100:.2f}%")
    print(f"Annual Return: {metrics['Annual Return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.2f}%")
    print(f"Win Rate: {metrics['Win Rate']:.2f}%")
    print(f"Alpha vs Benchmark: {metrics['Alpha']:.2f}%")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative returns
    ax.plot(results['cumulative_returns'].index, results['cumulative_returns']*100, 
            label='Strategy', linewidth=2)
    ax.plot(results['benchmark_cumulative_returns'].index, 
            results['benchmark_cumulative_returns']*100, 
            label='Benchmark', linewidth=2)
    
    ax.set_title('Cumulative Returns (%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()