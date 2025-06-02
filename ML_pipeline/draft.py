
##原来不考虑无法做空的代码
"""
backtest_framework.py
回测框架，负责执行回测逻辑和计算回测绩效
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import math


class BacktestFramework:
    """
    回测框架类，负责执行回测逻辑和计算回测绩效
    """

    def __init__(self, data, strategy, signal_generator, 
            start_date, end_date, initial_cash=10000000, 
            use_technical_signal=False, signal_method='AND', 
            technical_indicator='three_ma', 
            realistic_trading=False, 
            transaction_fee_rate=0.0003, 
            slippage_rate=0.0001):
        """
        初始化回测框架
        
        参数:
        data: 数据源，包含日期、股票ID、收盘价和因子数据
        strategy: 机器学习策略实例
        signal_generator: 信号生成器实例
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_cash: 初始资金
        use_technical_signal: 是否使用技术指标信号
        signal_method: 信号组合方法，'AND'或'OR'
        technical_indicator: 使用的技术指标类型
        realistic_trading: 是否考虑交易费用和滑点影响
        transaction_fee_rate: 交易费率，默认为万分之三(0.0003)
        slippage_rate: 滑点比例，默认为万分之一(0.0001)
        """
        self.data = data
        self.strategy = strategy
        self.signal_generator = signal_generator
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_cash = initial_cash
        self.use_technical_signal = use_technical_signal
        self.signal_method = signal_method
        self.technical_indicator = technical_indicator
        self.realistic_trading = realistic_trading
        self.transaction_fee_rate = transaction_fee_rate
        self.slippage_rate = slippage_rate
        
        # 获取交易日
        self.trading_days = self.data.index.get_level_values('date').unique()
        self.trading_days = sorted([day for day in self.trading_days if self.start_date <= day <= self.end_date])
        
        # 获取月初和月末日期
        self.month_begin_days = self._get_month_begin_days()
        self.month_end_days = self._get_month_end_days()
        
        # 初始化资产状态
        self.positions = {}       # 持仓情况 {stock_id: volume}
        self.cash = initial_cash  # 可用资金
        self.portfolio_values = []  # 组合价值历史
        self.dates = []           # 对应的日期
        self.trades = []          # 交易记录
        
        # 用于记录交易成本
        self.total_transaction_fees = 0.0
        self.total_slippage_cost = 0.0


    def calculate_transaction_cost(self, price, volume, is_buy=True):
        """
     买入操作：

    实际买入价格 = 原始价格 + 滑点调整
    总买入成本 = 买入数量 × 实际买入价格 + 交易手续费


    卖出操作：

    实际卖出价格 = 原始价格 - 滑点调整
    实际卖出收入 = 卖出数量 × 实际卖出价格 - 交易手续费
        """
        if not self.realistic_trading:
            return 0.0, 0.0
        
        # 交易总额
        total_amount = price * volume
        
        # 计算交易手续费
        transaction_fee = total_amount * self.transaction_fee_rate
        
        # 计算滑点成本
        # 买入时价格上涨，卖出时价格下跌
        slippage_direction = 1 if is_buy else -1
        slippage_cost = total_amount * self.slippage_rate * slippage_direction
        
        return transaction_fee, slippage_cost


    def run_backtest(self, display_progress=True):
        """
        执行回测
        
        参数:
        display_progress: 是否显示进度条
        """
        print(f"开始回测 {self.strategy.name} 策略...")
        print(f"回测时间段: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        if self.realistic_trading:
            print(f"开启真实交易模式，考虑交易成本")
            print(f"交易费率: {self.transaction_fee_rate*100:.4f}%, 滑点比例: {self.slippage_rate*100:.4f}%")
        else:
            print("向量化回测模式，不考虑交易成本")
        
        # 遍历交易日
        trading_days_iter = tqdm(self.trading_days) if display_progress else self.trading_days
        
        for current_date in trading_days_iter:
            # 如果不是月初，只更新组合价值
            if current_date not in self.month_begin_days:
                self.update_portfolio_value(current_date)
                continue
            
            # 跳过第一个月，因为没有上个月的数据
            if current_date == self.month_begin_days[0]:
                self.update_portfolio_value(current_date)
                continue
            
            print(f"\n调仓日期: {current_date.strftime('%Y-%m-%d')}")
            
            # 获取上个月的月初和月末
            month_idx = list(self.month_begin_days).index(current_date)
            prev_month_begin = self.month_begin_days[month_idx - 1]
            prev_month_end = self.month_end_days[month_idx - 1]
            
            # 从数据源中提取训练数据
            from ML_pipeline.data_processor import DataProcessor
            X, y, X_test, test_stocks = DataProcessor.prepare_daily_training_data(
                self.data, self.strategy.factor_codes, 
                prev_month_begin, prev_month_end, current_date
            )
            
            if X is None or len(X) == 0 or len(y) == 0 or len(X_test) == 0 or len(test_stocks) == 0:
                self.update_portfolio_value(current_date)
                continue
            
            # 训练模型
            self.strategy.train(X, y)
            
            # 预测
            predictions = self.strategy.predict(X_test)
            
            # 获取持仓信息
            positions = np.zeros(len(test_stocks))
            stock_indices = {}  # 映射股票ID到索引
            for i, stock_id in enumerate(test_stocks):
                positions[i] = self.positions.get(stock_id, 0)
                stock_indices[stock_id] = i
            
            # 获取机器学习交易信号
            ml_buy, ml_sell, high_return, low_return = self.strategy.get_signals(predictions, positions)
            
            print(f"收益率上分位数阈值: {high_return:.4f}, 收益率下分位数阈值: {low_return:.4f}")
            print(f"机器学习买入信号数量: {sum(ml_buy)}, 卖出信号数量: {sum(ml_sell)}")
            
            # 如果使用技术指标信号
            if self.use_technical_signal:
                # 获取每只股票的收盘价历史数据，用于计算技术指标
                close_data = np.zeros((len(test_stocks), 60))  # 假设需要60天的数据计算技术指标
                
                for i, stock_id in enumerate(test_stocks):
                    # 获取历史收盘价
                    try:
                        historical_prices = self.data.xs(stock_id, level='stock_id').loc[:current_date, 'close_qfq']
                        historical_prices = historical_prices.iloc[-60:].values
                        # 填充收盘价数据
                        close_data[i, -len(historical_prices):] = historical_prices
                    except:
                        # 如果获取数据失败，使用0填充
                        close_data[i, :] = 0
                
                # 生成技术指标信号
                tech_buy, tech_sell = self.get_technical_signals(close_data)
                            
                print(f"技术指标买入信号数量: {sum(tech_buy)}, 卖出信号数量: {sum(tech_sell)}")
                
                # 组合信号
                final_buy, final_sell = self.signal_generator.combine_signals(
                    (ml_buy, ml_sell), (tech_buy, tech_sell), self.signal_method
                )
            else:
                # 只使用机器学习信号
                final_buy, final_sell = ml_buy, ml_sell
            
            print(f"最终买入信号数量: {sum(final_buy)}, 卖出信号数量: {sum(final_sell)}")
            
            # 获取当前价格
            prices = np.zeros(len(test_stocks))
            for i, stock_id in enumerate(test_stocks):
                price = self.get_price(stock_id, current_date)
                if price is not None:
                    prices[i] = price
            
            # 首先，执行卖出操作
            # 1. 检查卖出信号
            for i, stock_id in enumerate(test_stocks):
                if final_sell[i] and stock_id in self.positions and self.positions[stock_id] > 0:
                    # 获取价格
                    price = prices[i]
                    if price is not None and price > 0:
                        # 执行卖出
                        position = self.positions[stock_id]
                        
                        # 计算交易成本
                        transaction_fee, slippage_cost = self.calculate_transaction_cost(price, position, is_buy=False)
                        
                        # 应用滑点影响
                        adjusted_price = price + slippage_cost / position if position > 0 else price
                        
                        # 计算卖出收入
                        sell_amount = position * adjusted_price - transaction_fee
                        self.cash += sell_amount
                        
                        # 更新总交易成本
                        self.total_transaction_fees += transaction_fee
                        self.total_slippage_cost += abs(slippage_cost)
                        
                        self.trades.append({
                            'date': current_date,
                            'stock_id': stock_id,
                            'action': 'SELL',
                            'price': price,
                            'adjusted_price': adjusted_price,
                            'volume': position,
                            'transaction_fee': transaction_fee,
                            'slippage_cost': slippage_cost,
                            'reason': 'SIGNAL'  # 标记为信号卖出
                        })
                        
                        if self.realistic_trading:
                            print(f"卖出信号触发: 卖出 {stock_id}: {position}股，原价: {price}，调整价: {adjusted_price}，手续费: {transaction_fee}，滑点成本: {slippage_cost}，总收入: {sell_amount}")
                        else:
                            print(f"卖出信号触发: 卖出 {stock_id}: {position}股，价格：{price}，总收入：{position * price}")
                        
                        self.positions[stock_id] = 0
            
            # 2. 月度调仓 - 卖出不符合持仓条件的股票（非卖出信号导致的卖出）
            for stock_id, position in list(self.positions.items()):
                # 只处理有持仓且不在卖出信号中的股票
                if position > 0 and (stock_id not in stock_indices or not final_sell[stock_indices[stock_id]]):
                    price = self.get_price(stock_id, current_date)
                    if price is not None and price > 0:
                        # 计算交易成本
                        transaction_fee, slippage_cost = self.calculate_transaction_cost(price, position, is_buy=False)
                        
                        # 应用滑点影响
                        adjusted_price = price + slippage_cost / position if position > 0 else price
                        
                        # 计算卖出收入
                        sell_amount = position * adjusted_price - transaction_fee
                        self.cash += sell_amount
                        
                        # 更新总交易成本
                        self.total_transaction_fees += transaction_fee
                        self.total_slippage_cost += abs(slippage_cost)
                        
                        self.trades.append({
                            'date': current_date,
                            'stock_id': stock_id,
                            'action': 'SELL',
                            'price': price,
                            'adjusted_price': adjusted_price,
                            'volume': position,
                            'transaction_fee': transaction_fee,
                            'slippage_cost': slippage_cost,
                            'reason': 'REBALANCE'  # 标记为调仓卖出
                        })
                        
                        if self.realistic_trading:
                            print(f"月度调仓: 卖出 {stock_id}: {position}股，原价: {price}，调整价: {adjusted_price}，手续费: {transaction_fee}，滑点成本: {slippage_cost}，总收入: {sell_amount}")
                        else:
                            print(f"月度调仓: 卖出 {stock_id}: {position}股，价格：{price}，总收入：{position * price}")
                        
                        self.positions[stock_id] = 0
            
            # 计算购买股数
            buy_shares = self.strategy.calculate_position_sizes(final_buy, self.cash, prices)
            
            # 执行买入操作
            for i, stock_id in enumerate(test_stocks):
                if buy_shares[i] > 0:
                    price = prices[i]
                    
                    # 计算交易成本
                    transaction_fee, slippage_cost = self.calculate_transaction_cost(price, buy_shares[i], is_buy=True)
                    
                    # 应用滑点影响
                    adjusted_price = price + slippage_cost / buy_shares[i] if buy_shares[i] > 0 else price
                    
                    # 计算买入成本
                    buy_cost = buy_shares[i] * adjusted_price + transaction_fee
                    
                    if buy_cost <= self.cash:
                        # 更新持仓和现金
                        if stock_id not in self.positions:
                            self.positions[stock_id] = 0
                        self.positions[stock_id] += buy_shares[i]
                        self.cash -= buy_cost
                        
                        # 更新总交易成本
                        self.total_transaction_fees += transaction_fee
                        self.total_slippage_cost += abs(slippage_cost)
                        
                        self.trades.append({
                            'date': current_date,
                            'stock_id': stock_id,
                            'action': 'BUY',
                            'price': price,
                            'adjusted_price': adjusted_price,
                            'volume': buy_shares[i],
                            'transaction_fee': transaction_fee,
                            'slippage_cost': slippage_cost
                        })
                        
                        if self.realistic_trading:
                            print(f"买入 {stock_id}: {buy_shares[i]}股，原价: {price}，调整价: {adjusted_price}，手续费: {transaction_fee}，滑点成本: {slippage_cost}，总成本: {buy_cost}")
                        else:
                            print(f"买入 {stock_id}: {buy_shares[i]}股，价格：{price}，总成本：{buy_shares[i] * price}")
            
            # 更新组合价值
            self.update_portfolio_value(current_date)
        
        # 回测结束，计算绩效指标
        self.performance = self.calculate_performance()
        
        # 打印交易成本统计
        if self.realistic_trading:
            print("\n===== 交易成本统计 =====")
            print(f"总交易费用: {self.total_transaction_fees:.2f} ({self.total_transaction_fees / self.initial_cash * 100:.4f}% 初始资金)")
            print(f"总滑点成本: {self.total_slippage_cost:.2f} ({self.total_slippage_cost / self.initial_cash * 100:.4f}% 初始资金)")
            print(f"总交易成本: {self.total_transaction_fees + self.total_slippage_cost:.2f} ({(self.total_transaction_fees + self.total_slippage_cost) / self.initial_cash * 100:.4f}% 初始资金)")
        
        return self.performance


    def calculate_performance(self):
        """
        计算回测绩效指标
        
        返回:
        回测绩效指标字典
        """
        if len(self.portfolio_values) <= 1:
            print("回测周期太短，无法计算绩效指标")
            return None
        
        # 计算收益率
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        # 计算年化收益率
        days = (self.dates[-1] - self.dates[0]).days
        annual_return = (1 + cumulative_returns[-1]) ** (365 / days) - 1
        
        # 计算波动率
        volatility = np.std(returns) * np.sqrt(252)  # 假设一年252个交易日
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # 计算最大回撤
        max_drawdown = 0
        peak = self.portfolio_values[0]
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 返回绩效指标
        performance = {
            'total_return': self.portfolio_values[-1] / self.initial_cash - 1,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        # 如果考虑交易成本，增加交易成本相关指标
        if self.realistic_trading:
            performance.update({
                'total_transaction_fees': self.total_transaction_fees,
                'total_slippage_cost': self.total_slippage_cost,
                'total_cost_ratio': (self.total_transaction_fees + self.total_slippage_cost) / self.initial_cash
            })
        
        return performance


    def print_performance(self):
        """
        打印回测绩效指标
        """
        if self.performance is None:
            print("未计算绩效指标")
            return
        
        print("\n====== 回测结果 ======")
        print(f"策略名称: {self.strategy.name}")
        print(f"回测时间段: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"初始资金: {self.initial_cash:,.2f}")
        print(f"最终资金: {self.portfolio_values[-1]:,.2f}")
        print(f"总收益率: {self.performance['total_return'] * 100:.2f}%")
        print(f"年化收益率: {self.performance['annual_return'] * 100:.2f}%")
        print(f"波动率: {self.performance['volatility'] * 100:.2f}%")
        print(f"夏普比率: {self.performance['sharpe_ratio']:.4f}")
        print(f"最大回撤: {self.performance['max_drawdown'] * 100:.2f}%")
        
        # 如果考虑交易成本，打印交易成本相关指标
        if self.realistic_trading:
            print("\n------ 交易成本 ------")
            print(f"总交易费用: {self.performance['total_transaction_fees']:,.2f} ({self.performance['total_transaction_fees'] / self.initial_cash * 100:.4f}% 初始资金)")
            print(f"总滑点成本: {self.performance['total_slippage_cost']:,.2f} ({self.performance['total_slippage_cost'] / self.initial_cash * 100:.4f}% 初始资金)")
            print(f"总交易成本: {self.performance['total_transaction_fees'] + self.performance['total_slippage_cost']:,.2f} ({self.performance['total_cost_ratio'] * 100:.4f}% 初始资金)")
            
            # 计算成本对收益率的影响
            cost_impact = (self.performance['total_transaction_fees'] + self.performance['total_slippage_cost']) / self.initial_cash
            print(f"成本对总收益率的影响: -{cost_impact * 100:.4f}%")
            
            # 计算无成本情况下的收益率
            no_cost_return = self.performance['total_return'] + cost_impact
            print(f"无成本情况下的总收益率: {no_cost_return * 100:.2f}%")


    def plot_results(self):
        """
        绘制回测结果
        """
        if len(self.portfolio_values) <= 1:
            print("回测周期太短，无法绘制结果")
            return
        
        plt.figure(figsize=(15, 12))
        
        # 绘制资产价值曲线
        plt.subplot(3, 1, 1)
        plt.plot([d.strftime('%Y-%m-%d') for d in self.dates], self.portfolio_values)
        plt.title(f'{self.strategy.name} 策略资产价值曲线 {"(考虑交易成本)" if self.realistic_trading else "(不考虑交易成本)"}')
        plt.xlabel('日期')
        plt.ylabel('资产价值')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 绘制收益率曲线
        plt.subplot(3, 1, 2)
        returns = np.array(self.portfolio_values) / self.initial_cash - 1
        plt.plot([d.strftime('%Y-%m-%d') for d in self.dates], returns * 100)
        plt.title(f'{self.strategy.name} 策略收益率曲线')
        plt.xlabel('日期')
        plt.ylabel('收益率(%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 如果考虑交易成本，绘制累计交易成本曲线
        if self.realistic_trading and len(self.trades) > 0:
            plt.subplot(3, 1, 3)
            
            # 获取交易日期和成本
            trade_dates = [trade['date'] for trade in self.trades]
            transaction_fees = [trade.get('transaction_fee', 0) for trade in self.trades]
            slippage_costs = [abs(trade.get('slippage_cost', 0)) for trade in self.trades]
            
            # 计算累计成本
            cumulative_fees = np.cumsum(transaction_fees)
            cumulative_slippage = np.cumsum(slippage_costs)
            cumulative_total = cumulative_fees + cumulative_slippage
            
            # 绘制累计成本曲线
            plt.plot([d.strftime('%Y-%m-%d') for d in trade_dates], cumulative_fees, label='交易费用')
            plt.plot([d.strftime('%Y-%m-%d') for d in trade_dates], cumulative_slippage, label='滑点成本')
            plt.plot([d.strftime('%Y-%m-%d') for d in trade_dates], cumulative_total, label='总交易成本')
            
            plt.title(f'{self.strategy.name} 策略累计交易成本')
            plt.xlabel('日期')
            plt.ylabel('累计成本')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
        
    def print_performance(self):
        """
        打印回测绩效指标
        """
        if self.performance is None:
            print("未计算绩效指标")
            return
        
        print("\n====== 回测结果 ======")
        print(f"策略名称: {self.strategy.name}")
        print(f"回测时间段: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"初始资金: {self.initial_cash}")
        print(f"最终资金: {self.portfolio_values[-1]}")
        print(f"总收益率: {self.performance['total_return'] * 100:.2f}%")
        print(f"年化收益率: {self.performance['annual_return'] * 100:.2f}%")
        print(f"波动率: {self.performance['volatility'] * 100:.2f}%")
        print(f"夏普比率: {self.performance['sharpe_ratio']:.4f}")
        print(f"最大回撤: {self.performance['max_drawdown'] * 100:.2f}%")


#向量化回测


"""
vectorized_backtest_framework.py
基于NumPy向量化的回测框架，大幅提高回测速度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import time


class VectorizedBacktestFramework:
    """
    向量化回测框架类，使用NumPy数组进行高效计算
    """
    
    def __init__(self, data, strategy, signal_generator, 
             start_date, end_date, initial_cash=10000000, 
             use_technical_signal=False, signal_method='AND', 
             technical_indicator='three_ma', 
             realistic_trading=False, 
             transaction_fee_rate=0.0003, 
             slippage_rate=0.0001,
             target_position_count=10,
             rebalance_freq='month'):
        """
        初始化回测框架
        
        参数:
        data: 数据源，包含日期、股票ID、收盘价和因子数据（Pandas MultiIndex DataFrame）
        strategy: 机器学习策略实例
        signal_generator: 信号生成器实例
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_cash: 初始资金
        use_technical_signal: 是否使用技术指标信号
        signal_method: 信号组合方法，'AND'或'OR'
        technical_indicator: 使用的技术指标类型
        realistic_trading: 是否考虑交易费用和滑点影响
        transaction_fee_rate: 交易费率，默认为万分之三(0.0003)
        slippage_rate: 滑点比例，默认为万分之一(0.0001)
        target_position_count: 目标持仓数量，默认为10
        rebalance_freq: 调仓频率，可选'day', 'week', 'month'
        """
        self.data = data
        self.strategy = strategy
        self.signal_generator = signal_generator
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_cash = initial_cash
        self.use_technical_signal = use_technical_signal
        self.signal_method = signal_method
        self.technical_indicator = technical_indicator
        self.target_position_count = target_position_count
        self.realistic_trading = realistic_trading
        self.transaction_fee_rate = transaction_fee_rate
        self.slippage_rate = slippage_rate
        self.rebalance_freq = rebalance_freq
        
        # 获取交易日
        self.trading_days = self.data.index.get_level_values('date').unique()
        self.trading_days = sorted([day for day in self.trading_days if self.start_date <= day <= self.end_date])
        
        # 获取调仓日期
        if rebalance_freq == 'month':
            self.rebalance_days = self._get_month_begin_days()
        elif rebalance_freq == 'week':
            self.rebalance_days = self._get_week_begin_days()
        else:  # 默认为日频
            self.rebalance_days = self.trading_days
        
        # 获取月末日期（如果需要）
        self.month_end_days = self._get_month_end_days()
        
        # 初始化状态变量
        self.positions = {}  # {date: {stock_id: volume}}
        self.cash = np.zeros(len(self.trading_days))
        self.cash[0] = initial_cash  # 初始资金
        self.portfolio_values = np.zeros(len(self.trading_days))
        self.portfolio_values[0] = initial_cash
        
        # 交易记录
        self.trades = []
        
        # 交易成本
        self.transaction_fees = np.zeros(len(self.trading_days))
        self.slippage_costs = np.zeros(len(self.trading_days))
        
        # 日期和交易日索引映射
        self.date_idx_map = {date: i for i, date in enumerate(self.trading_days)}
        
        # 所有股票列表
        self.all_stocks = self.data.index.get_level_values('stock_id').unique()
        
        # 股票池
        self.stock_pool = {}  # {date: [stock_ids]}
        
        # 创建价格矩阵（股票 x 日期）
        self._create_price_matrix()
    
    def _create_price_matrix(self):
        """
        创建价格矩阵，加速后续计算
        """
        print("创建价格矩阵...")
        
        # 创建日期和股票ID的唯一列表
        self.dates_list = self.trading_days
        self.stocks_list = list(self.all_stocks)
        
        # 创建日期和股票ID的索引映射
        self.date_to_idx = {date: i for i, date in enumerate(self.dates_list)}
        self.stock_to_idx = {stock: i for i, stock in enumerate(self.stocks_list)}
        
        # 创建价格矩阵：行是股票，列是日期
        self.price_matrix = np.zeros((len(self.stocks_list), len(self.dates_list)))
        
        print("填充价格矩阵...")
        # 从原始数据中填充价格矩阵
        for stock_id in tqdm(self.stocks_list):
            try:
                # 获取该股票的所有价格数据
                stock_prices = self.data.xs(stock_id, level='stock_id')['close_qfq']
                
                # 只保留交易日期范围内的数据
                stock_prices = stock_prices.loc[self.start_date:self.end_date]
                
                # 填充价格矩阵
                for date, price in stock_prices.items():
                    if date in self.date_to_idx:
                        date_idx = self.date_to_idx[date]
                        stock_idx = self.stock_to_idx[stock_id]
                        self.price_matrix[stock_idx, date_idx] = price
            except KeyError:
                # 如果股票不在某些日期中，跳过
                continue
        
        print("价格矩阵创建完成")
    
    def _get_month_begin_days(self):
        """
        获取每月第一个交易日
        
        返回:
        每月第一个交易日列表
        """
        # 将交易日转换为月份
        months = np.vectorize(lambda x: x.month)(self.trading_days)
        years = np.vectorize(lambda x: x.year)(self.trading_days)
        
        # 创建年月组合
        year_months = pd.Series([f"{y}-{m:02d}" for y, m in zip(years, months)], index=range(len(self.trading_days)))
        
        # 找出每个年月组合的第一个交易日
        month_begin_indices = ~year_months.duplicated()
        month_begin = [self.trading_days[i] for i in range(len(self.trading_days)) if month_begin_indices[i]]
        
        return month_begin
    
    def _get_week_begin_days(self):
        """
        获取每周第一个交易日
        
        返回:
        每周第一个交易日列表
        """
        # 将交易日转换为周数
        weeks = np.vectorize(lambda x: x.isocalendar()[1])(self.trading_days)
        years = np.vectorize(lambda x: x.year)(self.trading_days)
        
        # 创建年周组合
        year_weeks = pd.Series([f"{y}-{w:02d}" for y, w in zip(years, weeks)], index=range(len(self.trading_days)))
        
        # 找出每个年周组合的第一个交易日
        week_begin_indices = ~year_weeks.duplicated()
        week_begin = [self.trading_days[i] for i in range(len(self.trading_days)) if week_begin_indices[i]]
        
        return week_begin
    
    def _get_month_end_days(self):
        """
        获取每月最后一个交易日
        
        返回:
        每月最后一个交易日列表
        """
        # 将交易日转换为月份
        months = np.vectorize(lambda x: x.month)(self.trading_days)
        years = np.vectorize(lambda x: x.year)(self.trading_days)
        
        # 创建年月组合
        year_months = pd.Series([f"{y}-{m:02d}" for y, m in zip(years, months)], index=range(len(self.trading_days)))
        
        # 找出每个年月组合的最后一个交易日
        month_end_indices = ~year_months[::-1].duplicated()
        month_end_indices = month_end_indices[::-1]
        month_end = [self.trading_days[i] for i in range(len(self.trading_days)) if month_end_indices[i]]
        
        return month_end
    
    def get_price(self, stock_id, date):
        """
        从价格矩阵获取价格，比直接查询DataFrame更快
        
        参数:
        stock_id: 股票ID
        date: 日期
        
        返回:
        价格
        """
        if stock_id in self.stock_to_idx and date in self.date_to_idx:
            stock_idx = self.stock_to_idx[stock_id]
            date_idx = self.date_to_idx[date]
            return self.price_matrix[stock_idx, date_idx]
        return 0
    
    def get_historical_prices(self, stock_id, end_date, lookback=60):
        """
        获取历史价格数据
        
        参数:
        stock_id: 股票ID
        end_date: 结束日期
        lookback: 回溯天数
        
        返回:
        历史价格数组
        """
        if stock_id in self.stock_to_idx and end_date in self.date_to_idx:
            stock_idx = self.stock_to_idx[stock_id]
            end_idx = self.date_to_idx[end_date]
            start_idx = max(0, end_idx - lookback)
            return self.price_matrix[stock_idx, start_idx:end_idx+1]
        return np.array([])
    
    def get_current_position_vector(self, date_idx, stock_indices):
        """
        获取当前持仓向量（用于向量化信号生成）
        
        参数:
        date_idx: 日期索引
        stock_indices: 股票索引列表
        
        返回:
        持仓向量，形状与stock_indices相同
        """
        if date_idx <= 0:
            return np.zeros(len(stock_indices))
        
        # 获取前一天的持仓情况
        prev_date = self.trading_days[date_idx - 1]
        
        if prev_date not in self.positions:
            return np.zeros(len(stock_indices))
        
        position_vector = np.zeros(len(stock_indices))
        position_dict = self.positions[prev_date]
        
        for i, stock_idx in enumerate(stock_indices):
            stock_id = self.stocks_list[stock_idx]
            position_vector[i] = position_dict.get(stock_id, 0)
        
        return position_vector
    
    def update_portfolio_value(self, date_idx):
        """
        更新投资组合价值
        
        参数:
        date_idx: 日期索引
        """
        if date_idx <= 0:
            return
        
        date = self.trading_days[date_idx]
        prev_date = self.trading_days[date_idx - 1]
        
        # 如果今天没有持仓记录，复制前一天的持仓
        if date not in self.positions and prev_date in self.positions:
            self.positions[date] = self.positions[prev_date].copy()
        elif date not in self.positions:
            self.positions[date] = {}
        
        # 更新现金（如果没有交易，现金不变）
        if date_idx > 0 and self.cash[date_idx] == 0:
            self.cash[date_idx] = self.cash[date_idx - 1]
        
        # 计算投资组合价值 = 现金 + 所有股票持仓价值
        portfolio_value = self.cash[date_idx]
        for stock_id, volume in self.positions[date].items():
            price = self.get_price(stock_id, date)
            if price > 0:
                portfolio_value += price * volume
        
        self.portfolio_values[date_idx] = portfolio_value
    
    def create_stock_pool(self, date, test_stocks, predictions):
        """
        根据预测收益率创建股票池
        
        参数:
        date: 当前日期
        test_stocks: 预测的股票ID列表
        predictions: 预测收益率数组
        
        返回:
        选中的股票ID列表
        """
        # 创建股票ID和预测收益率的映射
        stock_predictions = list(zip(test_stocks, predictions))
        
        # 按预测收益率从高到低排序
        sorted_stocks = sorted(stock_predictions, key=lambda x: x[1], reverse=True)
        
        # 选择预测收益率最高的N只股票
        target_position_count = min(self.target_position_count, len(sorted_stocks))
        selected_stocks = [stock for stock, _ in sorted_stocks[:target_position_count]]
        
        # 保存到股票池
        self.stock_pool[date] = selected_stocks
        
        return selected_stocks
    
    def apply_technical_filter(self, date, stock_pool):
        """
        应用技术指标过滤股票池
        
        参数:
        date: 当前日期
        stock_pool: 原始股票池
        
        返回:
        过滤后的股票池
        """
        if not self.use_technical_signal:
            return stock_pool
        
        filtered_pool = []
        
        for stock_id in stock_pool:
            # 获取该股票的历史收盘价
            historical_prices = self.get_historical_prices(stock_id, date, lookback=60)
            
            if len(historical_prices) < 60:
                continue
            
            # 计算技术指标
            if self.technical_indicator == 'three_ma':
                # 简单计算移动平均线
                ma5 = np.mean(historical_prices[-5:])
                ma20 = np.mean(historical_prices[-20:])
                ma60 = np.mean(historical_prices[-60:])
                
                # 判断买入条件：短期和中期均线都在长期均线之上
                if ma5 > ma60 and ma20 > ma60:
                    filtered_pool.append(stock_id)
            
            elif self.technical_indicator == 'ma_cross':
                # 计算移动平均线
                ma5 = np.mean(historical_prices[-6:-1])  # 昨日5日均线
                ma20 = np.mean(historical_prices[-21:-1])  # 昨日20日均线
                current_ma5 = np.mean(historical_prices[-5:])  # 今日5日均线
                current_ma20 = np.mean(historical_prices[-20:])  # 今日20日均线
                
                # 判断均线交叉：5日均线上穿20日均线
                if ma5 <= ma20 and current_ma5 > current_ma20:
                    filtered_pool.append(stock_id)
            
            elif self.technical_indicator == 'bollinger':
                # 计算布林带
                ma20 = np.mean(historical_prices[-20:])
                std20 = np.std(historical_prices[-20:])
                lower_band = ma20 - 2 * std20
                
                # 判断买入条件：价格接近下轨
                current_price = historical_prices[-1]
                if current_price <= lower_band * 1.05:  # 允许5%的误差
                    filtered_pool.append(stock_id)
        
        return filtered_pool if filtered_pool else stock_pool  # 如果过滤后为空，返回原始股票池
    
    def execute_rebalance(self, date_idx):
        """
        执行投资组合再平衡（向量化操作）
        
        参数:
        date_idx: 日期索引
        """
        date = self.trading_days[date_idx]
        
        # 跳过第一个交易日
        if date_idx == 0:
            self.positions[date] = {}
            return
        
        # 获取前一天的日期和持仓
        prev_date_idx = date_idx - 1
        prev_date = self.trading_days[prev_date_idx]
        
        # 确保前一天的持仓存在
        if prev_date not in self.positions:
            self.positions[prev_date] = {}
        
        # 复制前一天的持仓作为起点
        current_position = self.positions[prev_date].copy()
        
        # 如果是调仓日，执行再平衡
        if date in self.rebalance_days and date in self.stock_pool:
            # 获取当前股票池
            new_portfolio = self.stock_pool[date]
            
            # 卖出不在新投资组合中的股票
            total_sell_value = 0
            transaction_fee = 0
            slippage_cost = 0
            
            for stock_id, volume in list(current_position.items()):
                if stock_id not in new_portfolio and volume > 0:
                    price = self.get_price(stock_id, date)
                    if price > 0:
                        # 计算卖出收入和交易成本
                        if self.realistic_trading:
                            fee = price * volume * self.transaction_fee_rate
                            slip = price * volume * self.slippage_rate
                            sell_value = price * volume - fee - slip
                            transaction_fee += fee
                            slippage_cost += slip
                        else:
                            sell_value = price * volume
                        
                        total_sell_value += sell_value
                        del current_position[stock_id]
                        
                        # 记录交易
                        self.trades.append({
                            'date': date,
                            'stock_id': stock_id,
                            'action': 'SELL',
                            'price': price,
                            'volume': volume,
                            'value': sell_value,
                            'reason': 'REBALANCE'
                        })
            
            # 更新现金
            available_cash = self.cash[prev_date_idx] + total_sell_value
            
            # 平均分配资金到新增持仓
            if new_portfolio:
                # 获取当前已持有的股票
                current_stocks = set(current_position.keys())
                
                # 需要买入的新股票
                stocks_to_buy = [s for s in new_portfolio if s not in current_stocks]
                
                if stocks_to_buy:
                    # 计算每只股票分配的资金
                    cash_per_stock = available_cash / len(stocks_to_buy)
                    
                    # 买入新股票
                    total_buy_cost = 0
                    
                    for stock_id in stocks_to_buy:
                        price = self.get_price(stock_id, date)
                        if price > 0:
                            # 计算可买入的股数（确保是100的整数倍）
                            max_shares = int(cash_per_stock / (price * 100)) * 100
                            
                            if max_shares > 0:
                                # 计算买入成本和交易成本
                                if self.realistic_trading:
                                    fee = price * max_shares * self.transaction_fee_rate
                                    slip = price * max_shares * self.slippage_rate
                                    buy_cost = price * max_shares + fee + slip
                                    transaction_fee += fee
                                    slippage_cost += slip
                                else:
                                    buy_cost = price * max_shares
                                
                                if buy_cost <= available_cash:
                                    current_position[stock_id] = max_shares
                                    total_buy_cost += buy_cost
                                    
                                    # 记录交易
                                    self.trades.append({
                                        'date': date,
                                        'stock_id': stock_id,
                                        'action': 'BUY',
                                        'price': price,
                                        'volume': max_shares,
                                        'value': buy_cost,
                                        'reason': 'REBALANCE'
                                    })
                    
                    # 更新现金
                    available_cash -= total_buy_cost
            
            # 更新现金
            self.cash[date_idx] = available_cash
            
            # 更新交易成本记录
            self.transaction_fees[date_idx] = transaction_fee
            self.slippage_costs[date_idx] = slippage_cost
        else:
            # 如果不是调仓日，现金不变
            self.cash[date_idx] = self.cash[prev_date_idx]
        
        # 更新持仓记录
        self.positions[date] = current_position
        
        # 更新组合价值
        self.update_portfolio_value(date_idx)
    
    def run_backtest(self, display_progress=True):
        """
        运行向量化回测
        
        参数:
        display_progress: 是否显示进度条
        
        返回:
        回测性能指标
        """
        start_time = time.time()
        
        print(f"开始回测 {self.strategy.name} 策略...")
        print(f"回测时间段: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"目标持仓数量: {self.target_position_count}")
        print(f"调仓频率: {self.rebalance_freq}")
        
        if self.realistic_trading:
            print(f"考虑交易成本，费率: {self.transaction_fee_rate*100:.4f}%, 滑点: {self.slippage_rate*100:.4f}%")
        else:
            print("不考虑交易成本")
        
        # 遍历交易日
        trading_days_iter = tqdm(enumerate(self.trading_days), total=len(self.trading_days)) if display_progress else enumerate(self.trading_days)
        
        for date_idx, current_date in trading_days_iter:
            # 如果是调仓日
            if current_date in self.rebalance_days:
                # 跳过第一个调仓日，因为没有上个月的数据用于训练
                if current_date == self.rebalance_days[0]:
                    self.update_portfolio_value(date_idx)
                    continue
                
                # 获取上个月的月初和月末
                rebalance_idx = list(self.rebalance_days).index(current_date)
                prev_rebalance_date = self.rebalance_days[rebalance_idx - 1]
                
                # 获取对应的月末日期（如果是月度调仓）
                if self.rebalance_freq == 'month':
                    prev_month_end_idx = list(self.month_begin_days).index(current_date) - 1
                    prev_month_end = self.month_end_days[prev_month_end_idx]
                else:
                    # 对于其他频率，使用上一个调仓日作为训练结束日期
                    prev_month_end = prev_rebalance_date
                
                # 从数据源中提取训练数据
                from ML_pipeline.data_processor import DataProcessor
                X, y, X_test, test_stocks = DataProcessor.prepare_daily_training_data(
                    self.data, self.strategy.factor_codes, 
                    prev_rebalance_date, prev_month_end, current_date
                )
                
                if X is None or len(X) == 0 or len(y) == 0 or len(X_test) == 0 or len(test_stocks) == 0:
                    self.update_portfolio_value(date_idx)
                    continue
                
                # 训练模型
                self.strategy.train(X, y)
                
                # 预测
                predictions = self.strategy.predict(X_test)
                
                # 创建股票池
                selected_stocks = self.create_stock_pool(current_date, test_stocks, predictions)
                
                # 应用技术指标过滤
                if self.use_technical_signal:
                    filtered_stocks = self.apply_technical_filter(current_date, selected_stocks)
                    self.stock_pool[current_date] = filtered_stocks
            
            # 执行再平衡（每日都会调用，但只有在调仓日才会实际执行交易）
            self.execute_rebalance(date_idx)
        
        # 计算回测性能指标
        performance = self.calculate_performance()
        
        # 计算回测时间
        elapsed_time = time.time() - start_time
        print(f"回测完成，耗时 {elapsed_time:.2f} 秒")
        
        return performance
    
    def calculate_performance(self):
        """
        计算回测性能指标
        
        返回:
        性能指标字典
        """
        # 计算日收益率
        daily_returns = np.zeros(len(self.trading_days) - 1)
        for i in range(1, len(self.trading_days)):
            if self.portfolio_values[i-1] > 0:
                daily_returns[i-1] = self.portfolio_values[i] / self.portfolio_values[i-1] - 1
        
        # 计算累计收益率
        cumulative_returns = np.cumprod(1 + daily_returns) - 1
        
        # 计算年化收益率
        total_days = (self.trading_days[-1] - self.trading_days[0]).days
        annual_return = (1 + cumulative_returns[-1]) ** (365 / total_days) - 1
        
        # 计算年化波动率
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # 计算最大回撤
        max_drawdown = 0
        peak = self.portfolio_values[0]
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算交易成本影响
        total_transaction_fees = np.sum(self.transaction_fees)
        total_slippage_costs = np.sum(self.slippage_costs)
        total_costs = total_transaction_fees + total_slippage_costs
        cost_ratio = total_costs / self.initial_cash
        
        performance = {
            'initial_value': self.initial_cash,
            'final_value': self.portfolio_values[-1],
            'total_return': self.portfolio_values[-1] / self.initial_cash - 1,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'transaction_fees': total_transaction_fees,
            'slippage_costs': total_slippage_costs,
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns
        }
        
        return performance
    
    def print_performance(self):
        """
        打印回测性能指标
        """
        perf = self.calculate_performance()
        
        print("\n====== 回测结果 ======")
        print(f"策略名称: {self.strategy.name}")
        print(f"回测时间段: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"初始资金: {perf['initial_value']:,.2f}")
        print(f"最终资金: {perf['final_value']:,.2f}")
        print(f"总收益率: {perf['total_return']*100:.2f}%")
        print(f"年化收益率: {perf['annual_return']*100:.2f}%")
        print(f"年化波动率: {perf['annual_volatility']*100:.2f}%")
        print(f"夏普比率: {perf['sharpe_ratio']:.4f}")
        print(f"最大回撤: {perf['max_drawdown']*100:.2f}%")
        
        if self.realistic_trading:
            print("\n------ 交易成本 ------")
            print(f"总交易费用: {perf['transaction_fees']:,.2f} ({perf['transaction_fees']/self.initial_cash*100:.4f}% 初始资金)")
            print(f"总滑点成本: {perf['slippage_costs']:,.2f} ({perf['slippage_costs']/self.initial_cash*100:.4f}% 初始资金)")
            print(f"总交易成本: {perf['total_costs']:,.2f} ({perf['cost_ratio']*100:.4f}% 初始资金)")
            
            # 计算无成本情况下的收益率
            no_cost_return = perf['total_return'] + perf['cost_ratio']
            print(f"成本对总收益率的影响: -{perf['cost_ratio']*100:.4f}%")
            print(f"无成本情况下的总收益率: {no_cost_return*100:.2f}%")
    
    def plot_results(self, save_path=None):
        """
        绘制回测结果图表
        
        参数:
        save_path: 图表保存路径（如果提供）
        """
        perf = self.calculate_performance()
        
        plt.figure(figsize=(15, 12))
        
        # 绘制资产价值曲线
        plt.subplot(3, 1, 1)
        plt.plot([d.strftime('%Y-%m-%d') for d in self.trading_days], self.portfolio_values)
        plt.title(f'{self.strategy.name} 策略资产价值曲线 {"(考虑交易成本)" if self.realistic_trading else "(不考虑交易成本)"}')
        plt.xlabel('日期')
        plt.ylabel('资产价值')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 绘制收益率曲线
        plt.subplot(3, 1, 2)
        returns = np.array(self.portfolio_values) / self.initial_cash - 1
        plt.plot([d.strftime('%Y-%m-%d') for d in self.trading_days], returns * 100)
        plt.title(f'{self.strategy.name} 策略收益率曲线')
        plt.xlabel('日期')
        plt.ylabel('收益率(%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 绘制交易成本曲线（如果考虑交易成本）
        if self.realistic_trading:
            plt.subplot(3, 1, 3)
            
            # 计算累计交易成本
            cumulative_fees = np.cumsum(self.transaction_fees)
            cumulative_slippage = np.cumsum(self.slippage_costs)
            cumulative_total = cumulative_fees + cumulative_slippage
            
            # 绘制累计成本曲线
            plt.plot([d.strftime('%Y-%m-%d') for d in self.trading_days], cumulative_fees, label='交易费用')
            plt.plot([d.strftime('%Y-%m-%d') for d in self.trading_days], cumulative_slippage, label='滑点成本')
            plt.plot([d.strftime('%Y-%m-%d') for d in self.trading_days], cumulative_total, label='总交易成本')
            
            plt.title(f'{self.strategy.name} 策略累计交易成本')
            plt.xlabel('日期')
            plt.ylabel('累计成本')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图表（如果提供路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_trades(self):
        """
        分析交易记录
        
        返回:
        交易分析结果
        """
        if not self.trades:
            print("没有交易记录可供分析")
            return {}
        
        # 将交易记录转换为DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # 计算交易统计数据
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
        sell_trades = len(trades_df[trades_df['action'] == 'SELL'])
        
        # 计算平均交易规模
        avg_trade_size = trades_df['value'].mean()
        
        # 计算换手率（总交易金额 / 平均资产规模）
        total_trade_value = trades_df['value'].sum()
        avg_portfolio_value = np.mean(self.portfolio_values)
        turnover_rate = total_trade_value / avg_portfolio_value
        
        # 统计交易频率（每月交易次数）
        trades_df['month'] = [d.strftime('%Y-%m') for d in trades_df['date']]
        monthly_trades = trades_df.groupby('month').size()
        avg_monthly_trades = monthly_trades.mean()
        
        # 计算平均持仓时间（难以精确计算，这里只是估计）
        # 获取每个股票的第一次买入和最后一次卖出时间
        first_buys = trades_df[trades_df['action'] == 'BUY'].groupby('stock_id')['date'].min()
        last_sells = trades_df[trades_df['action'] == 'SELL'].groupby('stock_id')['date'].max()
        
        # 对于能够匹配的股票，计算持有天数
        holding_periods = []
        for stock_id in first_buys.index:
            if stock_id in last_sells.index and last_sells[stock_id] > first_buys[stock_id]:
                days = (last_sells[stock_id] - first_buys[stock_id]).days
                holding_periods.append(days)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # 返回分析结果
        analysis = {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'avg_trade_size': avg_trade_size,
            'turnover_rate': turnover_rate,
            'avg_monthly_trades': avg_monthly_trades,
            'avg_holding_period': avg_holding_period
        }
        
        # 打印分析结果
        print("\n====== 交易分析 ======")
        print(f"总交易次数: {total_trades}")
        print(f"买入交易: {buy_trades}")
        print(f"卖出交易: {sell_trades}")
        print(f"平均交易规模: {avg_trade_size:,.2f}")
        print(f"换手率: {turnover_rate:.2f}")
        print(f"平均月交易次数: {avg_monthly_trades:.2f}")
        print(f"平均持仓天数: {avg_holding_period:.2f}")
        
        return analysis