"""
transaction_manager.py
负责计算交易规模、交易成本和执行交易操作的模块
"""

import numpy as np
import pandas as pd
from ML_pipeline.portfolio_optimizer import PortfolioOptimizer

class TransactionManager:
    """
    交易管理器，负责计算交易规模、交易成本和执行交易操作
    降低与回测框架的耦合性
    """
    
    def __init__(self, initial_cash=10000000, realistic_trading=False, 
                 transaction_fee_rate=0.0003, slippage_rate=0.0001):
        """
        初始化交易管理器
        
        参数:
        initial_cash: 初始资金
        realistic_trading: 是否考虑交易费用和滑点影响
        transaction_fee_rate: 交易费率，默认为万分之三(0.0003)
        slippage_rate: 滑点比例，默认为万分之一(0.0001)
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}  # 持仓情况 {stock_id: volume}
        self.realistic_trading = realistic_trading
        self.transaction_fee_rate = transaction_fee_rate
        self.slippage_rate = slippage_rate
        
        # 用于记录交易成本
        self.total_transaction_fees = 0.0
        self.total_slippage_cost = 0.0
        self.trades = []  # 交易记录
    
    def calculate_transaction_cost(self, price, volume, is_buy=True):
        """
        计算交易成本，包括手续费和滑点
        
        参数:
        price: 交易价格
        volume: 交易量
        is_buy: 是否为买入交易，买入和卖出的滑点方向不同
        
        返回:
        交易手续费, 滑点成本
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
    
    def calculate_buy_cost(self, price, volume):
        """
        计算买入成本，包含滑点和手续费影响
        
        参数:
        price: 买入价格
        volume: 买入数量
        
        返回:
        总买入成本, 实际价格, 交易费用, 滑点成本
        """
        # 计算交易成本
        transaction_fee, slippage_cost = self.calculate_transaction_cost(price, volume, is_buy=True)
        
        # 应用滑点影响
        adjusted_price = price + slippage_cost / volume if volume > 0 else price
        
        # 计算买入成本
        buy_cost = volume * adjusted_price + transaction_fee
        
        return buy_cost, adjusted_price, transaction_fee, slippage_cost
    
    def calculate_sell_proceeds(self, price, volume):
        """
        计算卖出收入，包含滑点和手续费影响
        
        参数:
        price: 卖出价格
        volume: 卖出数量
        
        返回:
        实际卖出收入, 实际价格, 交易费用, 滑点成本
        """
        # 计算交易成本
        transaction_fee, slippage_cost = self.calculate_transaction_cost(price, volume, is_buy=False)
        
        # 应用滑点影响
        adjusted_price = price + slippage_cost / volume if volume > 0 else price
        
        # 计算卖出收入
        sell_amount = volume * adjusted_price - transaction_fee
        
        return sell_amount, adjusted_price, transaction_fee, slippage_cost
    
    def execute_buy(self, stock_id, price, volume, date, reason='BUY'):
        """
        执行买入操作
        
        参数:
        stock_id: 股票ID
        price: 买入价格
        volume: 买入数量
        date: 交易日期
        reason: 买入原因
        
        返回:
        成功买入的数量，如果资金不足可能小于请求的数量
        """
        if volume <= 0 or price <= 0:
            return 0
        
        # 计算买入成本
        buy_cost, adjusted_price, transaction_fee, slippage_cost = self.calculate_buy_cost(price, volume)
        
        # 检查资金是否足够
        if buy_cost > self.cash:
            # 资金不足，调整买入数量
            possible_volume = int((self.cash - transaction_fee) / (adjusted_price))
            possible_volume = possible_volume // 100 * 100  # 确保是100的整数倍
            
            if possible_volume <= 0:
                return 0
            
            # 重新计算买入成本
            buy_cost, adjusted_price, transaction_fee, slippage_cost = self.calculate_buy_cost(price, possible_volume)
            volume = possible_volume
        
        # 更新持仓和现金
        if stock_id not in self.positions:
            self.positions[stock_id] = 0
        self.positions[stock_id] += volume
        self.cash -= buy_cost
        
        # 更新总交易成本
        self.total_transaction_fees += transaction_fee
        self.total_slippage_cost += abs(slippage_cost)
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_id': stock_id,
            'action': 'BUY',
            'price': price,
            'adjusted_price': adjusted_price,
            'volume': volume,
            'transaction_fee': transaction_fee,
            'slippage_cost': slippage_cost,
            'reason': reason
        })
        
        return volume
    
    def execute_sell(self, stock_id, price, volume, date, reason='SELL'):
        """
        执行卖出操作
        
        参数:
        stock_id: 股票ID
        price: 卖出价格
        volume: 卖出数量
        date: 交易日期
        reason: 卖出原因
        
        返回:
        成功卖出的数量，如持仓不足则小于请求的数量
        """
        if volume <= 0 or price <= 0:
            return 0
        
        # 检查持仓是否足够
        if stock_id not in self.positions or self.positions[stock_id] <= 0:
            return 0
        
        # 调整卖出数量不超过持仓
        actual_volume = min(volume, self.positions[stock_id])
        
        # 计算卖出收入
        sell_amount, adjusted_price, transaction_fee, slippage_cost = self.calculate_sell_proceeds(price, actual_volume)
        
        # 更新持仓和现金
        self.positions[stock_id] -= actual_volume
        self.cash += sell_amount
        
        # 持仓为0时删除该股票
        if self.positions[stock_id] == 0:
            del self.positions[stock_id]
        
        # 更新总交易成本
        self.total_transaction_fees += transaction_fee
        self.total_slippage_cost += abs(slippage_cost)
        
        # 记录交易
        self.trades.append({
            'date': date,
            'stock_id': stock_id,
            'action': 'SELL',
            'price': price,
            'adjusted_price': adjusted_price,
            'volume': actual_volume,
            'transaction_fee': transaction_fee,
            'slippage_cost': slippage_cost,
            'reason': reason
        })
        
        return actual_volume
    
    def calculate_portfolio_value(self, date, get_price_func):
        """
        计算当前组合价值
        
        参数:
        date: 当前日期
        get_price_func: 获取股票价格的函数，接受股票ID和日期作为参数
        
        返回:
        组合总价值（现金 + 持仓市值）
        """
        portfolio_value = self.cash
        
        for stock_id, volume in self.positions.items():
            price = get_price_func(stock_id, date)
            if price is not None:
                portfolio_value += price * volume
        
        return portfolio_value
    
    def rebalance_portfolio(self, date, new_portfolio_stocks, prices_dict, get_price_func, test_stocks=None, predictions=None, historical_data=None, use_optimizer=True):
        """
        执行投资组合再平衡
        
        参数:
        date: 当前日期
        new_portfolio_stocks: 新持仓股票ID列表
        prices_dict: 股票价格字典 {stock_id: price}
        get_price_func: 获取股票价格的函数，接受股票ID和日期作为参数
        test_stocks: 预测所用的股票ID列表
        predictions: 对应的预测收益率数组
        historical_data: 历史价格数据字典 {stock_id: [historical_prices]}
        use_optimizer: 是否使用投资组合优化器，如果为False则等权重买入
        
        返回:
        交易执行信息字典
        """
        # 1. 卖出不在新持仓中的股票
        sold_stocks = []
        for stock_id, position in list(self.positions.items()):
            if stock_id not in new_portfolio_stocks and position > 0:
                price = get_price_func(stock_id, date)
                if price is not None and price > 0:
                    sold_volume = self.execute_sell(stock_id, price, position, date, reason='REBALANCE')
                    if sold_volume > 0:
                        sold_stocks.append(stock_id)
        
        # 2. 计算新增买入的股票
        existing_stocks = [s for s in new_portfolio_stocks if s in self.positions]
        new_stocks = [s for s in new_portfolio_stocks if s not in self.positions]
        
        # 3. 分配资金买入新增的股票
        buy_stocks = []
        
        if new_stocks:
            # 准备价格字典
            stock_prices = {}
            for stock_id in new_stocks:
                price = prices_dict.get(stock_id)
                if price is None:
                    price = get_price_func(stock_id, date)
                if price is not None and price > 0:
                    stock_prices[stock_id] = price
            
            if use_optimizer:
                # 使用投资组合优化器
                print("使用投资组合优化器分配资金")
                optimizer = PortfolioOptimizer(risk_aversion=1.0, max_weight=0.2, min_weight=0.01)
                
                # 准备预期收益率数据
                expected_returns = {}
                
                # 检查是否提供了test_stocks和predictions
                if test_stocks is not None and predictions is not None:
                    # 使用提供的预测数据
                    for stock_id in new_stocks:
                        try:
                            idx = test_stocks.index(stock_id)
                            expected_returns[stock_id] = predictions[idx]
                        except (ValueError, IndexError):
                            # 如果在test_stocks中找不到，使用默认值
                            expected_returns[stock_id] = 0.0
                else:
                    # 如果没有提供预测数据，使用等权重
                    for stock_id in new_stocks:
                        expected_returns[stock_id] = 1.0
                
                # 使用优化器计算权重
                weights = optimizer.optimize_weights(expected_returns, historical_data)
            else:
                # 使用等权重分配
                print("使用等权重分配资金")
                equal_weight = 1.0 / len(new_stocks)
                weights = {stock_id: equal_weight for stock_id in new_stocks}
            
            # 计算每只股票的资金分配和股数
            for stock_id, weight in weights.items():
                # 计算分配金额
                allocation = self.cash * weight
                price = stock_prices.get(stock_id)
                
                if price is None or price <= 0:
                    continue
                    
                # 计算可买股数（确保是100的整数倍）
                shares = int(allocation / (price * 100)) * 100
                
                if shares > 0:
                    bought_volume = self.execute_buy(
                        stock_id, 
                        price, 
                        shares, 
                        date, 
                        reason='REBALANCE'
                    )
                    if bought_volume > 0:
                        buy_stocks.append(stock_id)
            
            print(f"分配后剩余现金: {self.cash:.2f}")
        
        return {
            'date': date,
            'sold_stocks': sold_stocks,
            'bought_stocks': buy_stocks,
            'cash_remaining': self.cash,
            'total_portfolio_value': self.calculate_portfolio_value(date, get_price_func)
        }
    def get_high_return_portfolio(self, predictions, test_stocks, upper_pos, down_pos):
        """
        根据预测收益率构建投资组合，选择预测收益率最高的N只股票
        
        参数:
        predictions: 预测收益率数组
        test_stocks: 对应的股票ID列表
        target_position_count: 目标持仓数量
        
        返回:
        选中的股票ID列表
        """
        positions = np.zeros(len(test_stocks))
        stock_indices = {}  # 映射股票ID到索引
        for i, stock_id in enumerate(test_stocks):
                positions[i] = self.positions.get(stock_id, 0)
                stock_indices[stock_id] = i

        high_return, low_return = np.percentile(predictions, [upper_pos, down_pos])
        buy_signals = (positions == 0) & (predictions > high_return)
        
        # 生成卖出信号：有持仓且预测收益率低于低阈值
        sell_signals = (positions > 0) & (predictions < low_return)
        
        buy_stocks = [test_stocks[i] for i in range(len(test_stocks)) if buy_signals[i]]
        sell_stocks = [test_stocks[i] for i in range(len(test_stocks)) if sell_signals[i]]
        return buy_stocks
     