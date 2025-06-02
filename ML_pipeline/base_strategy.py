"""
base_strategy.py
基本策略类的定义，包含策略的抽象基类和数据处理的基本函数
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import datetime


class DataProcessor:
    """
    数据处理器，负责数据的预处理、特征工程等
    """
    
    @staticmethod
    def filter_MAD(df, factor, n=3):
        """
        中位数去极值法
        
        参数:
        df: 待处理的DataFrame
        factor: 待去极值的因子名
        n: 中位数偏差值的上下界倍数
        
        返回:
        处理后的DataFrame
        """
        median = df[factor].median()
        mad = (df[factor] - median).abs().median()
        
        max_range = median + n * mad
        min_range = median - n * mad
        
        df_copy = df.copy()
        df_copy.loc[df_copy[factor] > max_range, factor] = max_range
        df_copy.loc[df_copy[factor] < min_range, factor] = min_range
        
        return df_copy
    
    @staticmethod
    def standardize_factors(df, factors):
        """
        对因子进行标准化处理
        
        参数:
        df: 待处理的DataFrame
        factors: 因子列表
        
        返回:
        处理后的DataFrame
        """
        result = df.copy()
        
        # 首先进行去极值
        for factor in factors:
            result = DataProcessor.filter_MAD(result, factor, 5)
        
        # 然后进行标准化
        scaler = StandardScaler()
        result[factors] = scaler.fit_transform(result[factors])
        
        return result
    
    @staticmethod
    def prepare_data(df, factor_codes, lookback_period=21, daily_returns=False):
        """
        准备机器学习模型的训练和测试数据
        
        参数:
        df: 输入的DataFrame，包含日期、股票ID和因子数据
        factor_codes: 使用的因子列表
        lookback_period: 回溯期，默认为21天（约一个月）
        daily_returns: 是否计算每日收益率，默认为False（计算整个回溯期的收益率）
        
        返回:
        训练特征、训练标签、测试特征、测试股票ID
        """
        # 按日期和股票ID分组处理数据
        result = []
        # 实现具体的数据准备逻辑
        
        return result


class MLStrategy(ABC):
    """
    机器学习策略抽象基类，定义了机器学习策略的基本接口
    """
    
    def __init__(self, factor_codes, lookback_period=21, upper_pos=80, down_pos=20, cash_rate=0.6):
        """
        初始化策略参数
        
        参数:
        factor_codes: 使用的因子列表
        lookback_period: 回溯期，默认为21天（约一个月）
        upper_pos: 股票预测收益率的上分位数，高于则买入，默认为80
        down_pos: 股票预测收益率的下分位数，低于则卖出，默认为20
        cash_rate: 计算可用资金比例的分子，默认为0.6
        """
        self.factor_codes = factor_codes
        self.lookback_period = lookback_period
        self.upper_pos = upper_pos
        self.down_pos = down_pos
        self.cash_rate = cash_rate
        self.model = None
    
    @abstractmethod
    def train(self, X, y):
        """
        训练模型
        
        参数:
        X: 训练特征
        y: 训练标签
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        预测
        
        参数:
        X: 预测特征
        
        返回:
        预测结果
        """
        pass
    
    def get_trading_signals(self, predictions, positions, threshold_high, threshold_low):
        """
        根据预测结果和持仓情况生成交易信号
        
        参数:
        predictions: 预测结果
        positions: 当前持仓
        threshold_high: 高阈值
        threshold_low: 低阈值
        
        返回:
        买入信号和卖出信号
        """
        # 获取预测收益率的高分位数和低分位数
        high_return = np.percentile(predictions, threshold_high)
        low_return = np.percentile(predictions, threshold_low)
        
        # 生成买入和卖出信号
        buy_signals = (positions == 0) & (predictions > high_return)
        sell_signals = (positions > 0) & (predictions < low_return)
        
        return buy_signals, sell_signals
    
    def adjust_position(self, buy_signals, sell_signals, positions, available_cash, prices):
        """
        根据交易信号调整仓位
        
        参数:
        buy_signals: 买入信号
        sell_signals: 卖出信号
        positions: 当前持仓
        available_cash: 可用资金
        prices: 当前价格
        
        返回:
        调整后的仓位和可用资金
        """
        # 实现根据信号调整仓位的逻辑
        # 卖出操作
        for i in np.where(sell_signals)[0]:
            available_cash += positions[i] * prices[i]
            positions[i] = 0
        
        # 买入操作
        buy_count = sum(buy_signals)
        if buy_count > 0:
            cash_per_stock = available_cash * self.cash_rate / buy_count
            for i in np.where(buy_signals)[0]:
                shares = int((cash_per_stock // (prices[i] * 100)) * 100)  # 确保是100的整数倍
                if shares > 0:
                    cost = shares * prices[i]
                    positions[i] += shares
                    available_cash -= cost
        
        return positions, available_cash


class SignalGenerator:
    """
    信号生成器，负责根据策略生成交易信号
    """
    
    @staticmethod
    def three_line_signal(close_data, win5=5, win20=20, win60=60):
        """
        三均线择时策略生成信号
        
        参数:
        close_data: 收盘价数据
        win5: 短期均线窗口，默认为5
        win20: 中期均线窗口，默认为20
        win60: 长期均线窗口，默认为60
        
        返回:
        买入信号和卖出信号
        """
        # 计算均线
        ma5 = close_data[:, -win5:].mean(axis=1)
        ma20 = close_data[:, -win20:].mean(axis=1)
        ma60 = close_data[:, -win60:].mean(axis=1)
        
        # 生成买入和卖出信号
        buy_signals = (ma5 > ma60) & (ma20 > ma60)
        sell_signals = (ma5 < ma60) & (ma20 < ma60)
        
        return buy_signals, sell_signals
    
    @staticmethod
    def combine_signals(ml_buy, ml_sell, time_buy, time_sell):
        """
        组合机器学习信号和择时信号
        
        参数:
        ml_buy: 机器学习买入信号
        ml_sell: 机器学习卖出信号
        time_buy: 择时买入信号
        time_sell: 择时卖出信号
        
        返回:
        组合后的买入和卖出信号
        """
        # 组合信号：机器学习信号和择时信号都为买入/卖出时才触发
        combined_buy = ml_buy & time_buy
        combined_sell = ml_sell & time_sell
        
        return combined_buy, combined_sell


class BacktestFramework:
    """
    回测框架，负责运行回测
    """
    
    def __init__(self, data_source, strategy, start_date, end_date, initial_cash=10000000):
        """
        初始化回测框架
        
        参数:
        data_source: 数据源
        strategy: 策略实例
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_cash: 初始资金
        """
        self.data_source = data_source
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.positions = {}
        self.cash = initial_cash
        self.portfolio_value = []
        self.dates = []
    
    def run(self):
        """
        运行回测
        """
        # 实现回测逻辑
        pass
    
    def get_performance_metrics(self):
        """
        计算回测绩效指标
        
        返回:
        回测绩效指标
        """
        # 计算各种回测绩效指标
        pass
    
    def plot_results(self):
        """
        绘制回测结果
        """
        # 绘制回测结果图表
        pass