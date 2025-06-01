"""
Description: 投资组合收益率计算模块
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Union


class PortfolioPerformance:
    """
    投资组合收益率计算类
    
    跟踪每日持仓、计算收益率并考虑交易成本
    """
    
    def __init__(self, initial_cash: float = 1e8):
        """
        初始化投资组合性能计算器
        
        参数:
            initial_cash (float): 初始资金
        """
        self.initial_cash = initial_cash
        
        # 持仓和收益跟踪
        self.daily_holdings = {}  # 每日持仓 {date: {code: amount}}
        self.daily_values = {}    # 每日总市值 {date: value}
        self.daily_cash = {}      # 每日现金 {date: cash}
        self.daily_returns = []   # 每日收益率
        self.daily_costs = {}     # 每日交易成本 {date: cost}
        
        # 收益率序列
        self.returns_series = None
        self.benchmark_returns_series = None
        
        # 初始化第一天
        self.current_holdings = {}  # {code: amount}
        self.current_cash = initial_cash
        self.current_value = initial_cash
        
    def update_holdings(self, date, holdings: Dict[str, int], prices: Dict[str, float], 
                        cost: float = 0.0):
        """
        更新某一天的持仓信息
        
        参数:
            date: 日期
            holdings: 持仓字典 {code: volume}
            prices: 价格字典 {code: price}
            cost: 当日交易成本
        """
        # 更新持仓
        self.current_holdings = holdings.copy()
        
        # 计算持仓市值
        holdings_value = sum([holdings.get(code, 0) * prices.get(code, 0) for code in holdings])
        
        # 更新总市值和现金
        self.current_value = holdings_value + self.current_cash - cost
        self.current_cash -= cost
        
        # 记录到每日数据中
        self.daily_holdings[date] = holdings.copy()
        self.daily_values[date] = self.current_value
        self.daily_cash[date] = self.current_cash
        self.daily_costs[date] = cost
    
    def calculate_daily_return(self, date, previous_date):
        """
        计算某日的日收益率
        
        参数:
            date: 当前日期
            previous_date: 上一个日期
        
        返回:
            float: 日收益率
        """
        if previous_date not in self.daily_values:
            return 0.0
            
        previous_value = self.daily_values[previous_date]
        current_value = self.daily_values[date]
        
        if previous_value <= 0:
            return 0.0
            
        return current_value / previous_value - 1
    
    def update_for_trades(self, date, order: Dict, prices: Dict[str, float], 
                          cost_buy: float, cost_sell: float, min_cost: float, position: Dict):
        """
        处理交易并更新持仓和资金
        
        参数:
            date: 日期
            order: 订单 {'buy': {code: volume}, 'sell': {code: volume}}
            prices: 价格字典
            cost_buy: 买入成本率
            cost_sell: 卖出成本率
            min_cost: 最低交易成本
            position: 当前持仓
        """
        # 计算买入成本
        buy_value = sum([vol * prices.get(code, 0) for code, vol in order.get('buy', {}).items()])
        buy_cost = max(min_cost, buy_value * cost_buy) if buy_value > 0 else 0
        
        # 计算卖出收入和成本
        sell_value = sum([vol * prices.get(code, 0) for code, vol in order.get('sell', {}).items()])
        sell_cost = max(min_cost, sell_value * cost_sell) if sell_value > 0 else 0
        
        # 总交易成本
        total_cost = buy_cost + sell_cost
        
        # 更新现金
        self.current_cash = self.current_cash - buy_value - buy_cost + sell_value - sell_cost
        
        # 更新持仓
        new_position = position.copy()
        
        # 记录到每日数据中
        self.update_holdings(date, new_position, prices, total_cost)
        
        # 计算并记录日收益率
        dates = sorted(self.daily_values.keys())
        if len(dates) > 1:
            prev_date = dates[-2]
            daily_return = self.calculate_daily_return(date, prev_date)
            self.daily_returns.append(daily_return)
    
    def calculate_returns_series(self, dates: List):
        """
        计算收益率序列
        
        参数:
            dates: 日期列表
        
        返回:
            pd.Series: 收益率序列
        """
        returns_data = []
        
        for i, date in enumerate(dates):
            if i == 0:
                returns_data.append(0)  # 第一天收益率为0
            else:
                prev_date = dates[i-1]
                ret = self.calculate_daily_return(date, prev_date)
                returns_data.append(ret)
        
        self.returns_series = pd.Series(returns_data, index=dates)
        return self.returns_series
    
    def calculate_benchmark_returns(self, benchmark_data: pd.Series):
        """
        计算基准收益率序列
        
        参数:
            benchmark_data: 包含基准每日收益率的Series
        
        返回:
            pd.Series: 基准收益率序列
        """
        self.benchmark_returns_series = benchmark_data
        return self.benchmark_returns_series
    
    def calculate_performance_metrics(self):
        """
        计算各种性能指标
        
        返回:
            Dict: 包含各项指标的字典
        """
        if self.returns_series is None:
            raise ValueError("请先计算收益率序列")
        
        # 累积收益率
        cum_returns = (1 + self.returns_series).cumprod() - 1
        
        # 年化收益率
        days = len(self.returns_series)
        annual_return = (1 + cum_returns.iloc[-1]) ** (252 / days) - 1
        
        # 计算最大回撤
        wealth_index = (1 + self.returns_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()
        
        # 计算夏普比率
        daily_rf = 0  # 假设无风险利率为0
        excess_returns = self.returns_series - daily_rf
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * (252 ** 0.5) if excess_returns.std() > 0 else 0
        
        # 计算总交易成本
        total_cost = sum(self.daily_costs.values())
        
        # 超额收益率 (相对于基准)
        if self.benchmark_returns_series is not None:
            # 确保长度一致
            min_len = min(len(self.returns_series), len(self.benchmark_returns_series))
            excess_return = self.returns_series.iloc[:min_len].mean() - self.benchmark_returns_series.iloc[:min_len].mean()
            benchmark_cum_return = (1 + self.benchmark_returns_series).cumprod().iloc[-1] - 1
        else:
            excess_return = None
            benchmark_cum_return = None
        
        metrics = {
            "total_return": cum_returns.iloc[-1] * 100,  # 转为百分比
            "annual_return": annual_return * 100,  # 转为百分比
            "max_drawdown": max_drawdown * 100,  # 转为百分比
            "sharpe_ratio": sharpe_ratio,
            "total_cost": total_cost,
            "excess_return": excess_return * 100 if excess_return is not None else None,  # 转为百分比
            "benchmark_return": benchmark_cum_return * 100 if benchmark_cum_return is not None else None  # 转为百分比
        }
        
        return metrics
    
    def get_daily_holdings_value(self, returns_data: pd.DataFrame):
        """
        计算每日每只股票的持仓市值和收益贡献
        
        参数:
            returns_data: 包含每日每只股票收益率的DataFrame
        
        返回:
            pd.DataFrame: 每日每只股票的持仓市值和收益贡献
        """
        results = []
        
        for date in sorted(self.daily_holdings.keys()):
            holdings = self.daily_holdings[date]
            
            # 从returns_data中获取当日收益率
            try:
                daily_returns = returns_data.xs(date, level=0)
            except KeyError:
                continue
            
            for code, amount in holdings.items():
                # 获取该股票的收益率
                try:
                    stock_return = daily_returns.loc[code]['returns']
                except (KeyError, ValueError):
                    stock_return = 0
                
                # 计算收益贡献
                position_value = amount  # 这里需要乘以价格，但我们没有直接存储价格
                return_contribution = position_value * stock_return
                
                results.append({
                    'date': date,
                    'code': code,
                    'position': amount,
                    'position_value': position_value,
                    'stock_return': stock_return,
                    'return_contribution': return_contribution
                })
        
        return pd.DataFrame(results)