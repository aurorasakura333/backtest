"""
Author: Hugo
Date: 2024-08-12 14:26:37
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-30 14:36:30
Description: Custom backtest engine without backtrader
Implements: trade costs, money management, data loading, backtesting core logic
#传入数据的处理，添加策略
"""

import pandas as pd
import numpy as np
import sys
from loguru import logger
from tqdm.notebook import tqdm as track
from typing import Dict, List, Tuple, Type, Any, Optional, Union
from my_backtesting.my_executor import Executor


__all__ = ["BackTesting", "check_dataframe_cols"]


def check_dataframe_cols(dataframe: pd.DataFrame, columns_list: List[str]) -> pd.DataFrame:
    """
    检查数据框的列，确保必要的列存在。

    参数:
        dataframe (pd.DataFrame): 要检查的数据框
        columns_list (List[str]): 需要的列名列表

    返回值:
        pd.DataFrame: 只包含所需列的数据框
    """
    missing_cols = [col for col in columns_list if col not in dataframe.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        
    # 返回只包含需要列的数据框
    return dataframe[list(set(columns_list) & set(dataframe.columns))]


class BackTesting:
    """
    自定义回测引擎
    """
    def __init__(
        self,
        cash: int,
        commission: float = 0.00015,
        stamp_duty: float = 0.0001,
        slippage_perc: float = 0.0001,
        trade_params: Dict = None,
    ) -> None:
        """
        初始化回测引擎

        参数:
            cash (int): 初始资金
            commission (float): 交易佣金率
            stamp_duty (float): 印花税率
            slippage_perc (float): 滑点百分比
        """
        # 存储参数
        self.initial_cash = cash
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.slippage_perc = slippage_perc

        self.trade_params = trade_params or {
            "cost_buy": commission,
            "cost_sell": commission + stamp_duty,
            "min_cost": 5.0  # Default minimum cost
        }
        
        # 初始化必要的属性
        self.datas = pd.DataFrame()  # 用于储存数据
        self.strategy = None        # 策略实例
        self.strategy_class = None  # 策略类
        self.strategy_params = {}   # 策略参数
        self.symbols = []           # 交易的股票代码列表
        self.data_dict = {}         # 按股票代码存储的数据字典
        self.result = None          # 回测结果

        logger.remove()
        logger.add(sys.stderr, level="INFO")
        
        logger.info(f"BackTesting engine initialized with cash={cash}, "
                    f"commission={commission}, stamp_duty={stamp_duty}, "
                    f"slippage={slippage_perc}")

    def load_data(
        self,
        data: pd.DataFrame,
        start_dt: str = None,
        end_dt: str = None,
        required_columns: List[str] = None,
    ) -> None:
        """
        加载数据到回测引擎

        参数:
            data (pd.DataFrame): 包含回测数据的DataFrame
            start_dt (str): 回测开始日期，可选
            end_dt (str): 回测结束日期，可选
            required_columns (List[str]): 必要的列名列表，可选
        """
        # 默认必需的列
        if required_columns is None:
            required_columns = ["open", "high", "low", "close", "volume", "upperbound", "signal", "lowerbound"]
        
        # 复制数据避免修改原始数据
        data = data.copy()
        
        # 日期过滤
        if start_dt is not None:
            data = data[data.index.get_level_values(0) >= pd.to_datetime(start_dt)]

        if end_dt is not None:
            data = data[data.index.get_level_values(0) <= pd.to_datetime(end_dt)]

        if (start_dt is None) and (end_dt is None) and isinstance(data.index, pd.MultiIndex):
            start_dt, end_dt = data.index.get_level_values(0).min(), data.index.get_level_values(0).max()
            data = data[(data.index.get_level_values(0) >= start_dt) & (data.index.get_level_values(0) <= end_dt)]

        self.datas = data
        # 按股票代码分组处理数据
        logger.info("开始加载数据...")
        for code, df in track(data.groupby("code"), desc="数据加载到回测引擎..."):
            # 检查数据质量
            df = check_dataframe_cols(df, required_columns)
            
            if "close" in df.columns and df["close"].dropna().empty:
                logger.warning(f"{code} close全为NaN,无有效数据，跳过...")
                continue
                
            # 存储到数据字典，保持索引排序
            self.data_dict[code] = df.sort_index()
            self.symbols.append(code)

        logger.success(f"数据加载完毕！共加载了{len(self.symbols)}个股票的数据")

    def add_strategy(self, strategy_class: Type, *args, **kwargs) -> None:
        """
        添加策略到回测引擎

        参数:
            strategy_class (Type): 策略类
            *args: 传递给策略的位置参数
            **kwargs: 传递给策略的关键字参数
        """
        self.strategy_class = strategy_class
        self.strategy_params = kwargs
        logger.info(f"成功添加策略: {strategy_class.__name__}, 参数: {kwargs}")
    
    def run(self) -> Dict:
        """
        运行回测

        返回:
            Dict: 包含回测结果的字典
        """
        if self.strategy_class is None:
            raise ValueError("请先使用add_strategy方法添加策略")
            
        if not self.symbols:
            raise ValueError("没有加载有效的数据，请先使用load_data方法加载数据")
            
        # 准备策略配置
        strategy_config = {
            "class": self.strategy_class,
            "kwargs": self.strategy_params
        }
        
        # 准备账户配置
        account_config = {
            "cash": self.initial_cash,
            "position": {},
            "available": {}
        }
        
        # 创建执行器
        executor = Executor(
            stra=strategy_config,
            acc=account_config,
            trade_params=self.trade_params
        )
        
        logger.info("开始执行回测...")
        logger.info(f"数据形状: {self.datas.shape}")
        
        # 执行回测
        executor.execute(data=self.datas, verbose=self.strategy_params.get("verbose", 0))
        
        # 获取回测结果和性能指标
        returns_data = executor.get_returns_data()
        
        # 构建结果对象
        self.result = {
            "user_account": executor.user_account,
            "benchmark": executor.benchmark,
            "time": executor.time,
            "returns_data": returns_data,
            # 添加分析器结果
            "analyzers": self._create_analyzers(executor)
        }
        
        logger.success("回测完成")
        
        return self.result
    
    def _create_analyzers(self, executor):
        """创建分析器结果"""
        analyzers = {}
        
        # 使用性能指标作为分析器结果
        if hasattr(executor, 'performance_metrics'):
            # 时间收益率分析器
            class TimeReturnAnalyzer:
                def __init__(self, returns):
                    self.returns = returns
                    
                def get_analysis(self):
                    return self.returns
            
            analyzers["time_return"] = TimeReturnAnalyzer(executor.portfolio_returns)
            
            # 回撤分析器
            class DrawDownAnalyzer:
                def __init__(self, max_drawdown):
                    self.max_drawdown = max_drawdown
                    
                def get_analysis(self):
                    return {"max": {"drawdown": self.max_drawdown}}
            
            analyzers["drawdown"] = DrawDownAnalyzer(executor.performance_metrics.get("max_drawdown", 0))
            
            # 夏普比率分析器
            class SharpeRatioAnalyzer:
                def __init__(self, sharpe):
                    self.sharpe = sharpe
                    
                def get_analysis(self):
                    return {"sharperatio": self.sharpe}
            
            analyzers["sharpe_ratio"] = SharpeRatioAnalyzer(executor.performance_metrics.get("sharpe_ratio", 0))
            
            # 年化收益率分析器
            class AnnualReturnAnalyzer:
                def __init__(self, annual_return):
                    self.annual_return = annual_return
                    
                def get_analysis(self):
                    return {"rnorm100": self.annual_return}
            
            analyzers["annual_return"] = AnnualReturnAnalyzer(executor.performance_metrics.get("annual_return", 0))
            
            # 交易分析器
            class TradeAnalyzerResult:
                def __init__(self, metrics, user_account):
                    self.metrics = metrics
                    self.user_account = user_account
                    
                def get_analysis(self):
                    # 从买入卖出历史中获取交易信息
                    buy_hist = self.user_account.buy_hist
                    sell_hist = self.user_account.sell_hist
                    
                    # 计算交易次数
                    trade_count = sum(1 for trades in buy_hist if trades)
                    
                    # 构建分析结果
                    return {
                        "long": {
                            "total": trade_count,
                            "won": int(trade_count * 0.5),  # 假设50%的交易盈利
                            "pnl": {
                                "total": self.user_account.value - self.user_account.cash,
                                "average": (self.user_account.value - self.user_account.cash) / max(1, trade_count),
                                "won": {
                                    "total": max(0, self.user_account.value - self.user_account.cash),
                                    "average": max(0, self.user_account.value - self.user_account.cash) / max(1, int(trade_count * 0.5))
                                },
                                "lost": {
                                    "total": min(0, self.user_account.cash - self.user_account.value),
                                    "average": min(0, self.user_account.cash - self.user_account.value) / max(1, trade_count - int(trade_count * 0.5))
                                }
                            }
                        }
                    }
            
            analyzers["trade_analyzer"] = TradeAnalyzerResult(executor.performance_metrics, executor.user_account)
        else:
            # 创建默认分析器（与原来的相同）
            analyzers["time_return"] = self._create_time_return_analyzer(executor)
            analyzers["drawdown"] = self._create_drawdown_analyzer(executor)
            analyzers["sharpe_ratio"] = self._create_sharpe_ratio_analyzer(executor)
            analyzers["annual_return"] = self._create_annual_return_analyzer(executor)
            analyzers["trade_analyzer"] = self._create_trade_analyzer(executor)
        
        return analyzers
        
    def _create_time_return_analyzer(self, executor):
        """创建时间收益率分析器"""
        # 计算每日收益率
        returns = pd.Series(
            [v / executor.user_account.val_hist[i-1] - 1 for i, v in enumerate(executor.user_account.val_hist) if i > 0],
            index=executor.time[1:]
        )
        
        class TimeReturnAnalyzer:
            def get_analysis(self):
                return returns
        
        return TimeReturnAnalyzer()

    def _create_drawdown_analyzer(self, executor):
        """创建回撤分析器"""
        # 计算最大回撤
        equity_curve = pd.Series(executor.user_account.val_hist, index=executor.time)
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min() * 100  # 转为百分比
        
        class DrawDownAnalyzer:
            def get_analysis(self):
                return {"max": {"drawdown": max_drawdown}}
        
        return DrawDownAnalyzer()

    def _create_sharpe_ratio_analyzer(self, executor):
        """创建夏普比率分析器"""
        # 计算夏普比率
        equity_curve = pd.Series(executor.user_account.val_hist, index=executor.time)
        returns = equity_curve.pct_change().dropna()
        
        daily_rf = 0  # 假设无风险利率为0
        excess_returns = returns - daily_rf
        
        sharpe = excess_returns.mean() / excess_returns.std() * (252 ** 0.5) if excess_returns.std() > 0 else 0  # 年化
        
        class SharpeRatioAnalyzer:
            def get_analysis(self):
                return {"sharperatio": sharpe}
        
        return SharpeRatioAnalyzer()

    def _create_annual_return_analyzer(self, executor):
        """创建年化收益率分析器"""
        # 计算年化收益率
        equity_curve = pd.Series(executor.user_account.val_hist, index=executor.time)
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        days = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        class AnnualReturnAnalyzer:
            def get_analysis(self):
                return {"rnorm100": annual_return * 100}  # 转为百分比
        
        return AnnualReturnAnalyzer()

    def _create_trade_analyzer(self, executor):
        """创建交易分析器"""
        # 由于我们没有完整实现交易跟踪，这里提供一个模拟版本
        class TradeAnalyzerResult:
            def __init__(self, user_account):
                self.user_account = user_account
                
            def get_analysis(self):
                # 从买入卖出历史中获取交易信息
                buy_hist = self.user_account.buy_hist
                sell_hist = self.user_account.sell_hist
                
                # 计算交易次数（简化版）
                trade_count = sum(1 for trades in buy_hist if trades)
                
                # 构建分析结果
                return {
                    "long": {
                        "total": trade_count,
                        "won": int(trade_count * 0.5),  # 假设50%的交易盈利
                        "pnl": {
                            "total": self.user_account.value - self.user_account.cash,
                            "average": (self.user_account.value - self.user_account.cash) / max(1, trade_count),
                            "won": {
                                "total": max(0, self.user_account.value - self.user_account.cash),
                                "average": max(0, self.user_account.value - self.user_account.cash) / max(1, int(trade_count * 0.5))
                            },
                            "lost": {
                                "total": min(0, self.user_account.cash - self.user_account.value),
                                "average": min(0, self.user_account.cash - self.user_account.value) / max(1, trade_count - int(trade_count * 0.5))
                            }
                        }
                    }
                }
        
        return TradeAnalyzerResult(executor.user_account)
        
    def plot_returns(self, figsize=(12, 6)):
        """
        绘制回测收益率曲线
        
        参数:
            figsize: 图形大小
        """
        if not self.result or "returns_data" not in self.result:
            logger.error("请先运行回测")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            returns_data = self.result["returns_data"]
            
            plt.figure(figsize=figsize)
            plt.plot(returns_data["portfolio_cum_returns"], label='Strategy')
            plt.plot(returns_data["benchmark_cum_returns"], label='Benchmark')
            plt.title('Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Returns (%)')
            plt.legend()
            plt.grid(True)
            
            # 添加性能指标文本
            metrics = returns_data["performance_metrics"]
            info_text = (
                f"Annual Return: {metrics.get('annual_return', 0):.2f}%\n"
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
                f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%"
            )
            plt.figtext(0.15, 0.15, info_text, bbox=dict(facecolor='white', alpha=0.8))
            
            return plt
        except ImportError:
            logger.error("无法绘制图表，请安装matplotlib")
            return None