"""
更新后的执行器模块，支持直接接收策略类并执行回测。
添加了收益率计算功能，考虑交易成本和滑点。
"""


from my_backtesting import my_account as account
from my_backtesting import my_strategy
from my_backtesting.my_performance import PortfolioPerformance
import pandas as pd
import numpy as np
import warnings
from loguru import logger
from typing import Dict, Type, Any, Union, List

warnings.filterwarnings("ignore")

def get_daily_inter(data: pd.DataFrame, shuffle=False):
    """获取每日数据的索引和计数"""
    daily_count = data.groupby(level=0).size().values 
    daily_index = np.roll(np.cumsum(daily_count), 1) #每日数据的起始索引
    daily_index[0] = 0
    if shuffle:
        daily_shuffle = list(zip(daily_index, daily_count))
        np.random.shuffle(daily_shuffle)
        daily_index, daily_count = zip(*daily_shuffle)
    return daily_index, daily_count


class Executor:
    def __init__(self, stra: Dict, acc: Dict, trade_params: Dict):
        """
        初始化执行器
        
        参数:
            stra (Dict): 策略配置，包含class和kwargs
            acc (Dict): 账户配置
            trade_params (Dict): 交易参数配置
        """
        # 初始化账户参数
        if acc is None:
            acc = {}
        keys = acc.keys()
        if "cash" not in keys:
            acc["cash"] = 1e8
        if "position" not in keys:
            acc["position"] = {}
        if "available" not in keys:
            acc["available"] = {}

        self.init_cash: float = acc['cash']
        self.position: dict = acc['position']
        self.value_hold: float = 0.0
        self.available: dict = acc['available']
        self.ben_cash: float = acc['cash']

        self.price = None
        self.time = []

        self.user_account = None
        self.benchmark = None
        self.cost_buy = trade_params["cost_buy"]
        self.cost_sell = trade_params["cost_sell"]
        self.min_cost = trade_params["min_cost"]
        
        # 初始化收益率计算器
        self.portfolio_performance = PortfolioPerformance(initial_cash=self.init_cash)
        self.benchmark_performance = PortfolioPerformance(initial_cash=self.ben_cash)

        # 初始化策略
        strategy_class = stra["class"]
        kwargs = stra["kwargs"]
        
        # 支持两种方式：1. 直接传入策略类 2. 传入策略名称字符串从strategy模块导入
        if isinstance(strategy_class, type):
            # 直接实例化传入的策略类
            self.s = strategy_class(kwargs)
        else:
            # 从strategy模块中获取策略类
            self.s = getattr(my_strategy, strategy_class)(kwargs)
        
        logger.info(f"初始化策略: {strategy_class.__name__ if isinstance(strategy_class, type) else strategy_class}")

    def init_account(self, data: pd.DataFrame):
        """
        初始化账户
        
        参数:
            data (pd.DataFrame): 包含价格数据的DataFrame，索引为[('time', 'code')]
        """
        data_copy = data.copy()
        t0 = data_copy.index.get_level_values(0)[0]
        code = data_copy[data_copy.index.get_level_values(0) == t0].index.get_level_values(1).values
        
        # 获取价格列名
        price_col = 'price' if 'price' in data_copy.columns else 'close'
        price0 = data_copy[data_copy.index.get_level_values(0) == t0][price_col]
        
        price_zip = zip(code, price0)
        self.price = dict(price_zip)
        
        if self.position is None:  # 如果没有position自然也没有available, 将它们初始化为0
            zero_list = [0 for _ in range(len(code))]
            position_zip, available_zip = zip(code, zero_list), zip(code, zero_list)
            self.position = dict(position_zip)
            self.available = dict(available_zip)

    def create_account(self):
        """创建账户实例"""
        self.user_account = account.Account(self.init_cash, self.position, self.available, self.price)
        self.benchmark = account.Account(self.ben_cash, {}, {}, self.price.copy())

    def get_cash_available(self):
        """
        计算可用资金
        
        Returns:
            float: 可用资金
        """
        return self.user_account.value * self.s.risk_degree

    def execute(self, data: pd.DataFrame, verbose: int = 0):
        """
        执行回测
        
        参数:
            data (pd.DataFrame): 包含交易数据的DataFrame
            verbose (int): 是否输出详细信息 (0=不输出, 1=输出)
        """
        def check_names(data, predict_col='signal', price_col='price'):
            """检查数据格式"""

            names = data.index.names
            time_idx = None
            code_idx = None
            
            # 兼容time或date作为第一个索引名
            if names[0] in ['time', 'date']:
                time_idx = 0
            if names[1] == 'code':
                code_idx = 1
                
            if time_idx is None or code_idx is None:
                raise ValueError(f"索引应为[('time'/'date', 'code')]，但得到了{names}")
                
            # 检查至少有一个信号列存在
            if predict_col not in data.columns and 'predict' not in data.columns:
                logger.warning(f"data should include column {predict_col} or 'predict'")
                
            # 检查至少有一个价格列存在    
            if 'close' not in data.columns:
                raise ValueError("data should include column 'close'")
                
            # 如果没有R列（收益率），创建一个全0的列
            if 'returns' not in data.columns:
                data['returns'] = 0
                logger.warning("No 'returns' column found, created a zero-filled column")

        try:
            # 检查数据格式
            check_names(data = data)
            
            # 初始化账户
            self.init_account(data)
            self.create_account()
            
            # 获取基准收益率
            if 'returns' in data.columns:
                benchmark = data["returns"].groupby(level=0).transform(lambda x: x.mean())  # 大盘收益率
            else:
                # 如果没有returns列，创建一个全0的Series作为基准收益率
                benchmark = pd.Series(0, index=data.index)
                logger.warning("No 'returns' column found, using zero returns as benchmark")
            
            # 按日期分组数据
            daily_idx, daily_count = get_daily_inter(data)
            
            # 确保数据按日期排序
            data = data.sort_index(level=0)
            benchmark = benchmark.sort_index(level=0)
            
            # 遍历每个交易日
            for idx, count in zip(daily_idx, daily_count):
                try:
                    # 获取当天的数据批次
                    batch = slice(idx, idx + count)
                    data_batch = data.iloc[batch]
                    logger.info(f"数据批次形状: {data_batch.shape}")
                    
                    benchmark_batch = benchmark.iloc[batch]
                    
                    # 获取当前日期
                    if len(data_batch.index) > 0:
                        current_day = data_batch.index.get_level_values(0)[0]
                        self.date = current_day
                        logger.info(f"当前交易日: {current_day}")
                        self.time.append(current_day)
                    else:
                        logger.warning("当前批次数据为空")
                        continue
                    
                    # 打印当前持仓信息
                    logger.info(f"当前持仓: {self.user_account.position}")
                    logger.info(f"当前可用资金: {self.get_cash_available()}")
                                        
                    day_buy_signals = {}
                    day_sell_signals = {}
                    current_price = {}
                    logger.info("开始生成交易信号...")
                    try:
                        order, current_price = self.s.to_signal(
                            data = data_batch, 
                            position=self.user_account.position,
                            cash_available=self.get_cash_available()
                        )
                        day_buy_signals.update(order['buy'])
                        day_sell_signals.update(order['sell'])
                        logger.info(f"生成的订单: buy={order['buy']}, sell={order['sell']}")
                    except Exception as e:
                        logger.error(f"生成交易信号时出错: {str(e)}")
                        order = {"buy": {}, "sell": {}}

                    # 检查订单是否有足够的资金完成
                    logger.info("检查订单资金...")
                    try:
                        order = self.user_account.check_order(
                            order, 
                            current_price, 
                            cost_rate=self.cost_buy,
                            min_cost=self.min_cost,
                            risk=self.s.risk_degree
                        )
                        logger.info(f"调整后的订单: buy={order['buy']}, sell={order['sell']}")
                    except Exception as e:
                        logger.error(f"检查订单资金时出错: {str(e)}")
                        order = {"buy": {}, "sell": {}}
                
                    # 输出交易信息
                    if verbose == 1:
                        print(current_day, '\n', "cash_available:", self.get_cash_available(), '\n',
                              "num_hold:", len((self.user_account.position)), '\n',
                              "buy:", '\n', order["buy"], '\n', "sell:", order["sell"], '\n')

                    # 计算交易成本
                    buy_value = sum([vol * current_price.get(code, 0) for code, vol in order.get('buy', {}).items()])
                    buy_cost = max(self.min_cost, buy_value * self.cost_buy) if buy_value > 0 else 0
                    
                    sell_value = sum([vol * current_price.get(code, 0) for code, vol in order.get('sell', {}).items()])
                    sell_cost = max(self.min_cost, sell_value * self.cost_sell) if sell_value > 0 else 0
                    
                    total_cost = buy_cost + sell_cost

                    # 执行交易
                    self.user_account.update_all(
                        order=order, 
                        price=current_price, 
                        cost_buy=self.cost_buy,
                        cost_sell=self.cost_sell, 
                        min_cost=self.min_cost
                    )
                    
                    # 风险控制
                    self.user_account.risk_control(
                        risk_degree=self.s.risk_degree, 
                        cost_rate=self.cost_sell,
                        min_cost=self.min_cost
                    )
                    
                    # 更新投资组合性能计算
                    self.portfolio_performance.update_for_trades(
                        date=current_day,
                        order=order,
                        prices=current_price,
                        cost_buy=self.cost_buy,
                        cost_sell=self.cost_sell,
                        min_cost=self.min_cost,
                        position=self.user_account.position
                    )

                    # 更新基准账户
                    if len(benchmark_batch) > 0:
                        benchmark_value = benchmark_batch.values[0]
                    else:
                        benchmark_value = 0
                    self.benchmark.value *= 1 + benchmark_value  # 乘上1+大盘收益率
                    self.benchmark.val_hist.append(self.benchmark.value)
                    
                    # 更新基准性能计算
                    self.benchmark_performance.update_holdings(
                        date=current_day,
                        holdings={},  # 基准没有具体持仓
                        prices={},
                        cost=0.0
                    )
                    
                except Exception as e:
                    logger.error(f"处理交易日时出错: {str(e)}")
                    logger.error(f"错误详情: {e.__class__.__name__}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # 继续处理下一个交易日
                    continue
                    
            # 计算投资组合和基准的收益率序列
            self.portfolio_returns = self.portfolio_performance.calculate_returns_series(self.time)
            
            # 提取基准收益率序列
            benchmark_returns = []
            for date in self.time:
                date_data = data[data.index.get_level_values(0) == date]
                if len(date_data) > 0:
                    mean_return = date_data['returns'].mean()
                    benchmark_returns.append(mean_return)
                else:
                    benchmark_returns.append(0)
            
            self.benchmark_returns = pd.Series(benchmark_returns, index=self.time)
            self.portfolio_performance.calculate_benchmark_returns(self.benchmark_returns)
            
            # 计算性能指标
            self.performance_metrics = self.portfolio_performance.calculate_performance_metrics()
            
            logger.info(f"回测完成，共{len(self.time)}个交易日")
            logger.info(f"性能指标: {self.performance_metrics}")
            
            return self.user_account, self.benchmark
            
        except Exception as e:
            logger.error(f"执行回测时出错: {str(e)}")
            logger.error(f"错误详情: {e.__class__.__name__}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 确保即使出错也返回账户对象
            if not hasattr(self, 'user_account') or self.user_account is None:
                self.init_account(data)
                self.create_account()
            if not hasattr(self, 'benchmark') or self.benchmark is None:
                self.benchmark = account.Account(self.ben_cash, {}, {}, self.price.copy())
                
            return self.user_account, self.benchmark
            
    def get_returns_data(self):
        """
        获取回测收益率数据
        
        返回:
            Dict: 包含收益率和性能指标的字典
        """
        return {
            "portfolio_returns": self.portfolio_returns,
            "benchmark_returns": self.benchmark_returns,
            "portfolio_cum_returns": (1 + self.portfolio_returns).cumprod() - 1,
            "benchmark_cum_returns": (1 + self.benchmark_returns).cumprod() - 1,
            "performance_metrics": self.performance_metrics
        }
        
    def plot_returns(self, figsize=(12, 6)):
        """
        绘制收益率曲线
        
        参数:
            figsize: 图形大小
        """
        try:
            import matplotlib.pyplot as plt
            
            # 计算累积收益率
            portfolio_cum_returns = (1 + self.portfolio_returns).cumprod() - 1
            benchmark_cum_returns = (1 + self.benchmark_returns).cumprod() - 1
            
            plt.figure(figsize=figsize)
            plt.plot(portfolio_cum_returns, label='Portfolio')
            plt.plot(benchmark_cum_returns, label='Benchmark')
            plt.title('Cumulative Returns')
            plt.xlabel('Date')
            plt.ylabel('Returns (%)')
            plt.legend()
            plt.grid(True)
            
            return plt
        except ImportError:
            logger.error("无法绘制图表，请安装matplotlib")
            return None