"""
所有信息以字典的形式呈现, 例如
price = {
    "000001.SZ": 10,
    ......
}
"""
'当前价格'
#计算买卖的头寸，根据BaseStrategy类的一个买卖规则，得到每期买入和卖出的股票字典
import pandas as pd
import numpy as np
import sys
import os
import traceback
from typing import Dict, Tuple, List

# 确保日志文件存在
debug_file = debug_file = os.path.abspath(r'C:\Users\sirui\Desktop\研报\my_back\strategy_debug.txt')
with open(debug_file, 'w', encoding='utf-8') as f:
    f.write("=== 策略调试日志 ===\n")

# 定义直接写入文件的函数
def debug_log(msg):
    with open(debug_file, 'a', encoding='utf-8') as f:
          f.write(f"{msg}\n")
          f.flush()
    sys.stderr.write(f"STRATEGY: {msg}\n")
    sys.stderr.flush()

debug_log("策略模块初始化")

def raw_prediction_to_signal(pred: pd.Series, total_cash: float, long_only: bool = False) -> pd.Series:
    """
    构造中性化组合并乘可投资资金, 得到每只股票分配的资金,pred_:权重
    """
    if len(pred) > 1:
        if not long_only:
            pred -= pred.groupby(level=0).mean()
        pred_ = pred.copy()
        abs_sum = abs(pred_).groupby(level=0).sum()
        pred_ /= abs_sum
        return pred_ * total_cash
    else:
        return pred * total_cash


def get_trade_volume(signal: pd.Series, price: pd.Series, volume: pd.Series, threshold: float = 0.05,
                     unit: str = "lot") -> pd.Series:
    """
    :param signal: 中性化后的组合份额 * 总资金：该股票的份额
    :param price: 股票价格
    :param volume: 股票成交量
    :param threshold:
    :param unit:
    :return:
    """
    trade_volume = abs(signal) / price  # 得到单位是股
    multiplier = 100 if unit == "lot" else 1
    trade_volume /= multiplier
    max_volume = volume * threshold
    trade_volume_ = trade_volume.copy()
    trade_volume_[trade_volume_ > max_volume] = max_volume
    return (trade_volume_ + 0.5).astype(int) * multiplier  # 四舍五入取整, 最后以股作为单位


def get_price(data: pd.DataFrame) -> dict:
    """获取价格数据,注意用收盘价/开盘价计算"""
    if "open" not in data.columns:
        available_cols = data.columns.tolist()
        debug_log(f"找不到必要的close列。可用列：{available_cols}")
        raise ValueError(f"找不到必要的close列。可用列：{available_cols}")
    
    # 确保数据包含多重索引
    if isinstance(data.index, pd.MultiIndex):
        current_price = data.droplevel(0)["open"].to_dict()
    else:
        current_price = data["open"].to_dict()
    
    return current_price


def get_vol(data: pd.DataFrame, volume: str = "volume") -> dict:
    """获取交易量数据"""
    # 尝试多个可能的交易量列名
    volume_column_candidates = ["volume", "Volume", "vol", "trading_volume"]
    
    # 如果指定的交易量列存在
    if volume in data.columns:
        volume_column = volume
    else:
        # 自动寻找适合的交易量列
        for col in volume_column_candidates:
            if col in data.columns:
                volume_column = col
                debug_log(f"使用'{col}'列作为交易量数据")
                break
        else:
            # 如果找不到适合的交易量列，使用默认值(很大的值，相当于不限制)
            debug_log(f"找不到交易量列。将使用默认值。尝试的列名: {volume_column_candidates}")
            # 创建一个包含很大值的Series
            if isinstance(data.index, pd.MultiIndex):
                codes = data.index.get_level_values(1).unique()
                return {code: 1e9 for code in codes}  # 使用一个非常大的默认值
            else:
                return {idx: 1e9 for idx in data.index}
    
    # 确保数据包含多重索引
    if isinstance(data.index, pd.MultiIndex):
        current_volume = data.droplevel(0)[volume_column].to_dict()
    else:
        current_volume = data[volume_column].to_dict()
        
    return current_volume

def check_signal(order: dict) -> dict:
    """
    只返回value>0的键值对
    """
    return {k: v for k, v in order.items() if v > 0}

#抽象总类
class BaseStrategy:
    def __init__(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if "k" not in kwargs.keys():
            kwargs["k"] = 0.2
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = None
        if "long_only" not in kwargs.keys():
            kwargs["long_only"] = False

        self.k = kwargs["k"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]
        self.long_only = kwargs["long_only"]

    def to_signal(self, **kwargs):
        """
        对输入的每个tick的价格和预测值，输出买和卖的信息
        注意这里是先买后卖, 但执行的时候是先卖后买

        :return: dict like {'buy': {'code': volume}, 'sell': {'code': volume}}
        """
        pass

class RSRSStrategy(BaseStrategy):
    """
    Modified RSRS Strategy that only returns which stocks to buy/sell without calculating quantities.
    Quantities will be calculated later using check_order with equal weighting.
    
    Signal rules:
    1. If signal crosses above upperbound, generate buy signal
    2. If signal crosses below lowerbound, generate sell signal
    """

    def __init__(self, kwargs=None):
        # Call parent class initializer
        super().__init__(kwargs)
        if kwargs is None:
            kwargs = {}
        
        # Set default parameters
        self.commission = kwargs.get("commission", 0.01)
        self.hold_num = kwargs.get("hold_num", 1)
        self.verbose = kwargs.get("verbose", True)  # Default to verbose for debugging
        self.max_volume = kwargs.get("max_volume", 0.05)
        self.k = kwargs.get("k", 0.2)
        self.unit = kwargs.get("unit", "lot")
        self.risk_degree = kwargs.get("risk_degree", 0.95)
        self.equal_weight = kwargs.get("equal_weight", True)
        self.long_only = True

        # Store previous signal states
        self.prev_signals = {}
        self.prev_upper_bounds = {}
        self.prev_lower_bounds = {}
        
        # Add variables to track each call's state
        self.first_call = True
        
        # Track signal crossing events
        self.cross_up_count = 0
        self.cross_down_count = 0
        
        debug_log(f"初始化RSRSStrategy, kwargs={kwargs}")

    def log(self, msg: str, current_dt=None, verbose: bool = None):
        """Log message to debug file"""
        debug_log(msg)

    def check_cross_up(self, signal_value, upper_bound, symbol) -> bool:
        """Check if signal crossed above the upper bound"""
        # Initialize if first time seeing this symbol
        if symbol not in self.prev_signals:
            self.prev_signals[symbol] = signal_value
            self.prev_upper_bounds[symbol] = upper_bound
            return False
        crossed_up = (self.prev_signals[symbol] <= self.prev_upper_bounds[symbol]) and (signal_value > upper_bound)
        self.prev_upper_bounds[symbol] = upper_bound
                
        if crossed_up:
            debug_log(f"⚠️ {symbol} 检测到向上穿越: 前值={self.prev_signals[symbol]:.4f}, 当前值={signal_value:.4f}, 上边界={upper_bound:.4f}")
            self.cross_up_count += 1
            debug_log(f"向上穿越总计: {self.cross_up_count}")
        
        return crossed_up

    def check_cross_down(self, signal_value, lower_bound, symbol) -> bool:
        """Check if signal crossed below the lower bound"""
        # Initialize if first time seeing this symbol
        if symbol not in self.prev_signals:
            self.prev_signals[symbol] = signal_value
            self.prev_lower_bounds[symbol] = lower_bound
            return False
        # Check crossing condition with detailed logging
        prev_above_lower = self.prev_signals[symbol] >= self.prev_lower_bounds[symbol]
        curr_below_lower = signal_value < lower_bound
        crossed_down = prev_above_lower and curr_below_lower
        
  
        # Update stored values for next time AFTER checking the cross
        self.prev_lower_bounds[symbol] = lower_bound
        
        if crossed_down:
            debug_log(f"❌ {symbol} 检测到向下穿越: 前值={self.prev_signals[symbol]:.4f}, 当前值={signal_value:.4f}, 下边界={lower_bound:.4f}")
            self.cross_down_count += 1
            debug_log(f"向下穿越总计: {self.cross_down_count}")
    
        return crossed_down
    def to_signal(self, data: pd.DataFrame, position: dict, cash_available: float = None):
        """
        Generate trading signals - only identify which stocks to buy/sell without calculating quantities.
        
        Returns:
            order: dict with 'buy' and 'sell' keys, values are dicts with stock codes as keys and placeholder values
            price_dict: dict with stock codes as keys and current prices as values
        """
        try:
            debug_log(f"--------- 新的to_signal调用 ---------")
            debug_log(f"数据形状: {data.shape}")
            debug_log(f"当前持仓: {position}")
            
            # Get current date for logging
            if isinstance(data.index, pd.MultiIndex) and len(data.index) > 0:
                current_date = data.index.get_level_values(0)[0]
                debug_log(f"当前时间: {current_date}")
            
            # Check data columns
            debug_log(f"数据列: {data.columns.tolist()}")
            
            # First call just initializes signals without producing trading orders
            if self.first_call:
                debug_log("首次调用，仅初始化信号值和边界")
                self.first_call = False
                
                price_dict = get_price(data)
                
                # Initialize signals for each stock
                for idx, row in data.iterrows():
                    symbol = idx[1] if isinstance(idx, tuple) else idx
                    signal_value = row.get('signal', row.get('predict', 0))
                    upper_bound = row.get('upperbound', 0)
                    lower_bound = row.get('lowerbound', 0)
                    
                    self.prev_signals[symbol] = signal_value
                    self.prev_upper_bounds[symbol] = upper_bound
                    self.prev_lower_bounds[symbol] = lower_bound
                    
                    debug_log(f"初始化股票 {symbol} 的信号: signal={signal_value:.4f}, upper={upper_bound:.4f}, lower={lower_bound:.4f}")
                
                return {"buy": {}, "sell": {}}, price_dict
            
            # Analyze signal statistics for debugging
            signal_stats = {
                "total": 0,
                "above_upper": 0,
                "below_lower": 0,
                "max_signal": -float('inf'),
                "min_signal": float('inf'),
                "upperbound": None,
                "lowerbound": None
            }
            
            for idx, row in data.iterrows():
                signal_value = row.get('signal', row.get('predict', 0))
                upper_bound = row.get('upperbound', 0)
                lower_bound = row.get('lowerbound', 0)
                
                signal_stats["total"] += 1
                signal_stats["max_signal"] = max(signal_stats["max_signal"], signal_value)
                signal_stats["min_signal"] = min(signal_stats["min_signal"], signal_value)
                signal_stats["upperbound"] = upper_bound
                signal_stats["lowerbound"] = lower_bound
                
                if signal_value > upper_bound:
                    signal_stats["above_upper"] += 1
                if signal_value < lower_bound:
                    signal_stats["below_lower"] += 1
            
            debug_log(f"信号统计: {signal_stats}")
            
            # Initialize buy/sell dictionaries
            buy_dict = {}
            sell_dict = {}
            
            # Get price dictionary
            try:
                price_dict = get_price(data)
            except KeyError as e:
                debug_log(f"获取价格数据失败: {e}")
                return {"buy": {}, "sell": {}}, {}
            
            # Process each stock
            for idx, row in data.iterrows():
                try:
                    # Get stock code
                    symbol = idx[1] if isinstance(idx, tuple) else idx
                    
                    # Extract signal and boundary values
                    signal_value = row.get('signal', None)
                    if signal_value is None:
                        signal_value = row.get('predict', 0)
                        
                    upper_bound = row.get('upperbound', 0)
                    lower_bound = row.get('lowerbound', 0)
                    
                    # Get price
                    price = price_dict.get(symbol, 0)
                    if price <= 0:
                        debug_log(f"股票 {symbol} 价格不可用，跳过")
                        continue
                    
                    # Get current position
                    current_position = position.get(symbol, 0)
                    
                    # Log signal values
                    debug_log(f"股票 {symbol}: 当前信号={signal_value:.4f},上个信号 = {self.prev_signals[symbol]:.4f}, 上边界={upper_bound:.4f}, 下边界={lower_bound:.4f}, 当前持仓={current_position}")
                    
                    # Check buy signal (crossing above upper bound)
                    if self.check_cross_up(signal_value, upper_bound, symbol):
                        if current_position == 0:  # Only buy if no current position
                            # Use placeholder value (1) - actual quantity will be calculated later
                            buy_dict[symbol] = 1
                            debug_log(f"✅生成买入信号: {symbol}, 价格: {price}")
                    
                    # Check sell signal (crossing below lower bound)
                    if self.check_cross_down(signal_value, lower_bound, symbol):
                        if current_position > 0:  # Only sell if there's a position
                            # Use current position as placeholder - will be adjusted in check_order
                            sell_dict[symbol] = current_position
                            debug_log(f"❌生成卖出信号: {symbol}, 当前持仓: {current_position}, 价格: {price}")
                
          
                    self.prev_signals[symbol] = signal_value
                except Exception as e:
                    debug_log(f"处理股票时出错: {e}")
                    debug_log(traceback.format_exc())
                    continue
                    
            debug_log(f"最终订单: 买入 {buy_dict}, 卖出 {sell_dict}")
            
            # Return order and price dictionary
            order = {"buy": buy_dict, "sell": sell_dict}
            return order, price_dict
            
        except Exception as e:
            debug_log(f"生成交易信号时出现错误: {e}")
            debug_log(traceback.format_exc())
            return {"buy": {}, "sell": {}}, {}

class WeeklyRebalancingStrategy:
    """
    周度调仓策略：
    - 每周最后一日调仓
    - 如果该日 position*signal=0，仓位设定为半仓
    - 用下周第一个收盘价买入市值最小的五支股票，持仓一周计算
    - 每天计算收益率
    - 到下周的调仓日，如果不在新持仓的持仓股卖出，买入原来不在仓中的新股票
    """
    def __init__(self, initial_cash=1e8, commission=0.00015, stamp_duty=0.0001, min_cost=5.0):
        """初始化策略参数"""
        self.initial_cash = initial_cash
        self.commission = commission  # 交易佣金率
        self.stamp_duty = stamp_duty  # 印花税率
        self.min_cost = min_cost      # 最低交易成本
        
        self.current_positions = {}   # 当前持仓
        self.total_value = initial_cash  # 总资产价值
        self.cash = initial_cash      # 当前现金
        
        self.history = {
            'date': [],
            'total_value': [],
            'cash': [],
            'positions': [],
            'returns': []
        }
        
        self.last_rebalance_date = None  # 上次调仓日期
        self.next_rebalance_stocks = []  # 下次调仓预定的股票
        self.half_position = False       # 是否设为半仓
        self.is_first_day_of_week = False  # 是否是周的第一天

    def is_last_day_of_week(self, date):
        """判断是否是周最后一个交易日"""
        # 获取该日期的下一个交易日
        next_trading_date = self.trading_dates[self.trading_dates > date][0] if date < self.trading_dates[-1] else None
        
        if next_trading_date is None:
            return False  # 如果是最后一个交易日，视为周最后一天
            
        # 如果当前日期和下一个交易日不在同一周，则当前日期是该周最后一个交易日
        return pd.Timestamp(date).week != pd.Timestamp(next_trading_date).week
    
    def is_first_day_of_week(self, date):
        """判断是否是周第一个交易日"""
        # 获取该日期的前一个交易日
        prev_trading_date = self.trading_dates[self.trading_dates < date][-1] if date > self.trading_dates[0] else None
        
        if prev_trading_date is None:
            return True  # 如果是第一个交易日，视为周第一天
            
        # 如果当前日期和前一个交易日不在同一周，则当前日期是该周第一个交易日
        return pd.Timestamp(date).week != pd.Timestamp(prev_trading_date).week
    
    def select_smallest_cap_stocks(self, date, price_and_cap_df, stock_pool, num_stocks=5):
        """选择市值最小的n支股票"""
        # 获取当前日期的市值数据
        try:
            # 从市值数据中获取当前日期的数据
            current_date_df = price_and_cap_df.xs(date, level='date')
            
            # 筛选当前股票池中的股票
            available_stocks = current_date_df[current_date_df.index.isin(stock_pool)]
            
            # 按市值排序，选择最小的n支
            smallest_cap_stocks = available_stocks.sort_values('cap').head(num_stocks)
            
            return smallest_cap_stocks.index.tolist()
        except Exception as e:
            print(f"Error selecting smallest cap stocks for date {date}: {e}")
            return []
    
    def calculate_rebalance(self, date, price_and_cap_df, ks_df, stock_pool):
        """计算调仓信号"""
        # 检查当前日期在ks_df中是否有数据
        if date not in ks_df.index:
            print(f"Warning: {date} not found in ks_df")
            return False, []
        
        # 获取当前日期的position*signal值
        position_signal = ks_df.loc[date, 'position'] * ks_df.loc[date, 'signal']
        
        # 如果position*signal为0，设置半仓标志
        half_position = (position_signal == 0)
        
        # 选择市值最小的5支股票
        smallest_stocks = self.select_smallest_cap_stocks(date, price_and_cap_df, stock_pool)
        
        return half_position, smallest_stocks
    
    def execute_trades(self, date, price_and_cap_df, new_stocks):
        """执行交易，卖出不在新持仓的股票，买入新的股票"""
        # 获取当前日期的价格数据
        try:
            current_prices = price_and_cap_df.xs(date, level='date')['close'].to_dict()
        except KeyError:
            print(f"Warning: No price data for {date}")
            return
        
        # 计算卖出操作
        for stock in list(self.current_positions.keys()):
            if stock not in new_stocks:
                # 卖出股票
                shares = self.current_positions[stock]
                price = current_prices.get(stock, 0)
                
                if price > 0:
                    # 计算卖出收入和成本
                    sell_value = shares * price
                    sell_cost = max(self.min_cost, sell_value * (self.commission + self.stamp_duty))
                    
                    # 更新现金
                    self.cash += (sell_value - sell_cost)
                    
                    # 移除持仓
                    del self.current_positions[stock]
                    print(f"Sold {shares} shares of {stock} at {price}, value: {sell_value}, cost: {sell_cost}")
        
        # 计算买入操作
        if self.half_position:
            # 半仓模式，只使用一半资金
            available_cash = self.cash * 0.5
        else:
            # 全仓模式，使用所有资金
            available_cash = self.cash
        
        # 计算每只股票分配的资金(等权重)
        stocks_to_buy = [s for s in new_stocks if s not in self.current_positions]
        if not stocks_to_buy:
            return
            
        cash_per_stock = available_cash / len(stocks_to_buy)
        
        # 执行买入
        for stock in stocks_to_buy:
            if stock in current_prices and current_prices[stock] > 0:
                price = current_prices[stock]
                
                # 计算买入股数(按整手100股计算)
                max_shares = int((cash_per_stock / price) / 100) * 100
                
                if max_shares > 0:
                    # 计算买入成本
                    buy_value = max_shares * price
                    buy_cost = max(self.min_cost, buy_value * self.commission)
                    
                    # 更新现金和持仓
                    self.cash -= (buy_value + buy_cost)
                    self.current_positions[stock] = max_shares
                    
                    print(f"Bought {max_shares} shares of {stock} at {price}, value: {buy_value}, cost: {buy_cost}")
    