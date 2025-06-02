import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import math
from ML_pipeline.transaction_manager import TransactionManager


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
             slippage_rate=0.0001,
             upper_pos=80,
             down_pos=20,
             use_optimizer=True):
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
        target_position_count: 目标持仓数量，默认为10
        """
        self.data = data
        self.strategy = strategy
        self.signal_generator = signal_generator
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.use_technical_signal = use_technical_signal
        self.signal_method = signal_method
        self.technical_indicator = technical_indicator
        self. upper_pos = upper_pos
        self.down_pos = down_pos
        self.use_optimizer = use_optimizer
      
        # 初始化交易管理器
        self.transaction_manager = TransactionManager(
            initial_cash=initial_cash,
            realistic_trading=realistic_trading,
            transaction_fee_rate=transaction_fee_rate,
            slippage_rate=slippage_rate
        )
        
        # 获取交易日
        self.trading_days = self.data.index.get_level_values('date').unique()
        self.trading_days = sorted([day for day in self.trading_days if self.start_date <= day <= self.end_date])
        
        # 获取月初和月末日期
        self.month_begin_days = self._get_month_begin_days()
        self.month_end_days = self._get_month_end_days()
        
        # 记录组合价值历史
        self.positions = {}        # 持仓情况 {stock_id: volume}
        self.portfolio_values = []  # 组合价值历史
        self.dates = []           # 对应的日期
        
        # 属性引用简化
        self.initial_cash = initial_cash
        self.realistic_trading = realistic_trading
        self.trades = self.transaction_manager.trades

        self.preprocess_data()
    def _get_month_begin_days(self):
        """
        获取每月第一个交易日
        
        返回:
        每月第一个交易日列表
        """
        trading_days_series = pd.Series(self.trading_days)
    
        # 将交易日转换为月份
        months = np.vectorize(lambda x: x.month)(self.trading_days)
        years = np.vectorize(lambda x: x.year)(self.trading_days)
        
        # 创建年月组合
        year_months = pd.Series([f"{y}-{m:02d}" for y, m in zip(years, months)], index=range(len(self.trading_days)))
        
        # 找出每个年月组合的第一个交易日
        month_begin_indices = ~year_months.duplicated()
        month_begin = [self.trading_days[i] for i in range(len(self.trading_days)) if month_begin_indices[i]]
        
        return month_begin
    
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
    def preprocess_data(self):
        # 创建价格查询字典，避免每次都使用.loc索引
        self.price_cache = {}
        for date in self.trading_days:
            self.price_cache[date] = {}
            try:
                date_slice = self.data.xs(date, level='date')
                for stock_id, row in date_slice.iterrows():
                    if 'close_qfq' in row:
                        self.price_cache[date][stock_id] = row['close_qfq']
            except:
                continue  # 如果某个日期没有数据，跳过
        
        # 预计算月初和月末，避免每次都要计算
        self.month_begin_days = self._get_month_begin_days()
        self.month_end_days = self._get_month_end_days()
    def get_price(self, stock_id, date, price_field='close_qfq'):
        """
        获取指定股票在指定日期的价格，优先使用缓存
        
        参数:
        stock_id: 股票ID
        date: 日期
        price_field: 价格字段，默认为'close_qfq'
        
        返回:
        价格，如果数据不存在则返回None
        """
        # 首先尝试从缓存中获取
        if hasattr(self, 'price_cache') and date in self.price_cache and stock_id in self.price_cache[date]:
            return self.price_cache[date][stock_id]
        try:
            price = self.data.loc[(date, stock_id), price_field]
            return price
        except KeyError:
            return None

    # def get_price(self, stock_id, date, price_field='close_qfq'):
    #     """
    #     获取指定股票在指定日期的价格
        
    #     参数:
    #     stock_id: 股票ID
    #     date: 日期
    #     price_field: 价格字段，默认为'close_qfq'
        
    #     返回:
    #     价格，如果数据不存在则返回None
    #     """
    #     try:
    #         price = self.data.loc[(date, stock_id), price_field]
    #         return price
    #     except KeyError:
    #         return None
   
    def get_technical_signals(self, close_data):
        """
        根据技术指标类型动态获取买卖信号
        
        参数:
        close_data: 收盘价数据
        
        返回:
        买入信号和卖出信号
        """
        # 建立技术指标名称到方法名的映射
        indicator_method_map = {
            'three_ma': 'three_ma_signal',
            'ma_cross': 'ma_cross_signal',
            'bollinger': 'bollinger_bands_signal',
        }
        
        # 获取对应的方法名
        method_name = indicator_method_map.get(self.technical_indicator, 'three_ma_signal')
        
        # 动态获取方法
        signal_method = getattr(self.signal_generator, method_name)
        
        # 调用方法并返回结果
        return signal_method(close_data)
    
    def update_portfolio_value(self, date):
        """
        更新指定日期的组合价值
        
        参数:
        date: 日期
        """
        portfolio_value = self.transaction_manager.calculate_portfolio_value(date, self.get_price)
        self.portfolio_values.append(portfolio_value)
        self.dates.append(date)
    
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
            print(f"交易费率: {self.transaction_manager.transaction_fee_rate*100:.4f}%, 滑点比例: {self.transaction_manager.slippage_rate*100:.4f}%")
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

            # 根据预测选择新的投资组合股票
            new_portfolio_stocks = self.transaction_manager.get_high_return_portfolio(
                predictions, test_stocks,self.upper_pos, self.down_pos
            )
            
            print(f"新选出的持仓股票数量: {len(new_portfolio_stocks)}")
            
            # 如果使用技术指标信号，过滤掉技术面不好的股票
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
                
                # 创建技术指标信号的映射
                tech_signal_map = {test_stocks[i]: tech_buy[i] for i in range(len(test_stocks))}
                
                # 过滤掉技术面不好的股票
                filtered_portfolio = [stock for stock in new_portfolio_stocks if tech_signal_map.get(stock, False)]
                
                print(f"技术指标过滤后的持仓股票数量: {len(filtered_portfolio)}")
                
                # 如果过滤后没有股票，则使用原来的投资组合
                if filtered_portfolio and len(filtered_portfolio) > 10:
                    new_portfolio_stocks = filtered_portfolio
            
            # 获取当前价格字典
            prices_dict = {}
            historical_data = {}
            for stock_id in new_portfolio_stocks:
                    # 获取上个月的历史收盘价数据
                    stock_data = self.data.xs(stock_id, level='stock_id')
                    if 'close_qfq' in stock_data.columns:
                        # 提取上个月的数据范围
                        monthly_data = stock_data.loc[prev_month_begin:prev_month_end, 'close_qfq']
                        if not monthly_data.empty:
                            historical_data[stock_id] = monthly_data.values.tolist()

            # 执行投资组合再平衡
            rebalance_result = self.transaction_manager.rebalance_portfolio(
                current_date, 
                new_portfolio_stocks, 
                prices_dict, 
                self.get_price,
                test_stocks=test_stocks,
                predictions=predictions,
                historical_data=historical_data,
                use_optimizer=self.use_optimizer
            )
            
            print(f"卖出股票数量: {len(rebalance_result['sold_stocks'])}")
            print(f"买入股票数量: {len(rebalance_result['bought_stocks'])}")
            print(f"剩余现金: {rebalance_result['cash_remaining']:.2f}")
            print(f"组合价值: {rebalance_result['total_portfolio_value']:.2f}")
            
            # 更新组合价值
            self.update_portfolio_value(current_date)
        
        # 回测结束，计算绩效指标
        self.performance = self.calculate_performance()
        
        # 打印交易成本统计
        if self.realistic_trading:
            print("\n===== 交易成本统计 =====")
            print(f"总交易费用: {self.transaction_manager.total_transaction_fees:.2f} ({self.transaction_manager.total_transaction_fees / self.initial_cash * 100:.4f}% 初始资金)")
            print(f"总滑点成本: {self.transaction_manager.total_slippage_cost:.2f} ({self.transaction_manager.total_slippage_cost / self.initial_cash * 100:.4f}% 初始资金)")
            print(f"总交易成本: {self.transaction_manager.total_transaction_fees + self.transaction_manager.total_slippage_cost:.2f} ({(self.transaction_manager.total_transaction_fees + self.transaction_manager.total_slippage_cost) / self.initial_cash * 100:.4f}% 初始资金)")
        
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
                'total_transaction_fees': self.transaction_manager.total_transaction_fees,
                'total_slippage_cost': self.transaction_manager.total_slippage_cost,
                'total_cost_ratio': (self.transaction_manager.total_transaction_fees + self.transaction_manager.total_slippage_cost) / self.initial_cash
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
