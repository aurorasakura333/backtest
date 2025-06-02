"""
data_processor.py
数据处理模块，负责数据的加载、预处理、特征工程等
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import datetime


class DataLoader:
    """
    数据加载器，负责从不同来源加载数据
    """
    
    @staticmethod
    def load_from_csv(folder_path, columns_to_drop=None, start_date='20100101'):
        """
        从CSV文件加载数据
        
        参数:
        folder_path: CSV文件所在文件夹路径
        columns_to_drop: 需要删除的列
        start_date: 开始日期，格式为'YYYYMMDD'
        
        返回:
        加载后的DataFrame
        """
        if columns_to_drop is None:
            columns_to_drop = ['open', 'open_hfq', 'open_qfq', 'high', 'high_hfq', 'high_qfq', 
                              'low', 'low_hfq', 'low_qfq', 'close', 'close_hfq', 'pre_close', 
                              'change', 'vol', 'amount', 'turnover_rate', 'turnover_rate_f', 'volume_ratio']
        
        # 存储结果的字典
        df_dict = {}
        
        # 遍历文件夹中的所有 CSV 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):  # 只处理 CSV 文件
                file_path = os.path.join(folder_path, file_name)
                
                # 读取 CSV 文件
                df = pd.read_csv(file_path, dtype={'trade_date': str})  # 确保 trade_date 是字符串
                
                # 删除指定列（如果存在）
                df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                
                # 过滤开始日期之后的数据
                df = df[df['trade_date'] >= start_date]
                
                # 以文件名（去掉.csv）为键，DataFrame 为值存入字典
                df_dict[file_name.replace('.csv', '')] = df
        
        # 合并所有数据
        df = pd.concat(df_dict.values(), axis=0, ignore_index=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.rename(columns={'trade_date': 'date', 'ts_code': 'stock_id'})
        df = df.set_index(['date', 'stock_id']).sort_index()
        
        return df


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
    def normalize_factors( df, factors):
        """
        对因子进行标准化处理
        
        参数:
        df: 待处理的DataFrame
        factors: 因子列表
        
        返回:
        处理后的DataFrame
        """
        result = df.copy()
        all_dates = df.index.get_level_values('date').unique()
        # 按日期分组进行标准化
        for factor in factors:
            # 一次进行去极值而不是每个日期单独处理
            result = DataProcessor.filter_MAD(result, factor, 5)
            
            # 创建日期分组的索引
            date_groups = {}
            for date in all_dates:
                date_mask = result.index.get_level_values('date') == date
                date_groups[date] = date_mask
            
            # 批量计算均值和标准差
            for date, mask in date_groups.items():
                values = result.loc[mask, factor]
                mean = values.mean()
                std = values.std()
                
                # 避免除以0
                if std > 0:
                    result.loc[mask, factor] = (values - mean) / std
                else:
                    result.loc[mask, factor] = 0
        
        return result
    @staticmethod
    def prepare_monthly_training_data(df, factor_codes, prev_month_begin, prev_month_end, current_month_begin):
        """
        准备月度训练数据
        
        参数:
        df: 输入的DataFrame，包含日期、股票ID和因子数据
        factor_codes: 使用的因子列表
        prev_month_begin: 上个月第一个交易日
        prev_month_end: 上个月最后一个交易日
        current_month_begin: 当前月第一个交易日
        
        返回:
        X: 训练特征
        Y: 训练标签
        X_test: 测试特征
        test_stocks: 测试股票ID
        """
        # 获取上个月的数据
        prev_month_days = df.index.get_level_values('date').unique()
        prev_month_days = [day for day in prev_month_days 
                           if day >= prev_month_begin and day <= prev_month_end]
        
        if len(prev_month_days) < 2:
            return None, None, None, None
        
        # 获取上个月所有股票
        all_stocks = df.loc[prev_month_begin:prev_month_end].index.get_level_values('stock_id').unique()
        
        # 准备训练数据
        train_data = []
        
        for stock in all_stocks:
            try:
                # 获取该股票的收盘价数据
                stock_data = df.xs(stock, level='stock_id')
                
                if 'close_qfq' not in stock_data.columns:
                    continue
                
                # 获取该股票在上个月初和月末的价格
                if prev_month_begin in stock_data.index and prev_month_end in stock_data.index:
                    start_price = stock_data.loc[prev_month_begin, 'close_qfq']
                    end_price = stock_data.loc[prev_month_end, 'close_qfq']
                    
                    if start_price > 0:
                        # 计算收益率
                        benefit = (end_price - start_price) / start_price
                        
                        # 获取上个月初的因子数据
                        factors = stock_data.loc[prev_month_begin, factor_codes].to_dict()
                        
                        # 构建样本
                        row = {'stock_id': stock, 'benefit': benefit}
                        row.update(factors)
                        train_data.append(row)
            except Exception as e:
                print(f"Error processing stock {stock}: {str(e)}")
                continue
        
        # 构建训练集DataFrame
        train_df = pd.DataFrame(train_data)
        
        if train_df.empty:
            return None, None, None, None
        
        # 获取上个月末的因子数据作为测试数据
        test_data = []
        
        for stock in all_stocks:
            try:
                # 获取该股票的数据
                stock_data = df.xs(stock, level='stock_id')
                
                if prev_month_end in stock_data.index:
                    # 获取上个月末的因子数据
                    factors = stock_data.loc[prev_month_end, factor_codes].to_dict()
                    
                    # 构建样本
                    row = {'stock_id': stock}
                    row.update(factors)
                    test_data.append(row)
            except Exception as e:
                print(f"Error processing test data for stock {stock}: {str(e)}")
                continue
        
        # 构建测试集DataFrame
        test_df = pd.DataFrame(test_data)
        
        if test_df.empty:
            return None, None, None, None
        
        # 数据清洗和预处理
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        # 构建numpy数组
        X = np.array(train_df[factor_codes])
        Y = np.array(train_df['benefit'])
        X_test = np.array(test_df[factor_codes])
        test_stocks = test_df['stock_id'].values
        
        return X, Y, X_test, test_stocks
    
    
    
    @staticmethod
    def prepare_daily_training_data(df, factor_codes, prev_month_begin, prev_month_end, current_month_begin):
        # Get trading days in the previous month
        prev_month_days = df.index.get_level_values('date').unique()
        prev_month_days = sorted([day for day in prev_month_days 
                        if day >= prev_month_begin and day <= prev_month_end])
        
        if len(prev_month_days) < 2:
            return None, None, None, None
        
        # Get data subset for the previous month
        prev_month_data = df.loc[prev_month_begin:prev_month_end].copy()
        
        # Only normalize the factor columns (not price or other data)
        prev_month_data = DataProcessor.normalize_factors(prev_month_data, factor_codes)
        
        # Continue with your data preparation as before
        all_stocks = prev_month_data.index.get_level_values('stock_id').unique()
        daily_train_data = []

        for i in range(len(prev_month_days)-1):
            current_day = prev_month_days[i]
            next_day = prev_month_days[i+1]
            
            for stock in all_stocks:
                try:
                    # Get data for this stock
                    stock_data = prev_month_data.xs(stock, level='stock_id')
                    
                    if 'close_qfq' not in stock_data.columns:
                        continue
                    
                    # Get prices and factors
                    if current_day in stock_data.index and next_day in stock_data.index:
                        current_price = stock_data.loc[current_day, 'close_qfq']
                        next_price = stock_data.loc[next_day, 'close_qfq']
                        
                        # Get the already normalized factors
                        factors = stock_data.loc[current_day, factor_codes].to_dict()
                        
                        # Calculate return
                        if current_price > 0:
                            benefit = (next_price - current_price) / current_price
                            
                            row = {
                                'stock_id': stock,
                                'date': current_day,
                                'benefit': benefit
                            }
                            row.update(factors)
                            daily_train_data.append(row)
                except Exception as e:
                    continue
        
        # Build DataFrame from collected data
        train_df = pd.DataFrame(daily_train_data)
        
        if train_df.empty:
            return None, None, None, None
        
        # For test data, get the normalized factors from the last day of the month
        test_data = []
        for stock in all_stocks:
            try:
                if (prev_month_end, stock) in prev_month_data.index:
                    # Get the already normalized factors
                    factors = prev_month_data.loc[(prev_month_end, stock), factor_codes].to_dict()
                    
                    row = {'stock_id': stock}
                    row.update(factors)
                    test_data.append(row)
            except Exception:
                continue
        
        # Build test DataFrame
        test_df = pd.DataFrame(test_data)
        
        if test_df.empty:
            return None, None, None, None
        
        # Clean data
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
        # Since factors are already normalized, just convert to numpy arrays
        X = np.array(train_df[factor_codes])
        Y = np.array(train_df['benefit'])
        X_test = np.array(test_df[factor_codes])
        test_stocks = test_df['stock_id'].tolist()  # Convert to list to avoid the .index() issue
        
        return X, Y, X_test, test_stocks


class TradingCalendar:
    """
    交易日历类，处理交易日期相关的功能
    """
    
    @staticmethod
    def get_trading_days(exchange, start_date, end_date):
        """
        获取交易日
        
        参数:
        exchange: 交易所代码，如'SSE'代表上海证券交易所
        start_date: 开始日期
        end_date: 结束日期
        
        返回:
        交易日列表
        """
        # 这里应该对接实际的交易日历数据
        # 为了演示，我们简单地生成一个日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        all_days = pd.date_range(start=start, end=end)
        
        # 过滤出工作日（周一至周五）
        trading_days = all_days[all_days.dayofweek < 5]
        
        return trading_days
    
    @staticmethod
    def get_month_begin_days(trading_days):
        """
        获取每月第一个交易日
        
        参数:
        trading_days: 交易日列表
        
        返回:
        每月第一个交易日列表
        """
        # 将交易日转换为月份
        months = np.vectorize(lambda x: x.month)(trading_days)
        years = np.vectorize(lambda x: x.year)(trading_days)
        
        # 创建年月组合
        year_months = [f"{y}-{m:02d}" for y, m in zip(years, months)]
        year_months_series = pd.Series(year_months, index=trading_days)
        
        # 找出每个年月组合的第一个交易日
        month_begin = trading_days[~year_months_series.duplicated()]
        
        return month_begin
    
    @staticmethod
    def get_month_end_days(trading_days):
        """
        获取每月最后一个交易日
        
        参数:
        trading_days: 交易日列表
        
        返回:
        每月最后一个交易日列表
        """
        # 将交易日转换为月份
        months = np.vectorize(lambda x: x.month)(trading_days)
        years = np.vectorize(lambda x: x.year)(trading_days)
        
        # 创建年月组合
        year_months = [f"{y}-{m:02d}" for y, m in zip(years, months)]
        year_months_series = pd.Series(year_months, index=trading_days)
        
        # 找出每个年月组合的最后一个交易日
        # 先将序列反转，找出不重复的第一个值，再反转回来
        month_end = trading_days[~year_months_series[::-1].duplicated()][::-1]
        
        return month_end