"""
signal_generator.py
交易信号生成器，包含各种交易信号生成策略
"""

import numpy as np
import pandas as pd


class SignalGenerator:
    """
    信号生成器基类，包含各种交易信号生成策略
    """
    
    @staticmethod
    def three_ma_signal(close_data, win5=5, win20=20, win60=60):
        """
        三均线择时策略
        
        参数:
        close_data: 收盘价数据，形状为 [股票数, 时间长度]
        win5: 短期均线窗口
        win20: 中期均线窗口
        win60: 长期均线窗口
        
        返回:
        买入信号和卖出信号
        """
        # 确保收盘价数据形状正确
        if len(close_data.shape) == 1:
            close_data = close_data.reshape(1, -1)
        
        # 确保有足够的数据计算均线
        if close_data.shape[1] < win60:
            raise ValueError(f"收盘价数据长度不足，需要至少{win60}个数据点")
        
        # 计算均线
        ma5 = np.mean(close_data[:, -win5:], axis=1)
        ma20 = np.mean(close_data[:, -win20:], axis=1)
        ma60 = np.mean(close_data[:, -win60:], axis=1)
        
        # 生成买入信号：5日均线和20日均线都大于60日均线
        buy_signals = (ma5 > ma60) & (ma20 > ma60)
        
        # 生成卖出信号：5日均线和20日均线都小于60日均线
        sell_signals = (ma5 < ma60) & (ma20 < ma60)
        
        return buy_signals, sell_signals
    
    @staticmethod
    def ma_cross_signal(close_data, short_win=5, long_win=20):
        """
        均线交叉策略
        
        参数:
        close_data: 收盘价数据，形状为 [股票数, 时间长度]
        short_win: 短期均线窗口
        long_win: 长期均线窗口
        
        返回:
        买入信号和卖出信号
        """
        # 确保收盘价数据形状正确
        if len(close_data.shape) == 1:
            close_data = close_data.reshape(1, -1)
        
        # 确保有足够的数据计算均线
        if close_data.shape[1] < long_win + 1:
            raise ValueError(f"收盘价数据长度不足，需要至少{long_win + 1}个数据点")
        
        # 计算昨日均线
        prev_short_ma = np.mean(close_data[:, -(short_win+1):-1], axis=1)
        prev_long_ma = np.mean(close_data[:, -(long_win+1):-1], axis=1)
        
        # 计算今日均线
        curr_short_ma = np.mean(close_data[:, -short_win:], axis=1)
        curr_long_ma = np.mean(close_data[:, -long_win:], axis=1)
        
        # 生成买入信号：短期均线上穿长期均线
        buy_signals = (prev_short_ma <= prev_long_ma) & (curr_short_ma > curr_long_ma)
        
        # 生成卖出信号：短期均线下穿长期均线
        sell_signals = (prev_short_ma >= prev_long_ma) & (curr_short_ma < curr_long_ma)
        
        return buy_signals, sell_signals
    
    @staticmethod
    def bollinger_bands_signal(close_data, window=20, num_std=2):
        """
        布林带策略
        
        参数:
        close_data: 收盘价数据，形状为 [股票数, 时间长度]
        window: 移动窗口大小
        num_std: 标准差倍数
        
        返回:
        买入信号和卖出信号
        """
        # 确保收盘价数据形状正确
        if len(close_data.shape) == 1:
            close_data = close_data.reshape(1, -1)
        
        # 确保有足够的数据计算布林带
        if close_data.shape[1] < window:
            raise ValueError(f"收盘价数据长度不足，需要至少{window}个数据点")
        
        # 计算移动平均线
        ma = np.mean(close_data[:, -window:], axis=1)
        
        # 计算移动标准差
        std = np.std(close_data[:, -window:], axis=1)
        
        # 计算上轨和下轨
        upper_band = ma + num_std * std
        lower_band = ma - num_std * std
        
        # 获取最新收盘价
        latest_close = close_data[:, -1]
        
        # 生成买入信号：价格触及下轨
        buy_signals = latest_close <= lower_band
        
        # 生成卖出信号：价格触及上轨
        sell_signals = latest_close >= upper_band
        
        return buy_signals, sell_signals
    
    @staticmethod
    def combine_signals(ml_signals, tech_signals, method='AND'):
        """
        组合机器学习信号和技术指标信号
        
        参数:
        ml_signals: 机器学习信号 (buy_signals, sell_signals)
        tech_signals: 技术指标信号 (buy_signals, sell_signals)
        method: 组合方法，'AND'表示两个信号同时满足才触发，'OR'表示任一信号满足即触发
        
        返回:
        组合后的买入信号和卖出信号
        """
        ml_buy, ml_sell = ml_signals
        tech_buy, tech_sell = tech_signals
        
        if method == 'AND':
            # 组合买入信号：机器学习和技术指标都给出买入信号
            combined_buy = ml_buy & tech_buy
            
            # 组合卖出信号：机器学习和技术指标都给出卖出信号
            combined_sell = ml_sell & tech_sell
        
        elif method == 'OR':
            # 组合买入信号：机器学习或技术指标给出买入信号
            combined_buy = ml_buy | tech_buy
            
            # 组合卖出信号：机器学习或技术指标给出卖出信号
            combined_sell = ml_sell | tech_sell
        
        else:
            raise ValueError("方法参数必须为'AND'或'OR'")
        
        return combined_buy, combined_sell