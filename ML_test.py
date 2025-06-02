"""
examples.py
示例用法脚本，展示如何运行LSTM和RandomForest策略
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import argparse

# 导入自定义模块
from ML_pipeline.main import main
from ML_pipeline.data_processor import DataLoader, DataProcessor
from ML_pipeline.signal_generator import SignalGenerator
from ML_pipeline.backtest_framework import BacktestFramework
# 修改这行
from ML_pipeline.ml_strategies import LSTMStrategy, RandomForestStrategy, AdaBoostStrategy, XGBoostStrategy


def get_df():
    import pandas as pd
    import os

    # 设定文件夹路径
    folder_path = r"C:\Users\sirui\Desktop\机器学习\data02"

    # 需要删除的列,保留code_id,date,close_qfq,所有基本面，指标因子
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
            
            # 过滤 2010 年之后的数据
            df = df[df['trade_date'] >= '20100101']
            
            # 以文件名（去掉.csv）为键，DataFrame 为值存入字典
            df_dict[file_name.replace('.csv', '')] = df
    
    # 修正变量名: df_dic -> df_dict
    df = pd.concat(df_dict.values(), axis=0, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.rename(columns={'trade_date': 'date', 'ts_code': 'stock_id'})
    df = df.set_index(['date', 'stock_id']).sort_index()
    return df
def run_xgboost_example(use_technical=False, technical_indicator='three_ma', realistic_trading=False, 
                       use_optimizer=True, use_pca=False, n_components=20):
    """
    运行XGBoost策略示例
    
    参数:
    use_technical: 是否使用技术指标
    technical_indicator: 使用的技术指标类型
    realistic_trading: 是否考虑交易成本
    use_optimizer: 是否使用投资组合优化器
    use_pca: 是否使用PCA降维
    n_components: PCA降维后的特征数量
    """
    print("===== 运行XGBoost策略示例 =====")
    
    # 保存原始的命令行参数
    original_argv = sys.argv.copy()
    
    # 设置新的命令行参数以运行XGBoost策略
    sys_argv = [
        'main.py',  # 脚本名称
        '--strategy', 'xgboost',  # 使用XGBoost策略
        '--start_date', '2018-01-01',  # 回测开始日期
        '--end_date', '2018-12-31',  # 回测结束日期
        '--initial_cash', '10000000',  # 初始资金
        '--signal_method', 'AND',  # 信号组合方法
        '--data_folder', r"C:\Users\sirui\Desktop\机器学习\data02",  # 数据文件夹路径
        '--upper_pos', '80',  # 上分位数阈值
        '--down_pos', '20'    # 下分位数阈值
    ]
    
    # 根据参数添加技术指标选项
    if use_technical:
        sys_argv.append('--use_technical')
        sys_argv.append('--technical_indicator')
        sys_argv.append(technical_indicator)
    
    # 添加交易成本选项
    if realistic_trading:
        sys_argv.append('--realistic_trading')
        
    # 添加优化器选项
    if not use_optimizer:
        sys_argv.append('--no_optimizer')
    
    # 添加PCA选项
    if use_pca:
        sys_argv.append('--use_pca')
        sys_argv.append('--n_components')
        sys_argv.append(str(n_components))
        
    sys.argv = sys_argv
    try:
        # 调用main函数
        backtest = main()
        return backtest
    finally:
        # 恢复原始的命令行参数
        sys.argv = original_argv


def run_lstm_example(use_technical=False, technical_indicator='three_ma', realistic_trading=False, use_optimizer=True):
    """
    运行LSTM策略示例
    """
    print("===== 运行LSTM策略示例 =====")
    
    original_argv = sys.argv.copy()
    sys_argv = [
        'main.py',  # 脚本名称
        '--strategy', 'lstm',  # 使用AdaBoost策略
        '--start_date', '2018-01-01',  # 回测开始日期
        '--end_date', '2018-12-31',  # 回测结束日期
        '--initial_cash', '10000000',  # 初始资金
        '--signal_method', 'AND',  # 信号组合方法
        '--data_folder', r"C:\Users\sirui\Desktop\机器学习多因子\data02",  # 数据文件夹路径
        '--upper_pos', '80',  # 上分位数阈值
        '--down_pos', '20'    # 下分位数阈值
    ]
    if use_technical:
        sys_argv.append('--use_technical')
        sys_argv.append('--technical_indicator')
        sys_argv.append(technical_indicator)

    if realistic_trading:
        sys_argv.append('--realistic_trading')
        
    if not use_optimizer:
        sys_argv.append('--no_optimizer')
        
    sys.argv = sys_argv
    try:
        # 调用main函数
        backtest = main()
        return backtest
    finally:
        # 恢复原始的命令行参数
        sys.argv = original_argv


def run_rf_line3_example(use_technical=False, technical_indicator='three_ma', realistic_trading=False, use_optimizer=True):
    """
    运行RF_line3策略示例
    """
    print("===== 运行RF_line3策略示例 =====")
    original_argv = sys.argv.copy()
    sys_argv = [
        'main.py',  # 脚本名称
        '--strategy', 'rf',  # 使用AdaBoost策略
        '--start_date', '2018-01-01',  # 回测开始日期
        '--end_date', '2018-12-31',  # 回测结束日期
        '--initial_cash', '10000000',  # 初始资金
        '--signal_method', 'AND',  # 信号组合方法
        '--data_folder', r"C:\Users\sirui\Desktop\机器学习多因子\data02",  # 数据文件夹路径
        '--upper_pos', '80',  # 上分位数阈值
        '--down_pos', '20'    # 下分位数阈值
    ]
    if use_technical:
        sys_argv.append('--use_technical')
        sys_argv.append('--technical_indicator')
        sys_argv.append(technical_indicator)

    if realistic_trading:
        sys_argv.append('--realistic_trading')
        
    if not use_optimizer:
        sys_argv.append('--no_optimizer')
        
    sys.argv = sys_argv
    try:
        # 调用main函数
        backtest = main()
        return backtest
    finally:
        # 恢复原始的命令行参数
        sys.argv = original_argv
    

def run_adaboost_example(use_technical=False, technical_indicator='three_ma', realistic_trading=False):
    """
    运行AdaBoost策略示例
    """
    print("===== 运行AdaBoost策略示例 =====")

    
    # 保存原始的命令行参数
    original_argv = sys.argv.copy()
    
    # 设置新的命令行参数以运行AdaBoost策略
    sys_argv = [
        'main.py',  # 脚本名称
        '--strategy', 'adaboost',  # 使用AdaBoost策略
        '--start_date', '2018-01-01',  # 回测开始日期
        '--end_date', '2018-12-31',  # 回测结束日期
        '--initial_cash', '10000000',  # 初始资金
        '--signal_method', 'AND',  # 信号组合方法
        '--data_folder', r"C:\Users\sirui\Desktop\机器学习\data02",  # 数据文件夹路径
        '--upper_pos', '80',  # 上分位数阈值
        '--down_pos', '20'    # 下分位数阈值
    ]
    
    # 根据参数添加技术指标选项
    if use_technical:
        sys_argv.append('--use_technical')
        sys_argv.append('--technical_indicator')
        sys_argv.append(technical_indicator)
    
    # 添加交易成本选项
    if realistic_trading:
        sys_argv.append('--realistic_trading')
        
    sys.argv = sys_argv
    try:
        # 调用main函数
        backtest = main()
        return backtest
    finally:
        # 恢复原始的命令行参数
        sys.argv = original_argv

def compare_strategies():
    """
    比较不同策略的性能
    """
    print("===== 比较不同策略的性能 =====")
    
    # 运行各个策略并收集性能指标
    print("\n运行LSTM策略...")
    lstm_backtest = run_lstm_example()
    lstm_performance = lstm_backtest.performance
    
    print("\n运行RandomForest策略...")
    rf_backtest = run_rf_line3_example()
    rf_performance = rf_backtest.performance
    
    print("\n运行AdaBoost策略...")
    adaboost_backtest = run_adaboost_example()
    adaboost_performance = adaboost_backtest.performance
    
    # 创建性能指标对比表
    metrics = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    metrics_names = ['总收益率', '年化收益率', '波动率', '夏普比率', '最大回撤']
    
    # 创建DataFrame存储性能指标
    performance_df = pd.DataFrame(index=metrics_names, columns=['LSTM', 'RandomForest', 'AdaBoost'])
    
    # 填充数据
    for i, metric in enumerate(metrics):
        performance_df.loc[metrics_names[i], 'LSTM'] = lstm_performance.get(metric, np.nan)
        performance_df.loc[metrics_names[i], 'RandomForest'] = rf_performance.get(metric, np.nan)
        performance_df.loc[metrics_names[i], 'AdaBoost'] = adaboost_performance.get(metric, np.nan)
    
    # 格式化百分比
    for metric in ['总收益率', '年化收益率', '波动率', '最大回撤']:
        performance_df.loc[metric] = performance_df.loc[metric] * 100
    
    # 打印性能指标对比表
    print("\n===== 策略性能对比 =====")
    print(performance_df.to_string(float_format=lambda x: f"{x:.2f}"))
    
    # 绘制资产价值对比曲线
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    
    # 将日期转换为相同格式以便比较
    lstm_dates = pd.to_datetime([d.strftime('%Y-%m-%d') for d in lstm_backtest.dates])
    rf_dates = pd.to_datetime([d.strftime('%Y-%m-%d') for d in rf_backtest.dates])
    adaboost_dates = pd.to_datetime([d.strftime('%Y-%m-%d') for d in adaboost_backtest.dates])
    
    # 绘制资产价值曲线
    plt.plot(lstm_dates, lstm_backtest.portfolio_values, label='LSTM')
    plt.plot(rf_dates, rf_backtest.portfolio_values, label='RandomForest')
    plt.plot(adaboost_dates, adaboost_backtest.portfolio_values, label='AdaBoost')
    
    plt.title('策略资产价值对比')
    plt.xlabel('日期')
    plt.ylabel('资产价值')
    plt.legend()
    plt.grid(True)
    
    # 绘制收益率对比曲线
    plt.subplot(2, 1, 2)
    
    # 计算收益率
    lstm_returns = np.array(lstm_backtest.portfolio_values) / lstm_backtest.initial_cash - 1
    rf_returns = np.array(rf_backtest.portfolio_values) / rf_backtest.initial_cash - 1
    adaboost_returns = np.array(adaboost_backtest.portfolio_values) / adaboost_backtest.initial_cash - 1
    
    # 绘制收益率曲线
    plt.plot(lstm_dates, lstm_returns * 100, label='LSTM')
    plt.plot(rf_dates, rf_returns * 100, label='RandomForest')
    plt.plot(adaboost_dates, adaboost_returns * 100, label='AdaBoost')
    
    plt.title('策略收益率对比')
    plt.xlabel('日期')
    plt.ylabel('收益率(%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'lstm': lstm_backtest,
        'rf': rf_backtest,
        'adaboost': adaboost_backtest,
        'performance_df': performance_df
    }


if __name__ == "__main__":
    # 根据需要选择运行单个策略或比较策略
    # run_lstm_example(use_technical=False, technical_indicator='three_ma', realistic_trading=False, use_optimizer=False)
    # run_rf_line3_example(use_technical=True, technical_indicator='three_ma', realistic_trading=False, use_optimizer=True)
    # run_adaboost_example(use_technical=False)
    # results = compare_strategies()
    run_xgboost_example(use_technical=False, realistic_trading=False, use_optimizer=True, use_pca=False)
    
   