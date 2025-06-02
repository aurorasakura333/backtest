"""
main.py
主要执行脚本，用于运行机器学习选股策略
"""
import sys
import os
sys.path.append("C:\\Users\\sirui\\Desktop\\机器学习多因子\\TIDIBEI-master")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# 导入自定义模块
from ML_pipeline.data_processor import DataLoader, DataProcessor, TradingCalendar
from ML_pipeline.ml_strategies import LSTMStrategy, RandomForestStrategy, AdaBoostStrategy, XGBoostStrategy
from ML_pipeline.signal_generator import SignalGenerator
from ML_pipeline.backtest_framework import BacktestFramework

def get_df():
    import pandas as pd
    import os

    # 设定文件夹路径
    folder_path = r"C:\Users\sirui\Desktop\机器学习多因子\data02"

    # 需要删除的列,保留code_id,date,close_qfq,所有基本面，指标因子
    columns_to_drop = ['open', 'open_hfq', 'open_qfq', 'high', 'high_hfq', 'high_qfq', 
                    'low', 'low_hfq', 'low_qfq', 'close', 'close_hfq', 'pre_close', 'pct_chg',
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

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='机器学习选股策略回测')
    
    # 数据参数
    parser.add_argument('--data_folder', type=str, default=r"C:\Users\sirui\Desktop\机器学习\data02", 
                        help='数据文件夹路径')
    
    # 策略参数
    parser.add_argument('--strategy', type=str, default='lstm', 
                        choices=['lstm', 'rf', 'adaboost', 'xgboost'],  # 添加xgboost选项
                        help='使用的策略')
    parser.add_argument('--use_technical', action='store_true', default=False, 
                        help='是否使用技术指标信号')
    parser.add_argument('--signal_method', type=str, default='AND', choices=['AND', 'OR'], 
                        help='信号组合方法')
    parser.add_argument('--technical_indicator', type=str, default='three_ma', 
                        choices=['three_ma', 'ma_cross', 'bollinger'], 
                        help='使用的技术指标策略')
    
    # 添加这两个参数
    parser.add_argument('--upper_pos', type=float, default=80.0, 
                        help='上分位数阈值，高于则买入')
    parser.add_argument('--down_pos', type=float, default=20.0, 
                        help='下分位数阈值，低于则卖出')
    
    # 回测参数
    parser.add_argument('--start_date', type=str, default='2016-01-01', 
                        help='回测开始日期')
    parser.add_argument('--end_date', type=str, default='2018-09-30', 
                        help='回测结束日期')
    parser.add_argument('--initial_cash', type=float, default=10000000, 
                        help='初始资金')

    # 交易成本参数
    parser.add_argument('--realistic_trading', action='store_true', default=False,
                        help='是否启用真实交易模式（考虑手续费和滑点）')
    parser.add_argument('--transaction_fee_rate', type=float, default=0.0003,
                        help='交易费率，默认为0.0003（万分之三）')
    parser.add_argument('--slippage_rate', type=float, default=0.0001,
                        help='滑点比例，默认为0.0001（万分之一）')
    
    # 添加是否使用投资组合优化器的参数
    parser.add_argument('--use_optimizer', action='store_true', default=True,
                        help='是否使用投资组合优化器分配资金')
    parser.add_argument('--no_optimizer', dest='use_optimizer', action='store_false',
                        help='不使用优化器，采用等权重分配资金')
    
    # XGBoost特定参数
    parser.add_argument('--use_pca', action='store_true', default=False,
                        help='是否对特征使用PCA降维')
    parser.add_argument('--n_components', type=int, default=20,
                        help='PCA降维后的组件数量')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("===== 机器学习选股策略回测 =====")
    print(f"策略: {args.strategy}")
    print(f"回测时间段: {args.start_date} 至 {args.end_date}")
    print(f"数据文件夹: {args.data_folder}")
    print(f"使用技术指标: {args.use_technical}")
    print(f"信号组合方法: {args.signal_method}")
    print(f"初始资金: {args.initial_cash}")
    print(f"上分位数阈值: {args.upper_pos}")
    print(f"下分位数阈值: {args.down_pos}")
    
    # 交易成本信息
    if args.realistic_trading:
        print("\n====== 真实交易模式 ======")
        print(f"交易费率: {args.transaction_fee_rate*100:.4f}%")
        print(f"滑点比例: {args.slippage_rate*100:.4f}%")
    else:
        print("\n使用向量化回测模式（不考虑交易成本）")
    
    # 加载数据
    print("\n正在加载数据...")
    data = get_df()
    print(f"加载完成，数据形状: {data.shape}")
    
    # 因子代码列表
    factor_codes =  [col for col in data.columns if col not in ['close_qfq','pct_chg']]
    
    # 创建策略实例
    print("\n初始化策略...")
    if args.strategy == 'lstm':
        strategy = LSTMStrategy(
            factor_codes=factor_codes,
            seq_length=21,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            lr=0.01,
            batch_size=5,
            epochs=3,
            upper_pos=args.upper_pos,
            down_pos=args.down_pos,
            cash_rate=0.6,
            use_pca=args.use_pca,
            n_components=args.n_components
        )
        print("创建LSTM策略")
    elif args.strategy == 'rf':
        strategy = RandomForestStrategy(
            factor_codes=factor_codes,
            n_estimators=50,
            max_depth=5,
            upper_pos=args.upper_pos,
            down_pos=args.down_pos,
            cash_rate=0.6
        )
        print("创建RandomForest策略")
    elif args.strategy == 'adaboost':
        strategy = AdaBoostStrategy(
            factor_codes=factor_codes,
            n_estimators=60,
            max_depth=9,
            learning_rate=1.0,
            use_pca=args.use_pca,
            n_components=args.n_components,
            upper_pos=args.upper_pos,
            down_pos=args.down_pos,
            cash_rate=0.6
        )
        print("创建AdaBoost策略")
    elif args.strategy == 'xgboost':
        strategy = XGBoostStrategy(
            factor_codes=factor_codes,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1,
            use_pca=args.use_pca,
            n_components=args.n_components,
            upper_pos=args.upper_pos,
            down_pos=args.down_pos,
            cash_rate=0.6
        )
        print("创建XGBoost策略")
    else:
        raise ValueError(f"不支持的策略: {args.strategy}")
    
    # 创建信号生成器
    signal_generator = SignalGenerator()
    
    # 创建回测框架
    backtest = BacktestFramework(
        data=data,
        strategy=strategy,
        signal_generator=signal_generator,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        use_technical_signal=args.use_technical,
        signal_method=args.signal_method,
        technical_indicator=args.technical_indicator,
        realistic_trading=args.realistic_trading,
        transaction_fee_rate=args.transaction_fee_rate,
        slippage_rate=args.slippage_rate,
        upper_pos=args.upper_pos,
        down_pos=args.down_pos,
        use_optimizer=args.use_optimizer
    )
    
    # 运行回测
    print("\n开始回测...")
    performance = backtest.run_backtest(display_progress=True)
    
    # 打印回测结果
    backtest.print_performance()
    
    # 如果是XGBoost并且没有使用PCA，显示特征重要性
    if args.strategy == 'xgboost' and not args.use_pca:
        feature_importance = strategy.get_feature_importance()
        if feature_importance:
            print("\n===== 特征重要性 =====")
            for i, (feature, importance) in enumerate(feature_importance.items()):
                if i < 10:  # 仅显示前10个最重要的特征
                    print(f"{feature}: {importance:.6f}")
                else:
                    break
    
    return backtest


if __name__ == "__main__":
    backtest = main()