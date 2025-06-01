"""
Author: Hugo
Date: 2024-10-26 21:31:21
LastEditors: shen.lan123@gmail.com
LastEditTime: 2024-10-28 16:24:06
Description: 用于运行自定义回测策略的模板
高层调用接口：设置参数，喂数据，评估指标
"""

from my_backtesting.my_engine import BackTesting
import pandas as pd
import numpy as np
from typing import Dict, List, Type, Union, Any
from loguru import logger

__all__ = ["run_template_strategy", "COMMISSION"]

# 设置初始金额及手续费
COMMISSION: Dict = dict(
    cash=1e8, 
    commission=0.00015, 
    stamp_duty=0.0001, 
    slippage_perc=0.0001
)

# 设置策略参数
STRATEGY_PARAMS: Dict = {"verbose": False, "hold_num": 1}

# 设置交易费用
TRADE_PARAMS: Dict = dict(
    cost_buy=0.00015,   # 买入手续费
    cost_sell=0.00115,  # 卖出手续费(包含印花税)
    min_cost=5.0        # 最低手续费
)


def update_params(default_params: Dict, custom_params: Dict) -> Dict:
    """合并默认参数和自定义参数"""
    if custom_params is None:
        return default_params.copy()
    result = default_params.copy()
    result.update(custom_params)
    return result


def run_template_strategy(
    data: pd.DataFrame,
    code: Union[str, List[str]],
    strategy_class: Type,
    strategy_kwargs: Dict = None,
    commission_kwargs: Dict = None,
    trade_params: Dict = None
):
    """
    运行自定义回测策略的模板函数
    
    参数:
        data (pd.DataFrame): 回测数据
        code (str | List[str]): 股票代码或代码列表
        strategy_class (Type): 策略类
        strategy_kwargs (Dict): 策略参数
        commission_kwargs (Dict): 佣金参数
        trade_params (Dict): 交易参数
        
    返回:
        Dict: 回测结果
    """
    # 更新参数
    commission_kwargs = update_params(COMMISSION, commission_kwargs)
    strategy_kwargs = update_params(STRATEGY_PARAMS, strategy_kwargs)
    trade_params = update_params(TRADE_PARAMS, trade_params)
    
    # 根据代码过滤数据
    if isinstance(code, str):
        df = data.query("code == @code").copy()
    elif isinstance(code, list):
        df = data.query("code in @code").copy()
        strategy_kwargs["hold_num"] = len(code)
    else:
        raise ValueError("code must be a string or list of strings")
    
    # 去掉空值
    if "close" in df.columns and "upperbound" in df.columns:
        df = df.dropna(subset=["close", "upperbound","signal","lowerbound"])
    
    # 初始化回测引擎
    
    bt_engine = BackTesting(**commission_kwargs, trade_params=trade_params)
    
    # 加载数据
    required_columns = ["open", "high", "low", "close", "volume", "upperbound", "signal", "lowerbound"]
    bt_engine.load_data(df, required_columns=required_columns)
    
    # 添加策略
    bt_engine.add_strategy(strategy_class, **strategy_kwargs)
    result = bt_engine.run()

    logger.info(f"回测完成: {strategy_class.__name__} on {code if isinstance(code, str) else '+'.join(code)}")
    
        
    return result
    
   