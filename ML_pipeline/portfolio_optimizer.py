"""
portfolio_optimizer.py
负责根据机器学习预测的预期收益率优化投资组合权重分配
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    投资组合优化器，负责计算新股票的资金分配权重
    基于预期收益率和风险进行优化，确保权重之和为1
    均值方差优化
    """
    
    def __init__(self, risk_aversion=1.0, max_weight=0.1, min_weight=0.01):
        """
        初始化投资组合优化器
        
        参数:
        risk_aversion: 风险厌恶系数，越高代表越注重风险控制
        max_weight: 单只股票的最大权重限制
        min_weight: 单只股票的最小权重限制
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def _objective_function(self, weights, expected_returns, cov_matrix):
        """
        目标函数：最大化效用函数 E(R) - λ * σ²
        
        参数:
        weights: 权重数组
        expected_returns: 预期收益率数组
        cov_matrix: 协方差矩阵
        
        返回:
        负的效用函数值（因为scipy.optimize.minimize是最小化函数）
        """
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # 负的效用函数（我们要最大化效用，但minimize函数是最小化的）
        return -1 * (portfolio_return - self.risk_aversion * portfolio_variance)
    
    def _calculate_covariance_matrix(self, historical_data, stock_ids):
        """
        计算历史数据的协方差矩阵，处理不同长度的时间序列
        
        参数:
        historical_data: 历史数据字典 {stock_id: [历史价格]}
        stock_ids: 股票ID列表

        返回:
        协方差矩阵
        """
        n_stocks = len(stock_ids)
        
        # 如果没有历史数据，返回单位矩阵
        if not historical_data:
            return np.eye(n_stocks)
        
        # 找出所有序列的最大长度并获取有效的股票列表
        valid_stocks = []
        max_length = 0
        stock_data = {}
        
        for stock_id in stock_ids:
            data = historical_data.get(stock_id, [])
            if len(data) > 1:  # 至少需要两个数据点
                valid_stocks.append(stock_id)
                max_length = max(max_length, len(data))
                stock_data[stock_id] = data
        
        # 如果没有足够的历史数据，使用单位矩阵
        if len(valid_stocks) < 2:
            return np.eye(n_stocks)
        
        # 创建一个均匀的矩阵，并填充数据（右对齐）
        returns_matrix = np.zeros((len(valid_stocks), max_length-1))
        
        for i, stock_id in enumerate(valid_stocks):
            data = stock_data[stock_id]
            # 计算收益率
            returns = np.diff(data) / data[:-1]
            # 填充returns_matrix的右侧
            returns_matrix[i, -(len(returns)):] = returns
        
        # 处理可能的NaN值
        returns_matrix = np.nan_to_num(returns_matrix, nan=0.0)
        
        # 计算协方差矩阵
        valid_cov_matrix = np.cov(returns_matrix)
        
        # 处理可能的奇异矩阵
        if np.isnan(valid_cov_matrix).any() or np.linalg.det(valid_cov_matrix) < 1e-10:
            valid_cov_matrix += np.eye(len(valid_stocks)) * 0.01
        
        # 创建完整的协方差矩阵（包括没有历史数据的股票）
        full_cov_matrix = np.eye(n_stocks)
        
        valid_indices = [stock_ids.index(stock) for stock in valid_stocks]
        for i, vi in enumerate(valid_indices):
            for j, vj in enumerate(valid_indices):
                full_cov_matrix[vi, vj] = valid_cov_matrix[i, j]
        
        return full_cov_matrix
    
    def optimize_weights(self, expected_returns, historical_data=None):
        """
        优化投资组合权重
        
        参数:
        expected_returns: 预期收益率字典 {stock_id: expected_return}
        historical_data: 历史数据字典 {stock_id: [历史价格]}，用于计算协方差矩阵
        
        返回:
        优化后的权重字典 {stock_id: weight}
        """
        n_stocks = len(expected_returns)
        stock_ids = list(expected_returns.keys())
        returns_array = np.array([expected_returns[stock_id] for stock_id in stock_ids])
        
        # 计算协方差矩阵
        if historical_data is not None:
            cov_matrix = self._calculate_covariance_matrix(historical_data, stock_ids)
        else:
            # 如果没有历史数据，假设股票之间不相关（对角协方差矩阵）
            cov_matrix = np.eye(n_stocks)
        
        # 初始权重：等权重分配
        initial_weights = np.ones(n_stocks) / n_stocks
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        ]
        
        # 边界条件：每只股票的权重在min_weight和max_weight之间
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_stocks)]
        
        # 使用scipy的优化器找到最优权重
        result = minimize(
            self._objective_function,
            initial_weights,
            args=(returns_array, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # 如果优化失败，使用初始等权重分配
        if not result.success:
            print("优化失败，使用等权重分配")
            optimized_weights = initial_weights
        else:
            optimized_weights = result.x
        
        # 归一化权重确保总和为1
        optimized_weights = optimized_weights / np.sum(optimized_weights)
        
        # 创建股票ID到权重的映射
        weight_dict = {stock_id: weight for stock_id, weight in zip(stock_ids, optimized_weights)}
        
        return weight_dict


class RiskParityOptimizer(PortfolioOptimizer):
    """
    风险平价优化器，基于每只股票对总风险的贡献相等的原则进行资金分配
    """
    
    def _risk_parity_objective(self, weights, cov_matrix):
        """
        风险平价目标函数
        
        参数:
        weights: 权重数组
        cov_matrix: 协方差矩阵
        
        返回:
        风险贡献不平等的惩罚值
        """
        # 计算总体投资组合风险（波动率）
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 计算每只股票的风险贡献
        risk_contribution = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
        
        # 计算每只股票的目标风险贡献（相等）
        target_risk = portfolio_vol / len(weights)
        
        # 风险贡献与目标风险的差异的平方和
        risk_diff = np.sum((risk_contribution - target_risk)**2)
        
        return risk_diff
    
    def optimize_weights(self, expected_returns, historical_data=None):
        """
        使用风险平价方法优化投资组合权重
        
        参数:
        expected_returns: 预期收益率字典 {stock_id: expected_return}
        historical_data: 历史数据字典 {stock_id: [历史价格]}，用于计算协方差矩阵
        
        返回:
        优化后的权重字典 {stock_id: weight}
        """
        n_stocks = len(expected_returns)
        stock_ids = list(expected_returns.keys())
        
        # 计算协方差矩阵
        if historical_data is not None:
            cov_matrix = self._calculate_covariance_matrix(historical_data, stock_ids)
        else:
            cov_matrix = np.eye(n_stocks)
        
        # 初始权重：等权重分配
        initial_weights = np.ones(n_stocks) / n_stocks
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_stocks)]
        
        # 优化
        result = minimize(
            self._risk_parity_objective,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # 处理优化结果
        if not result.success:
            print("风险平价优化失败，使用等权重分配")
            optimized_weights = initial_weights
        else:
            optimized_weights = result.x
        
        # 归一化权重
        optimized_weights = optimized_weights / np.sum(optimized_weights)
        
        # 创建股票ID到权重的映射
        weight_dict = {stock_id: weight for stock_id, weight in zip(stock_ids, optimized_weights)}
        
        return weight_dict


class MomentumOptimizer(PortfolioOptimizer):
    """
    动量优化器，基于预期收益率和历史动量进行加权
    """
    
    def __init__(self, risk_aversion=1.0, max_weight=0.1, min_weight=0.01, momentum_weight=0.5):
        """
        初始化动量优化器
        
        参数:
        risk_aversion: 风险厌恶系数
        max_weight: 单只股票的最大权重限制
        min_weight: 单只股票的最小权重限制
        momentum_weight: 动量因子的权重
        """
        super().__init__(risk_aversion, max_weight, min_weight)
        self.momentum_weight = momentum_weight
    
    def _calculate_momentum(self, historical_data):
        """
        计算每只股票的动量得分
        
        参数:
        historical_data: 历史数据字典 {stock_id: [历史价格]}
        
        返回:
        动量得分字典 {stock_id: momentum_score}
        """
        momentum_scores = {}
        
        for stock_id, data in historical_data.items():
            if len(data) < 2:
                momentum_scores[stock_id] = 0
                continue
                
            # 简单的动量计算：最近数据点相对于历史平均的增长率
            if data[-1] > 0 and np.mean(data) > 0:
                momentum = data[-1] / np.mean(data) - 1
            else:
                momentum = 0
                
            momentum_scores[stock_id] = momentum
        
        return momentum_scores
    
    def optimize_weights(self, expected_returns, historical_data=None):
        """
        结合动量因子优化投资组合权重
        
        参数:
        expected_returns: 预期收益率字典 {stock_id: expected_return}
        historical_data: 历史数据字典 {stock_id: [历史价格]}
        
        返回:
        优化后的权重字典 {stock_id: weight}
        """
        if historical_data is None:
            # 如果没有历史数据，退化为基本的优化方法
            return super().optimize_weights(expected_returns)
        
        # 计算动量得分
        momentum_scores = self._calculate_momentum(historical_data)
        
        # 将动量得分标准化到[0, 1]区间
        min_momentum = min(momentum_scores.values())
        max_momentum = max(momentum_scores.values())
        momentum_range = max_momentum - min_momentum
        
        normalized_momentum = {}
        for stock_id, score in momentum_scores.items():
            if momentum_range > 0:
                normalized_momentum[stock_id] = (score - min_momentum) / momentum_range
            else:
                normalized_momentum[stock_id] = 0.5  # 如果所有动量都相同，给予相同权重
        
        # 结合预期收益率和动量得分
        combined_scores = {}
        for stock_id in expected_returns:
            momentum_score = normalized_momentum.get(stock_id, 0)
            return_score = expected_returns[stock_id]
            combined_scores[stock_id] = (1 - self.momentum_weight) * return_score + self.momentum_weight * momentum_score
        
        # 使用组合得分进行优化
        return super().optimize_weights(combined_scores, historical_data)