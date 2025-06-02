"""
ml_strategies.py
机器学习策略的具体实现类
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
#ABC 是 Python 用来定义 “抽象基类” 的工具。abstractmethod 是装饰器，规定子类必须实现某个方法。

# 基础策略抽象类
class BaseMLStrategy(ABC):
    """
    机器学习策略基类，定义了所有ML策略需要实现的接口
    """
    
    def __init__(self, factor_codes, upper_pos=80, down_pos=20, cash_rate=0.6):
        """
        初始化
        
        参数:
        factor_codes: 因子代码列表
        upper_pos: 上分位数阈值，高于则买入
        down_pos: 下分位数阈值，低于则卖出
        cash_rate: 可用资金比例
        """
        self.factor_codes = factor_codes
        self.upper_pos = upper_pos
        self.down_pos = down_pos
        self.cash_rate = cash_rate
        self.model = None
        self.name = "BaseMLStrategy"
    
    @abstractmethod
    def train(self, X, y):
        """
        训练模型
        
        参数:
        X: 训练特征
        y: 训练标签
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        预测
        
        参数:
        X: 预测特征
        
        返回:
        预测结果
        """
        pass
    
    def get_signals(self, predictions, positions):
        """
        根据预测结果和持仓生成交易信号
        
        参数:
        predictions: 预测结果
        positions: 持仓情况
        
        返回:
        买入信号和卖出信号
        """
        # 获取预测收益率的高分位数和低分位数
        high_return, low_return = np.percentile(predictions, [self.upper_pos, self.down_pos])
        
        # 生成买入信号：无持仓且预测收益率高于高阈值
        buy_signals = (positions == 0) & (predictions > high_return)
        
        # 生成卖出信号：有持仓且预测收益率低于低阈值
        sell_signals = (positions > 0) & (predictions < low_return)
        
        return buy_signals, sell_signals, high_return, low_return

# LSTM策略实现
class LSTMStrategy(BaseMLStrategy):
    """
    基于LSTM的机器学习策略，增加了PCA降维功能
    """
    
    class LSTMModel(nn.Module):
        """
        LSTM模型定义
        """
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMStrategy.LSTMModel, self).__init__()
            
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            
            self.out = nn.Linear(hidden_size, 1)
        #最后一层的输出映射为 1 个数值
        def forward(self, x):
            r_out, (h_n, h_c) = self.rnn(x, None)
            out = self.out(r_out[:, -1, :])
            return out
    
    class StockDataset(Dataset):
        """
        股票数据集
        """
        def __init__(self, X, y, seq_length):
            self.X = X
            self.y = y
            self.seq_length = seq_length #样本长度
        
        def __getitem__(self, index):
            head = self.seq_length * index
            tail = self.seq_length * (index + 1)
            data = self.X[head:tail]
            label = self.y[index]
            return torch.FloatTensor(data), torch.FloatTensor([label])
        #切分出样本序列
        def __len__(self):
            return len(self.y)
    
    def __init__(self, factor_codes, seq_length=21, hidden_size=64, num_layers=2, 
                 dropout=0.2, lr=0.01, batch_size=5, epochs=10, 
                 upper_pos=80, down_pos=20, cash_rate=0.6,
                 use_pca=True, n_components=20):  # 添加PCA相关参数
      
        super().__init__(factor_codes, upper_pos, down_pos, cash_rate)
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 添加PCA相关属性
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None
        if use_pca:
            self.pca = PCA(n_components=n_components)
            # LSTM模型的输入维度为PCA降维后的维度
            self.model = self.LSTMModel(n_components, hidden_size, num_layers, dropout)
        else:
            # 不使用PCA时，输入维度为原始因子数量
            self.model = self.LSTMModel(len(factor_codes), hidden_size, num_layers, dropout)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.name = "LSTM"
    
    def train(self, X, y):
        """
        训练LSTM模型，增加PCA处理步骤
        
        参数:
        X: 训练特征，形状为 [样本数 * seq_length, 特征数]
        y: 训练标签，形状为 [样本数]
        """
        if self.use_pca:
            # 保存原始X的形状和特征维度
            original_shape = X.shape
            feature_dim = original_shape[1]
            
            # 使用PCA降维，直接对原始X进行降维而不重塑
            n_components = min(self.n_components, feature_dim)
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            
            # 替换原始X，不需要重塑
            X = X_pca
        
            
        # 创建数据集和数据加载器
        dataset = self.StockDataset(X, y, self.seq_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        # 训练模型
        self.model.train()
        for epoch in range(self.epochs):
            for step, (batch_x, batch_y) in enumerate(dataloader):
                # 前向传播
                output = self.model(batch_x)
                loss = self.loss_func(output, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        """
        使用LSTM模型进行预测，增加PCA处理步骤
        
        参数:
        X: 预测特征，通常形状为 [样本数, seq_length, 特征数]
        
        返回:
        预测结果
        """
        # 如果使用PCA，先进行降维处理
        if self.use_pca:
             X = self.pca.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            return predictions.numpy().flatten()
"""
xgboost_strategy.py
XGBoost策略的实现
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from ML_pipeline.ml_strategies import BaseMLStrategy


# XGBoost策略实现 - 添加到ml_strategies.py文件末尾
class XGBoostStrategy(BaseMLStrategy):
    """
    基于XGBoost的机器学习策略
    """
    
    def __init__(self, factor_codes, n_estimators=100, max_depth=5, learning_rate=0.1, 
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1,
                 use_pca=False, n_components=20, upper_pos=80, down_pos=20, cash_rate=0.6):
        """
        初始化XGBoost策略
        
        参数:
        factor_codes: 因子代码列表
        n_estimators: 使用的树的数量
        max_depth: 树的最大深度
        learning_rate: 学习率
        subsample: 样本采样比例
        colsample_bytree: 特征采样比例
        reg_alpha: L1正则化参数
        reg_lambda: L2正则化参数
        use_pca: 是否使用PCA降维
        n_components: PCA组件数
        upper_pos: 上分位数阈值，高于则买入
        down_pos: 下分位数阈值，低于则卖出
        cash_rate: 可用资金比例
        """
        super().__init__(factor_codes, upper_pos, down_pos, cash_rate)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample   # 采样比例
        self.colsample_bytree = colsample_bytree # 特征采样比例
        self.reg_alpha = reg_alpha  # L1正则化Lasso，稀疏性
        self.reg_lambda = reg_lambda #L2 正则化参数（Ridge，防止过拟合）
        self.use_pca = use_pca
        self.n_components = n_components
        
        # 初始化模型
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=-1,  # 使用所有CPU核心
            random_state=42
        )
        
        # 初始化PCA
        self.pca = None
        if use_pca:
            self.pca = PCA(n_components=n_components)
        
        self.name = "XGBoost"
    
    def train(self, X, y):
        """
        训练XGBoost模型
        
        参数:
        X: 训练特征
        y: 训练标签
        """
        # 检查输入数据的维度
        if len(X.shape) > 2:
            # 如果X是3D数组，则进行reshape成2D
            X = X.reshape(-1, X.shape[-1])
            # 如果y的长度小于X的样本数，需要调整
            if len(y) < X.shape[0]:
                y = np.repeat(y, X.shape[0] // len(y))
        
        # 如果使用PCA，先降维
        if self.use_pca:
            n_components = min(self.n_components, X.shape[1])
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            self.model.fit(X_pca, y)
        else:
            self.model.fit(X, y)
            
        # 获取特征重要性（可选）
        if not self.use_pca:
            self.feature_importances = self.model.feature_importances_
    
    def predict(self, X):
        """
        使用XGBoost模型进行预测
        
        参数:
        X: 预测特征
        
        返回:
        预测结果
        """
        # 检查输入数据的维度
        if len(X.shape) > 2:
            # 如果X是3D数组，则进行reshape成2D
            X = X.reshape(-1, X.shape[-1])
        
        # 如果使用PCA，先降维
        if self.use_pca:
            n_components = min(self.n_components, X.shape[1])
            if self.pca is None or self.pca.n_components_ != n_components:
                self.pca = PCA(n_components=n_components)
                X_pca = self.pca.fit_transform(X)
            else:
                X_pca = self.pca.transform(X)
            return self.model.predict(X_pca)
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        返回:
        特征重要性字典 {特征名: 重要性}
        """
        if self.use_pca or not hasattr(self, 'feature_importances'):
            return None
        
        importance_dict = {}
        for i, code in enumerate(self.factor_codes):
            importance_dict[code] = self.feature_importances[i]
        
        # 按重要性降序排序
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
# 随机森林策略实现
class RandomForestStrategy(BaseMLStrategy):
    """
    基于随机森林的机器学习策略
    """
    
    def __init__(self, factor_codes, n_estimators=50, max_depth=5, 
                 upper_pos=80, down_pos=40, cash_rate=0.6):
        """
        初始化随机森林策略
        
        参数:
        factor_codes: 因子代码列表
        n_estimators: 树的数量
        max_depth: 树的最大深度
        upper_pos: 上分位数阈值
        down_pos: 下分位数阈值
        cash_rate: 可用资金比例
        """
        super().__init__(factor_codes, upper_pos, down_pos, cash_rate)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        self.name = "RandomForest"
    
    def train(self, X, y):
        """
        训练随机森林模型
        
        参数:
        X: 训练特征
        y: 训练标签
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        使用随机森林模型进行预测
        
        参数:
        X: 预测特征
        
        返回:
        预测结果
        """
        return self.model.predict(X)


# AdaBoost策略实现
class AdaBoostStrategy(BaseMLStrategy):
    """
    基于AdaBoost的机器学习策略
    """
    
    def __init__(self, factor_codes, n_estimators=60, max_depth=9, 
                 learning_rate=1.0, use_pca=True, n_components=5,
                 upper_pos=80, down_pos=20, cash_rate=0.6):
        """
        初始化AdaBoost策略
        
        参数:
        factor_codes: 因子代码列表
        n_estimators: 弱学习器数量
        max_depth: 决策树最大深度
        learning_rate: 学习率
        use_pca: 是否使用PCA降维
        n_components: PCA组件数
        upper_pos: 上分位数阈值
        down_pos: 下分位数阈值
        cash_rate: 可用资金比例
        """
        super().__init__(factor_codes, upper_pos, down_pos, cash_rate)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_pca = use_pca
        self.n_components = n_components
        
        # 初始化模型
        self.base_estimator = DecisionTreeRegressor(max_depth=max_depth)
        self.model = AdaBoostRegressor(
            estimator=self.base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=1
        )
        
        # 初始化PCA
        self.pca = None
        if use_pca:
            self.pca = PCA(n_components=n_components)
        
        self.name = "AdaBoost"
    
    def train(self, X, y):
        """
        训练AdaBoost模型
        
        参数:
        X: 训练特征
        y: 训练标签
        """
        # 检查输入数据的维度
        if len(X.shape) > 2:
            # 如果X是3D数组，则进行reshape成2D
            # 假设X的形状为 [samples*seq_length, features] 或 [samples, seq_length, features]
            X = X.reshape(-1, X.shape[-1])
            # 如果y的长度小于X的样本数，需要调整
            if len(y) < X.shape[0]:
                # 复制y以匹配X的样本数
                y = np.repeat(y, X.shape[0] // len(y))
        
        # 如果使用PCA，先降维
        if self.use_pca:
            n_components = min(self.n_components, X.shape[1])
            self.pca = PCA(n_components=n_components)
            # 适配并变换数据
            X_pca = self.pca.fit_transform(X)
            self.model.fit(X_pca, y)
        else:
            self.model.fit(X, y)
    
    def predict(self, X):
        """
        使用AdaBoost模型进行预测
        
        参数:
        X: 预测特征
        
        返回:
        预测结果
        """
        # 检查输入数据的维度
        if len(X.shape) > 2:
            # 如果X是3D数组，则进行reshape成2D
            X = X.reshape(-1, X.shape[-1])
        
        # 如果使用PCA，先降维
        if self.use_pca:
            n_components = min(self.n_components, X.shape[1])
            if self.pca is None or self.pca.n_components_ != n_components:
                self.pca = PCA(n_components=n_components)
                # 需要先fit，因为可能是新的测试数据
                X_pca = self.pca.fit_transform(X)
            else:
                # 使用已经fit过的PCA
                X_pca = self.pca.transform(X)
            return self.model.predict(X_pca)
        else:
            return self.model.predict(X)