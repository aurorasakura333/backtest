from copy import deepcopy

class Account:
    """
    输入参数:
    init_cash: float - 初始资金
    position: dict {'code': volume} - 持仓字典
    available: dict {'code': volume} - 可用持仓字典
    init_price: dict {'code': price} - 初始价格字典
    order: dict {
                    'buy':{
                            'code': volume, - 买入订单
                        },
                    'sell':{
                            'code': volume, - 卖出订单
                        }
                    }
    price: dict {'code': price} - 当前价格字典
    cost_rate: float - 交易费率
    min_cost: int - 最低交易费用
    freq: int - 多少个时间点后自动平仓
    risk_degree: float - 最大风险度
    """

    def __init__(self, init_cash: float, position: dict, available: dict, init_price: dict):
        self.cash = init_cash  # 可用资金
        # self.cash_available = deepcopy(init_cash)  # 注释掉的可用资金
        self.position = position  # 持仓字典
        self.available = available  # 可用持仓，考虑T+1系统，今天借入的股票明天才能卖出，不能做空
        self.price = init_price  # 资产价格
        self.value = deepcopy(init_cash)  # 市值，包括现金和持仓市值
        self.cost = 0.0  # 交易费用
        self.val_hist = []  # 用于绘制市值曲线
        self.buy_hist = []  # 买入记录
        self.sell_hist = []  # 卖出记录
        self.risk = None  # 风险度
        self.risk_curve = []  # 风险度曲线
        self.turnover = []  # 换手率
        self.trade_value = 0.0  # 交易金额
        self.date = None  # 当前日期

    def check_order(self, order: dict, price: dict, cost_rate: float = 0.0015, min_cost: float = 5,
                risk: float = 0.95) -> dict:
        """
        检查是否有足够的资金完成订单。如果没有，调整订单
        （卖出保持不变，买入按比例减少）。
        """
        # 设置调试日志
        debug_file = 'check_order_debug.txt'
        
        def debug_log(msg):
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"{msg}\n")
                f.flush()
        
        # 初始日志记录
        debug_log("\n" + "="*50)
        debug_log({self.date})
        debug_log(f"CHECK_ORDER被调用")
        debug_log(f"初始订单: {order}")
        debug_log(f"当前现金: {self.cash:.2f}")
        debug_log(f"当前持仓: {self.position}")
        
        cash_inflow = 0.0  # 现金流入
        cash_outflow = 0.0  # 现金流出
        order_copy = deepcopy(order)  # 订单深拷贝
        
        # 检查卖出订单 - 保持不变
        debug_log("\n处理卖出订单:")
        for code in order_copy['sell'].keys():
            if code in price.keys() and code in self.available.keys():
                # 确保不卖出超过可用数量
                original_amount = order["sell"][code]
                order["sell"][code] = min(order["sell"][code], self.available[code])
                
                if original_amount != order["sell"][code]:
                    debug_log(f"  ⚠️ 调整{code}的卖出: {original_amount} → {order['sell'][code]} (受可用量限制)")
                
                sell_value = price[code] * order['sell'][code]  # 卖出价值
                cash_inflow += sell_value
                debug_log(f"  卖出 {code}: {order['sell'][code]} 股 @ {price[code]:.2f} = {sell_value:.2f}")
            else:
                debug_log(f"  ❌ 从卖出订单中移除 {code} - 不在可用资产中或价格缺失")
                order["sell"].pop(code)
        
        # 计算卖出后的预估现金（考虑交易成本）
        sell_cost = max(min_cost, cash_inflow * cost_rate) if cash_inflow > 0 else 0  # 卖出成本
        estimated_cash = self.cash + cash_inflow - sell_cost  # 卖出后估计现金
        
        debug_log(f"\n卖出后:")
        debug_log(f"  总卖出价值: {cash_inflow:.2f}")
        debug_log(f"  卖出成本: {sell_cost:.2f}")
        debug_log(f"  卖出后估计现金: {estimated_cash:.2f}")
        
        # 计算可用于买入股票的现金
        available_cash_for_buys = estimated_cash * risk  # 风险调整后可用于买入的现金
        debug_log(f"  可用于买入的现金 (风险调整 {risk*100:.0f}%): {available_cash_for_buys:.2f}")
        
        # 识别具有占位符值(1)的买入订单，用于等权重分配
        buy_placeholder_codes = [code for code, volume in order_copy['buy'].items() 
                                if volume == 1 and code in price.keys() and price[code] > 0]
        
        # 如果有带占位符值的买入订单，应用等权重分配
        debug_log("\n处理买入订单:")
        if buy_placeholder_codes:
            debug_log(f"  发现{len(buy_placeholder_codes)}只带有占位符买入信号(1)的股票: {buy_placeholder_codes}")
            
            # 计算每只股票可用的现金（等额分配）
            cash_per_stock = available_cash_for_buys / len(buy_placeholder_codes)  # 每只股票等额分配的现金
            debug_log(f"  每只股票等额分配: {cash_per_stock:.2f}")
            
            # 基于等额分配计算每只股票的数量
            for code in buy_placeholder_codes:
                # 计算要买入的股数（四舍五入到最接近的100股手数）
                max_shares_raw = cash_per_stock / price[code]  # 原始可买入股数
                max_shares_lot = int((max_shares_raw / 100)) * 100  # 舍入到100股的手数
                
                debug_log(f"  {code}: 价格={price[code]:.2f}, 最大股数={max_shares_raw:.2f}, 舍入到手数={max_shares_lot}")
                
                if max_shares_lot > 0:
                    order["buy"][code] = max_shares_lot
                    debug_log(f"  ✅ 设置{code}的买入数量: {max_shares_lot} 股 (价值: {max_shares_lot * price[code]:.2f})")
                else:
                    # 如果我们甚至负担不起一手，从买入字典中移除
                    if code in order["buy"]:
                        debug_log(f"  ❌ 从买入订单中移除 {code} - 无法负担一手")
                        order["buy"].pop(code)
        else:
            debug_log("  未发现占位符买入信号，使用原始买入数量")
        
        # 用更新后的数量计算最终现金流
        cash_outflow = sum(price.get(code, 0) * volume for code, volume in order["buy"].items())  # 总买入价值
        buy_cost = max(min_cost, cash_outflow * cost_rate) if cash_outflow > 0 else 0  # 买入成本
        total_buy_cost = cash_outflow + buy_cost  # 总买入成本（价值+费用）
        
        debug_log(f"\n最终检查:")
        debug_log(f"  总买入价值: {cash_outflow:.2f}")
        debug_log(f"  买入成本: {buy_cost:.2f}")
        debug_log(f"  总买入成本 (价值 + 费用): {total_buy_cost:.2f}")
        debug_log(f"  可用于买入的现金: {available_cash_for_buys:.2f}")
        
        # 最终检查 - 如果我们仍然超过可用现金，按比例减少所有买入订单
        if total_buy_cost > available_cash_for_buys and cash_outflow > 0:
            debug_log(f"  ⚠️ 买入成本超过可用现金，按比例减少订单")
            ratio = available_cash_for_buys / total_buy_cost  # 缩减比率
            debug_log(f"  缩减比率: {ratio:.4f}")
            
            # 按比例减少买入订单，保持最接近的100股
            for code in list(order["buy"].keys()):
                original_volume = order["buy"][code]  # 原始数量
                adjusted_volume = int(((original_volume * ratio) / 100) * 100)  # 调整后数量
                
                debug_log(f"  {code}: 原始={original_volume}, 调整后={adjusted_volume}")
                
                if adjusted_volume > 0:
                    order["buy"][code] = adjusted_volume
                else:
                    # 移除调整后变为0的任何买入订单
                    debug_log(f"  ❌ 从买入订单中移除 {code} - 调整后数量变为0")
                    order["buy"].pop(code)
        
        # 移除任何数量为零或负数的买入订单
        original_buy_len = len(order["buy"])
        order["buy"] = {k: v for k, v in order["buy"].items() if v > 0}
        if len(order["buy"]) < original_buy_len:
            debug_log(f"  移除了 {original_buy_len - len(order['buy'])} 个数量为零或负数的买入订单")
        
        original_sell_len = len(order["sell"])
        order["sell"] = {k: v for k, v in order["sell"].items() if v > 0}
        if len(order["sell"]) < original_sell_len:
            debug_log(f"  移除了 {original_sell_len - len(order['sell'])} 个数量为零或负数的卖出订单")
        
        # 最终总结
        debug_log("\n最终订单:")
        debug_log(f"  卖出订单: {order['sell']}")
        debug_log(f"  买入订单: {order['buy']}")
        
        return order
        
    def calculate_equal_weights(self, buy_signals: dict, price: dict, risk: float = 0.95):
        """
        计算买入信号的等权重数量。
    
        """
        # 筛选有效股票（有价格的股票）
        valid_stocks = {code: buy_signals[code] for code in buy_signals if code in price and price[code] > 0}
        
        if not valid_stocks:
            return {}
            
        # 计算可用现金
        available_cash = self.cash * risk  # 按风险度调整的可用现金
        
        # 平均分配现金给各股票
        num_stocks = len(valid_stocks)  # 股票数量
        cash_per_stock = available_cash / num_stocks  # 每只股票分配的现金
        
        # 计算每只股票的买入股数
        calculated_quantities = {}
        for code in valid_stocks:
            # 计算买入股数（四舍五入到最接近的100股手数）
            shares = int((cash_per_stock / price[code]) / 100) * 100  # 计算股数并舍入到100的整数倍
            if shares > 0:
                calculated_quantities[code] = shares
                
        return calculated_quantities

    def update_price(self, price: dict):  # 更新市场价格
        for code in price.keys():
            self.price[code] = price[code]

    def update_value(self):  # 更新市值
        value_hold = 0.0  # 持仓市值
        for code in self.price.keys():  # 更新持仓市值
            if code in self.position.keys():
                value_hold += self.position[code] * self.price[code]
        self.value = self.cash + value_hold  # 总市值 = 现金 + 持仓市值
        self.val_hist.append(self.value)  # 记录市值历史

    def update_trade_hist(self, order: dict):  # 更新交易记录
        if order is not None:
            self.buy_hist.append(order['buy'])  # 添加买入记录
            self.sell_hist.append(order['sell'])  # 添加卖出记录
        else:
            self.buy_hist.append({})
            self.sell_hist.append({})

    def buy(self, order_buy: dict, cost_rate: float = 0.0015, min_cost: float = 5):  # 买入函数
        buy_value = 0.0  # 买入价值
        for code in order_buy.keys():
            if code in self.position.keys():
                self.position[code] += order_buy[code]  # 更新持仓
                self.available[code] += order_buy[code]  # 更新可用持仓
            else:
                self.position[code] = order_buy[code]  # 新建持仓
                self.available[code] = order_buy[code]  # 新建可用持仓
            buy_value += self.price[code] * order_buy[code]  # 累计买入价值
        self.trade_value += buy_value  # 累计交易价值
        cost = max(min_cost, buy_value * cost_rate)  # 计算交易成本
        self.cost += cost  # 累计总成本
        self.cash -= (buy_value + cost)  # 更新现金

    def sell(self, order_sell: dict, cost_rate: float = 0.0005, min_cost: float = 5):  # 卖出函数
        sell_value = 0.0  # 卖出价值
        for code in order_sell.keys():
            if code in self.available.keys():
                self.position[code] -= order_sell[code]  # 减少持仓
                self.available[code] -= order_sell[code]  # 减少可用持仓
                sell_value += self.price[code] * order_sell[code]  # 累计卖出价值
        self.trade_value += sell_value  # 累计交易价值
        cost = max(min_cost, sell_value * cost_rate) if sell_value != 0 else 0  # 计算交易成本
        self.cash += (sell_value - cost)  # 更新现金

    def update_all(self, order, price, cost_buy=0.0015, cost_sell=0.0005, min_cost=5, trade=True):
        # 更新市场价格、交易记录、持仓和可交易数量、交易费用和现金、市值
        self.update_price(price)  # 首先更新市场价格
        self.trade_value = 0.0  # 重置交易价值
        value_before_trade = self.value  # 交易前市值
        if order is not None:
            if trade:
                # 然后更新交易记录
                self.update_trade_hist(order)  # 更新交易历史
                self.sell(order["sell"], cost_buy, min_cost)  # 执行卖出
                self.buy(order["buy"], cost_sell, min_cost)  # 执行买入
        self.trade_value = self.trade_value  # 重新赋值交易价值（可能多余）
        self.turnover.append(self.trade_value / value_before_trade / 2)  # 根据基金行业资格材料，应为(交易量/2)/平均净资产
        self.update_value()  # 更新持仓市值

    def risk_control(self, risk_degree: float = 0.95, cost_rate: float = 0.0005, min_cost: float = 5):
        """
        风险控制，当风险度超过计划风险度时，按比例减少持仓
        """
        self.risk = 1 - self.cash / self.value  # 非现金部分占总资产的比例即为风险度
        if self.risk > risk_degree:  # 如果风险度超过设定的风险度
            b = 1 - risk_degree / self.risk  # 计算减仓比例b
            sell_order = deepcopy(self.position)  # 复制当前持仓
            sell_order = {k: v for k, v in sell_order.items() if v > 0}  # 只保留持仓为正的股票
            for code in sell_order.keys():
                sell_order[code] *= b  # 当前持仓乘以减仓比例b，得到要卖出的数量
                sell_order[code] = int(sell_order[code] / 100 + 0.5) * 100  # 按手数进行减仓（四舍五入到最接近的100股）
            self.sell(sell_order, cost_rate, min_cost)  # 执行卖出
            self.risk = 1 - self.cash / self.value  # 更新风险度
        self.risk_curve.append(self.risk)  # 记录风险度曲线