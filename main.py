"""
股票价格预测与回测系统
=====================================
功能概述：
1. 从Excel/CSV文件读取股票历史数据（开盘价、最高价、最低价、收盘价）
2. 使用LSTM（长短期记忆网络）对股票收盘价进行预测
3. 基于预测信号，使用Backtrader框架进行回测交易
4. 自动对比不同时间窗口参数的效果，找出最优配置

技术栈：
- 数据处理：pandas, numpy
- 机器学习：TensorFlow/Keras (LSTM神经网络)
- 回测框架：Backtrader
- 可视化：matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # 用于数据归一化，将数值缩放到[0,1]区间
from tensorflow.keras.models import Sequential  # Keras序列模型，用于堆叠神经网络层
from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM层、全连接层、Dropout正则化层
import backtrader as bt  # 回测框架
import re  # 正则表达式，用于数据清洗

# ============================================================
# 第一部分：Matplotlib 中文显示配置
# ============================================================
# 设置中文字体支持，解决图表中文显示乱码问题
# Windows系统优先使用"微软雅黑"，其次是"黑体"，最后是通用字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
# 解决负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 第二部分：交易成本参数配置
# ============================================================
# 这些参数用于模拟真实交易中的成本，帮助更准确地估算收益
COMMISSION_RATE = 0.001   # 交易佣金费率：0.1%（买卖双向收取）
SLIP_PERC = 0.0005        # 滑点：0.05%（成交价与预期价的差异，防止订单被拒绝）

# ============================================================
# 第三部分：数据读取与预处理
# ============================================================
# 数据文件路径（支持Excel格式）
file_path = r"C:\Users\czx66\Desktop\我的大学生涯故事\专业课基础课资料\_3.2 智能课程系统设计\测试数据.xsl"

# Excel表头说明：时间, 开盘, 最高, 最低, 收盘, 涨幅, 振幅, 总手, 金额, 换手%, 成交次数
# 这些列分别代表：
# - 时间：交易日期
# - 开盘：当日开盘价
# - 最高：当日最高价
# - 最低：当日最低价
# - 收盘：当日收盘价（这是我们预测的主要目标）

# 尝试读取Excel文件，如果失败则尝试CSV格式（某些软件导出的格式可能不同）
try:
    # 尝试直接读取Excel文件
    df_raw = pd.read_excel(file_path)
except Exception:
    # 读取失败时，尝试使用CSV格式读取，指定制表符分隔和GBK编码（中文Windows常用）
    df_raw = pd.read_csv(file_path, sep="\t", encoding="gbk", engine="python")

# 清理列名中的空白字符
# 问题：有时Excel列名末尾会有隐藏空格，导致列名不匹配
# 解决：strip()去除首尾空白
df_raw.columns = [str(c).strip() for c in df_raw.columns]

# 检查必需列是否都存在
# 必需的列：时间、开盘、最高、最低、收盘（至少需要这些才能进行分析）
required_cols = ["时间", "开盘", "最高", "最低", "收盘"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    # 如果有缺失列，抛出错误并显示具体缺少哪些列
    raise ValueError(
        f"缺少必要列: {missing}。当前列名为: {list(df_raw.columns)}"
    )

# 提取需要的列，重命名为英文（代码中更方便使用）
df = pd.DataFrame()
df["trade_date"] = df_raw["时间"]  # 交易日期
df["open"] = df_raw["开盘"]         # 开盘价
df["high"] = df_raw["最高"]         # 最高价
df["low"] = df_raw["最低"]          # 最低价
df["close"] = df_raw["收盘"]        # 收盘价（预测目标）

# 清洗日期格式
# 原始日期格式可能包含星期几（如"2024-01-15, 星期一"），需要去掉逗号后的部分
# 使用str.split(",")取逗号前的日期部分
df["trade_date"] = df["trade_date"].astype(str).str.split(",").str[0].str.strip()
# 将字符串转换为datetime格式，errors="coerce"表示转换失败时设为NaT
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

# 清洗价格数据（去掉千分位逗号）
# 问题：中文数字常用逗号作为千分位分隔符（如1,000.00），这会导致字符串无法转为数字
# 解决：先去掉所有逗号，再转为数字
for col in ["open", "high", "low", "close"]:
    df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors="coerce")  # 转为浮点数

# 最终清洗：删除无效数据
df = df.dropna()              # 删除所有包含NaN的行
df = df[df["close"] > 0]      # 删除收盘价为非正数的行（价格必须为正）
df = df.set_index("trade_date").sort_index()  # 设置日期为索引，并按日期排序

# 如果清洗后没有数据，说明原始文件格式有问题
if df.empty:
    raise ValueError(
        "清洗后没有可用数据（df 为空）。请检查原始文件分隔符、列名映射和日期/价格格式。"
    )

print("✅ 读取成功！")
print("数据条数：", len(df))
print("时间范围：", df.index.min(), "→", df.index.max())
# ======================================================================

# ============================================================
# 第四部分：模型与回测参数配置
# ============================================================
"""
参数说明：
- WINDOW_LIST：时间窗口大小列表，用于LSTM模型lookback
  窗口越大，模型能看到的历史数据越多，但需要更多训练数据
- EPOCHS：训练轮数，越多越能拟合数据，但可能过拟合
- BUY_THRESHOLD：买入阈值，预测涨幅超过此值才买入
- SELL_THRESHOLD：卖出阈值，预测跌幅超过此值才卖出
- TAKE_PROFIT：止盈比例，盈利达到此比例时强制卖出
- STOP_LOSS：止损比例，亏损达到此比例时强制卖出
"""
WINDOW_LIST = [10, 15, 20, 30, 50]  # 可调整窗口列表：自动对比不同 window 的效果
EPOCHS = 5                           # 训练轮数
BATCH_SIZE = 16                      # 每批次样本数
BUY_THRESHOLD = 0.01                 # 买入阈值：预测涨幅超过1%时买入
SELL_THRESHOLD = 0.005               # 卖出阈值：预测跌幅超过0.5%时卖出
TAKE_PROFIT = 0.15                   # 止盈：盈利15%时强制卖出
STOP_LOSS = 0.07                     # 止损：亏损7%时强制卖出

# 模型输入特征：后续加特征时只需改这个列表，统一归一化会自动生效
# 当前使用4个特征：开盘价、最高价、最低价、收盘价
model_feature_cols = ["open", "high", "low", "close"]

# ============================================================
# 第五部分：Backtrader回测框架配置
# ============================================================

# 自定义数据源类：扩展PandasData以支持预测信号列
# Backtrader默认的PandasData不包含我们需要的预测价格列
# 这里通过继承并添加pred字段，让策略可以访问预测价格
class PredictData(bt.feeds.PandasData):
    """自定义数据源：在标准OHLC数据基础上添加预测价格列"""
    lines = ("pred",)           # 定义新的数据列：预测价格
    params = (("pred", "pred"),) # 将DataFrame中的"pred"列映射到lines.pred


class PredictionSignalStrategy(bt.Strategy):
    """
    基于LSTM预测信号的量化交易策略
    
    策略逻辑：
    1. 每个交易日比较预测价格和当前价格，计算预期收益率
    2. 如果预期收益率 > 买入阈值，且当前无持仓，则买入
    3. 如果预期收益率 < -卖出阈值，且当前有持仓，则卖出
    4. 无论预测如何，达到止盈/止损线时立即平仓
    
    这样的设计可以：
    - 在市场上涨时及时入场
    - 在市场下跌时及时离场
    - 通过止盈止损控制最大亏损和锁定利润
    """
    
    # 策略参数：通过params字典配置，方便调优
    params = (
        ("buy_threshold", BUY_THRESHOLD),     # 买入阈值
        ("sell_threshold", SELL_THRESHOLD),   # 卖出阈值
        ("take_profit", TAKE_PROFIT),         # 止盈比例
        ("stop_loss", STOP_LOSS),             # 止损比例
    )

    def __init__(self):
        """
        初始化策略状态
        - pending_order：跟踪待执行的订单，避免重复下单
        - order_attempts：订单尝试次数，用于调试日志控制
        """
        self.pending_order = None   # 存储待处理订单，防止同时下多个订单
        self.order_attempts = 0    # 计数器，用于限制调试信息输出

    def next(self):
        """
        每个交易日执行一次的核心逻辑
        
        执行流程：
        1. 检查是否有待处理订单，如果有则跳过（避免重复交易）
        2. 获取当前现金、当前价格、预测价格
        3. 验证数据有效性（价格必须为正数且有限）
        4. 如果有持仓，先检查是否触发止盈/止损
        5. 如果无持仓且满足买入条件，执行买入
        6. 如果有持仓且满足卖出条件，执行卖出
        """
        # 如果有待处理订单未完成，本次跳过
        if self.pending_order is not None:
            return

        # 获取当前账户现金和各项价格
        cash = self.broker.getcash()                      # 可用资金
        close_price = float(self.data.close[0])           # 当前收盘价
        pred_price = float(self.data.pred[0])             # LSTM预测价格
        
        # 数据有效性检查：价格必须为正且有限
        if (not np.isfinite(close_price)) or close_price <= 0 or (not np.isfinite(pred_price)):
            return

        # 计算预期收益率：(预测价格 - 当前价格) / 当前价格
        # 例如：当前100，预测105，则收益率为5%
        expected_return = pred_price / close_price - 1.0
        
        # 计算实际买入成本（考虑佣金和滑点）
        # 佣金费率 * 滑点比例 = 额外成本
        # 例如：100 * (1+0.001) * (1+0.0005) = 100.15
        effective_price = close_price * (1 + COMMISSION_RATE) * (1 + SLIP_PERC)
        if effective_price <= 0 or (not np.isfinite(effective_price)):
            return

        # ===== 止盈止损检查（优先级最高） =====
        if self.position:
            entry_price = float(self.position.price)  # 持仓入场价格
            if entry_price > 0 and np.isfinite(entry_price):
                # 计算当前盈亏比例：(当前价格 - 入场价格) / 入场价格
                pnl_ratio = close_price / entry_price - 1.0
                
                # 止盈条件：盈利达到设定比例
                if pnl_ratio >= self.params.take_profit:
                    size = self.position.size
                    if size > 0:
                        print(
                            f"[止盈卖出] entry={entry_price:.4f} close={close_price:.4f} "
                            f"pnl={pnl_ratio:.2%} size={size}"
                        )
                        self.pending_order = self.sell(size=size)
                        return
                
                # 止损条件：亏损达到设定比例
                if pnl_ratio <= -self.params.stop_loss:
                    size = self.position.size
                    if size > 0:
                        print(
                            f"[止损卖出] entry={entry_price:.4f} close={close_price:.4f} "
                            f"pnl={pnl_ratio:.2%} size={size}"
                        )
                        self.pending_order = self.sell(size=size)
                        return

        # ===== 买入信号：预测明显上涨且当前无持仓 =====
        # 条件：预期收益率 >= 买入阈值（默认1%）
        if (not self.position) and (expected_return >= self.params.buy_threshold):
            # 计算可买入的股数：用现金除以实际成本价
            size = int(cash / effective_price)
            if size <= 0:
                return
            print(
                f"[买入信号] close={close_price:.4f} pred={pred_price:.4f} "
                f"ret={expected_return:.2%} size={size}"
            )
            self.pending_order = self.buy(size=size)
            return

        # ===== 卖出信号：预测转弱且已有持仓 =====
        # 条件：预期收益率 <= -卖出阈值（默认-0.5%）
        if self.position and (expected_return <= -self.params.sell_threshold):
            size = self.position.size
            if size > 0:
                print(
                    f"[卖出信号] close={close_price:.4f} pred={pred_price:.4f} "
                    f"ret={expected_return:.2%} size={size}"
                )
                self.pending_order = self.sell(size=size)

    def notify_order(self, order):
        """
        订单状态通知回调
        
        Backtrader在订单状态变化时会自动调用此方法
        用于处理订单完成、取消、被拒等情况
        """
        # 订单提交或接受中，等待执行
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # 控制日志输出频率
        if self.order_attempts >= 3:
            return
        self.order_attempts += 1
        
        try:
            # 打印订单详细信息
            print(
                f"[订单状态] {order.Status[0] if hasattr(order, 'Status') else order.status} | "
                f"status={order.status} | size={order.created.size} | "
                f"exec_price={getattr(order.executed, 'price', None)} | "
                f"exec_value={getattr(order.executed, 'value', None)}"
            )
        except Exception:
            print(f"[订单状态] status={order.status}")

        # 当订单最终确定时（完成/部分成交/取消/拒绝/保证金不足/过期），清除待处理订单标记
        if order.status in [
            order.Completed,
            getattr(order, "Partial", None),
            order.Canceled,
            order.Rejected,
            order.Margin,
            getattr(order, "Expired", None),
        ]:
            self.pending_order = None

# ============================================================
# 第六部分：实验运行函数
# ============================================================
def run_experiment(window):
    """
    运行单次实验：使用指定窗口训练LSTM模型并进行回测
    
    参数:
        window: 时间窗口大小，即用多少天的历史数据预测下一天
    
    返回:
        包含实验结果的字典：
        - window: 使用的窗口大小
        - final: 回测结束时的资金
        - annual: 年化收益率
        - maxdd: 最大回撤
        - df_pred: 包含预测结果的DataFrame
    
    实验流程：
    1. 划分训练集（70%）和测试集（30%）
    2. 对训练集拟合归一化器（防止数据泄漏）
    3. 准备LSTM输入数据（滑动窗口格式）
    4. 构建并训练LSTM模型
    5. 在测试集上进行预测
    6. 使用Backtrader进行回测
    7. 计算收益率和风险指标
    """
    # 数据量检查：必须有足够的数据来创建至少一个样本
    # 公式：总数据量 > window（用于一个样本）
    if len(df) <= window + 1:
        raise ValueError(
            f"window={window} 时数据不足：当前仅 {len(df)} 条，至少需要大于 {window + 1} 条。"
        )
    
    # 划分训练集和测试集：70%用于训练，30%用于测试
    split_point = int(len(df) * 0.7)
    # 确保split_point在合理范围内
    split_point = max(1, min(split_point, len(df) - 1))

    # ============================================================
    # 归一化处理
    # ============================================================
    # 归一化原因：神经网络对输入数据的scale敏感
    # 方法：将数值缩放到[0,1]区间，加速收敛
    # 重要：只用训练集fit，测试时transform，防止数据泄漏
    feature_scaler = MinMaxScaler()  # 特征归一化器
    target_scaler = MinMaxScaler()    # 目标值归一化器
    
    # 用训练集拟合归一化器（只使用70%数据）
    train_features = df[model_feature_cols].iloc[:split_point].values
    train_close = df["close"].iloc[:split_point].values.reshape(-1, 1)
    
    # 检查训练数据是否有效
    if train_features.shape[0] == 0 or train_close.shape[0] == 0:
        raise ValueError(
            f"window={window} 时训练集为空：split_point={split_point}, 数据总数={len(df)}。"
        )
    
    # 拟合归一化器：学习数据的min和max
    feature_scaler.fit(train_features)
    target_scaler.fit(train_close)

    # 对所有数据进行归一化（用于后续预测）
    all_features_scaled = feature_scaler.transform(df[model_feature_cols].values)
    all_close_scaled = target_scaler.transform(df["close"].values.reshape(-1, 1))

    # ============================================================
    # 构建LSTM训练数据（滑动窗口）
    # ============================================================
    # LSTM输入格式：[样本数, 时间步, 特征数]
    # 例如：window=10, 特征=4，则input_shape=(10, 4)
    X, y, times = [], [], []
    
    # 滑动窗口遍历：从第window天开始，每天用一个窗口的数据预测下一天
    for i in range(window, len(df)):
        # 提取过去window天的特征数据
        X.append(all_features_scaled[i - window : i, :])
        # 目标：当天收盘价的归一化值
        y.append(all_close_scaled[i, 0])
        # 记录对应的日期
        times.append(df.index[i])
    
    X = np.array(X)
    y = np.array(y)
    times = np.array(times)

    # 根据时间划分训练集和测试集
    train_end_time = df.index[split_point]
    train_mask = times < train_end_time   # 时间在分割点之前的数据用于训练
    test_mask = times >= train_end_time   # 时间在分割点之后的数据用于测试
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]
    test_times = times[test_mask]
    
    # 检查划分结果
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"window={window} 时训练或测试窗口样本为空，请减小 window。"
        )

    # ============================================================
    # 构建LSTM神经网络模型
    # ============================================================
    """
    模型结构：
    1. LSTM层（64个单元）：return_sequences=True，因为后面还有LSTM层
       - 64个神经元，能够捕捉时序数据的长期依赖
       - Dropout(0.1)防止过拟合，随机丢弃10%的神经元
    2. LSTM层（32个单元）：最后一层LSTM，不需要return_sequences
    3. Dense层（1个单元）：输出预测的收盘价
    """
    model = Sequential([
        # 第一层LSTM：输入形状为(窗口大小, 特征数量)
        LSTM(64, return_sequences=True, input_shape=(window, len(model_feature_cols))),
        Dropout(0.1),  # 随机丢弃10%的神经元，防止过拟合
        # 第二层LSTM：输出一个序列
        LSTM(32),
        # 全连接层：输出单个预测值
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")  # Adam优化器，MSE均方误差损失
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # ============================================================
    # 预测并反归一化
    # ============================================================
    pred_scaled = model.predict(X_test, verbose=0)  # 预测归一化后的值
    pred = target_scaler.inverse_transform(pred_scaled).reshape(-1)  # 反归一化得到真实价格

    # 将预测结果合并到原始数据
    df_pred = df.loc[test_times].copy()
    df_pred["pred"] = pred

    # ============================================================
    # Backtrader回测配置
    # ============================================================
    feed = PredictData(
        dataname=df_pred,
        open="open",
        high="high",
        low="low",
        close="close",
        pred="pred",
    )
    cerebro = bt.Cerebro()  # 创建Cerebro引擎（回测的核心控制器）
    cerebro.addstrategy(PredictionSignalStrategy)  # 添加策略
    cerebro.adddata(feed)   # 添加数据源
    cerebro.broker.setcash(100000.0)  # 设置初始资金10万

    # 配置交易成本：佣金
    cerc = getattr(bt, "CommInfoBase", None)
    commtype_perc = None
    if cerc is not None:
        commtype_perc = getattr(cerc, "COMM_PERC", None)
    if commtype_perc is not None:
        cerebro.broker.setcommission(commission=COMMISSION_RATE, commtype=commtype_perc)
    else:
        cerebro.broker.setcommission(commission=COMMISSION_RATE)

    # 配置交易成本：滑点
    set_slip_fn = getattr(cerebro.broker, "set_slippage_perc", None)
    if callable(set_slip_fn):
        try:
            set_slip_fn(SLIP_PERC)
        except Exception:
            pass

    # 添加分析器：计算收益率和最大回撤
    cerebro.addanalyzer(bt.analyzers.Returns, _name="ret")  # 收益率分析
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")   # 回撤分析

    # 运行回测
    results = cerebro.run()
    final = cerebro.broker.getvalue()  # 获取最终资金

    # 提取分析结果
    analysis_ret = results[0].analyzers.ret.get_analysis()
    annual_raw = analysis_ret.get("rnorm100", np.nan)  # 年化收益率
    maxdd = results[0].analyzers.dd.get_analysis()["max"]["drawdown"]  # 最大回撤

    # ============================================================
    # 计算备用年化收益率
    # ============================================================
    # 由于Backtrader的年化收益率计算可能有特殊情况，使用备用公式
    initial_cash = 100000.0
    years = (df_pred.index[-1] - df_pred.index[0]).days / 365.25
    annual_fallback = np.nan
    if (years > 0) and np.isfinite(final) and (initial_cash > 0) and (final > 0):
        # 年化收益率 = (最终资金 / 初始资金) ^ (1/年数) - 1
        annual_fallback = (final / initial_cash) ** (1 / years) - 1

    # 处理年化收益率的NaN情况
    annual_raw_is_nan = annual_raw is None
    if not annual_raw_is_nan:
        try:
            annual_raw_is_nan = bool(np.isnan(annual_raw))
        except Exception:
            annual_raw_is_nan = False
    annual = annual_fallback if annual_raw_is_nan else annual_raw

    # 处理最大回撤的NaN情况
    maxdd_is_nan = maxdd is None
    if not maxdd_is_nan:
        try:
            maxdd_is_nan = bool(np.isnan(maxdd))
        except Exception:
            maxdd_is_nan = False
    if maxdd_is_nan:
        maxdd = 0.0

    # 返回实验结果
    return {
        "window": window,
        "final": float(final),
        "annual": float(annual) if np.isfinite(annual) else np.nan,
        "maxdd": float(maxdd),
        "df_pred": df_pred,
    }


# ============================================================
# 第七部分：主程序执行
# ============================================================

# 运行所有窗口大小的实验
all_results = []
for w in WINDOW_LIST:
    print(f"\n===== 开始实验 window={w} =====")
    result = run_experiment(w)
    all_results.append(result)
    print(
        f"window={w} | 最终资金={result['final']:.2f} | "
        f"年化收益率={result['annual']:.2%} | 最大回撤={result['maxdd']:.2f}%"
    )

# 汇总结果：创建结果DataFrame并按年化收益率降序排列
result_df = pd.DataFrame(
    [{"window": r["window"], "final": r["final"], "annual": r["annual"], "maxdd": r["maxdd"]} for r in all_results]
).sort_values(by="annual", ascending=False)

print("\n" + "=" * 60)
print("window 对比结果（按年化收益率降序）：")
print(result_df.to_string(index=False))
print("=" * 60)

# 找出最优窗口（年化收益率最高）
best_result = all_results[int(result_df.index[0])]
best_window = best_result["window"]
print(f"\n✅ 最优窗口（按年化）：window={best_window}")

# ============================================================
# 第八部分：可视化输出
# ============================================================

# 使用最优 window 的预测结果绘制对比图
df_pred_best = best_result["df_pred"]

# 创建图表：真实价格 vs 预测价格
plt.figure(figsize=(10, 4))  # 设置图表大小：宽10英寸，高4英寸
plt.plot(df_pred_best.index, df_pred_best["close"], label="真实价")  # 蓝色实线：真实价格
plt.plot(df_pred_best.index, df_pred_best["pred"], label="预测价", c="r")  # 红色线：预测价格
plt.title(f"LSTM 预测对比（best window={best_window}）")  # 图表标题
plt.legend()  # 显示图例
plt.savefig("predict.png")  # 保存为PNG图片
plt.show()  # 显示图表