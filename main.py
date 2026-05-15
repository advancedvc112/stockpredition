import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import backtrader as bt
import re

# Matplotlib 中文显示配置（Windows 优先微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 成本参数：用于下单数量估算，避免因手续费/滑点导致订单被拒绝
COMMISSION_RATE = 0.001   # 0.1%
SLIP_PERC = 0.0005        # 0.05%

# -------------------------- 从 Excel 读取数据 --------------------------
file_path = r"D:\智能课程设计\数据\Table.xls"

# Excel表头：时间, 开盘, 最高, 最低, 收盘, 涨幅, 振幅, 总手, 金额, 换手%, 成交次数
# 阅读数据
try:
    df_raw = pd.read_excel(file_path)
except Exception:
    df_raw = pd.read_csv(file_path, sep="\t", encoding="gbk", engine="python")

# 清理列名空白，避免隐藏空格导致 KeyError
df_raw.columns = [str(c).strip() for c in df_raw.columns]

required_cols = ["时间", "开盘", "最高", "最低", "收盘"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    raise ValueError(
        f"缺少必要列: {missing}。当前列名为: {list(df_raw.columns)}"
    )

# 5. 提取需要的列
df = pd.DataFrame()
df["trade_date"] = df_raw["时间"]
df["open"] = df_raw["开盘"]
df["high"] = df_raw["最高"]
df["low"] = df_raw["最低"]
df["close"] = df_raw["收盘"]

# 6. 清洗日期（去掉星期）
df["trade_date"] = df["trade_date"].astype(str).str.split(",").str[0].str.strip()
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

# 7. 清洗价格（去掉千分位逗号）
for col in ["open", "high", "low", "close"]:
    df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 8. 最终清洗
df = df.dropna()
df = df[df["close"] > 0]
df = df.set_index("trade_date").sort_index()

if df.empty:
    raise ValueError(
        "清洗后没有可用数据（df 为空）。请检查原始文件分隔符、列名映射和日期/价格格式。"
    )

print("✅ 读取成功！")
print("数据条数：", len(df))
print("时间范围：", df.index.min(), "→", df.index.max())
# ======================================================================

WINDOW_LIST = [10, 15, 20, 30, 50] # 可调整窗口列表：自动对比不同 window 的效果
EPOCHS = 5
BATCH_SIZE = 16
BUY_THRESHOLD = 0.01    # 预测涨幅超过该阈值买入
SELL_THRESHOLD = 0.005  # 预测跌幅超过该阈值卖出
TAKE_PROFIT = 0.15       # 止盈
STOP_LOSS = 0.07        # 止损

# 模型输入特征：后续加特征时只需改这个列表，统一归一化会自动生效
model_feature_cols = ["open", "high", "low", "close"]

# 5. 使用预测信号交易（预测涨跌阈值触发买卖）
class PredictData(bt.feeds.PandasData):
    lines = ("pred",)
    params = (("pred", "pred"),)


class PredictionSignalStrategy(bt.Strategy):
    params = (
        ("buy_threshold", BUY_THRESHOLD),
        ("sell_threshold", SELL_THRESHOLD),
        ("take_profit", TAKE_PROFIT),
        ("stop_loss", STOP_LOSS),
    )

    def __init__(self):
        self.pending_order = None
        self.order_attempts = 0

    def next(self):
        if self.pending_order is not None:
            return

        cash = self.broker.getcash()
        close_price = float(self.data.close[0])
        pred_price = float(self.data.pred[0])
        if (not np.isfinite(close_price)) or close_price <= 0 or (not np.isfinite(pred_price)):
            return

        expected_return = pred_price / close_price - 1.0
        effective_price = close_price * (1 + COMMISSION_RATE) * (1 + SLIP_PERC)
        if effective_price <= 0 or (not np.isfinite(effective_price)):
            return

        # 已持仓时，优先执行止盈止损
        if self.position:
            entry_price = float(self.position.price)
            if entry_price > 0 and np.isfinite(entry_price):
                pnl_ratio = close_price / entry_price - 1.0
                if pnl_ratio >= self.params.take_profit:
                    size = self.position.size
                    if size > 0:
                        print(
                            f"[止盈卖出] entry={entry_price:.4f} close={close_price:.4f} "
                            f"pnl={pnl_ratio:.2%} size={size}"
                        )
                        self.pending_order = self.sell(size=size)
                        return
                if pnl_ratio <= -self.params.stop_loss:
                    size = self.position.size
                    if size > 0:
                        print(
                            f"[止损卖出] entry={entry_price:.4f} close={close_price:.4f} "
                            f"pnl={pnl_ratio:.2%} size={size}"
                        )
                        self.pending_order = self.sell(size=size)
                        return

        # 买入信号：预测明显上涨且当前无持仓
        if (not self.position) and (expected_return >= self.params.buy_threshold):
            size = int(cash / effective_price)
            if size <= 0:
                return
            print(
                f"[买入信号] close={close_price:.4f} pred={pred_price:.4f} "
                f"ret={expected_return:.2%} size={size}"
            )
            self.pending_order = self.buy(size=size)
            return

        # 卖出信号：预测转弱（明显下跌）且已有持仓
        if self.position and (expected_return <= -self.params.sell_threshold):
            size = self.position.size
            if size > 0:
                print(
                    f"[卖出信号] close={close_price:.4f} pred={pred_price:.4f} "
                    f"ret={expected_return:.2%} size={size}"
                )
                self.pending_order = self.sell(size=size)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if self.order_attempts >= 3:
            return
        self.order_attempts += 1
        try:
            print(
                f"[订单状态] {order.Status[0] if hasattr(order, 'Status') else order.status} | "
                f"status={order.status} | size={order.created.size} | "
                f"exec_price={getattr(order.executed, 'price', None)} | "
                f"exec_value={getattr(order.executed, 'value', None)}"
            )
        except Exception:
            print(f"[订单状态] status={order.status}")

        if order.status in [
            order.Completed,
            getattr(order, "Partial", None),
            order.Canceled,
            order.Rejected,
            order.Margin,
            getattr(order, "Expired", None),
        ]:
            self.pending_order = None

def run_experiment(window):
    if len(df) <= window + 1:
        raise ValueError(
            f"window={window} 时数据不足：当前仅 {len(df)} 条，至少需要大于 {window + 1} 条。"
        )
    split_point = int(len(df) * 0.7)
    split_point = max(1, min(split_point, len(df) - 1))

    # 仅用训练集拟合归一化器，避免未来数据泄漏
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    train_features = df[model_feature_cols].iloc[:split_point].values
    train_close = df["close"].iloc[:split_point].values.reshape(-1, 1)
    if train_features.shape[0] == 0 or train_close.shape[0] == 0:
        raise ValueError(
            f"window={window} 时训练集为空：split_point={split_point}, 数据总数={len(df)}。"
        )
    feature_scaler.fit(train_features)
    target_scaler.fit(train_close)

    # 对所有输入特征统一归一化
    all_features_scaled = feature_scaler.transform(df[model_feature_cols].values)
    all_close_scaled = target_scaler.transform(df["close"].values.reshape(-1, 1))

    X, y, times = [], [], []
    for i in range(window, len(df)):
        X.append(all_features_scaled[i - window : i, :])
        y.append(all_close_scaled[i, 0])
        times.append(df.index[i])
    X = np.array(X)
    y = np.array(y)
    times = np.array(times)

    train_end_time = df.index[split_point]
    train_mask = times < train_end_time
    test_mask = times >= train_end_time
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]
    test_times = times[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"window={window} 时训练或测试窗口样本为空，请减小 window。"
        )

    # LSTM 稍微加深：两层 + 轻量 Dropout，增强时序表达能力
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, len(model_feature_cols))),
        Dropout(0.1),
        LSTM(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    pred_scaled = model.predict(X_test, verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled).reshape(-1)

    df_pred = df.loc[test_times].copy()
    df_pred["pred"] = pred

    feed = PredictData(
        dataname=df_pred,
        open="open",
        high="high",
        low="low",
        close="close",
        pred="pred",
    )
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PredictionSignalStrategy)
    cerebro.adddata(feed)
    cerebro.broker.setcash(100000.0)

    cerc = getattr(bt, "CommInfoBase", None)
    commtype_perc = None
    if cerc is not None:
        commtype_perc = getattr(cerc, "COMM_PERC", None)
    if commtype_perc is not None:
        cerebro.broker.setcommission(commission=COMMISSION_RATE, commtype=commtype_perc)
    else:
        cerebro.broker.setcommission(commission=COMMISSION_RATE)

    set_slip_fn = getattr(cerebro.broker, "set_slippage_perc", None)
    if callable(set_slip_fn):
        try:
            set_slip_fn(SLIP_PERC)
        except Exception:
            pass

    cerebro.addanalyzer(bt.analyzers.Returns, _name="ret")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    results = cerebro.run()
    final = cerebro.broker.getvalue()
    analysis_ret = results[0].analyzers.ret.get_analysis()
    annual_raw = analysis_ret.get("rnorm100", np.nan)
    maxdd = results[0].analyzers.dd.get_analysis()["max"]["drawdown"]

    initial_cash = 100000.0
    years = (df_pred.index[-1] - df_pred.index[0]).days / 365.25
    annual_fallback = np.nan
    if (years > 0) and np.isfinite(final) and (initial_cash > 0) and (final > 0):
        annual_fallback = (final / initial_cash) ** (1 / years) - 1

    annual_raw_is_nan = annual_raw is None
    if not annual_raw_is_nan:
        try:
            annual_raw_is_nan = bool(np.isnan(annual_raw))
        except Exception:
            annual_raw_is_nan = False
    annual = annual_fallback if annual_raw_is_nan else annual_raw

    maxdd_is_nan = maxdd is None
    if not maxdd_is_nan:
        try:
            maxdd_is_nan = bool(np.isnan(maxdd))
        except Exception:
            maxdd_is_nan = False
    if maxdd_is_nan:
        maxdd = 0.0

    return {
        "window": window,
        "final": float(final),
        "annual": float(annual) if np.isfinite(annual) else np.nan,
        "maxdd": float(maxdd),
        "df_pred": df_pred,
    }


all_results = []
for w in WINDOW_LIST:
    print(f"\n===== 开始实验 window={w} =====")
    result = run_experiment(w)
    all_results.append(result)
    print(
        f"window={w} | 最终资金={result['final']:.2f} | "
        f"年化收益率={result['annual']:.2%} | 最大回撤={result['maxdd']:.2f}%"
    )

result_df = pd.DataFrame(
    [{"window": r["window"], "final": r["final"], "annual": r["annual"], "maxdd": r["maxdd"]} for r in all_results]
).sort_values(by="annual", ascending=False)

print("\n" + "=" * 60)
print("window 对比结果（按年化收益率降序）：")
print(result_df.to_string(index=False))
print("=" * 60)

best_result = all_results[int(result_df.index[0])]
best_window = best_result["window"]
print(f"\n✅ 最优窗口（按年化）：window={best_window}")

# 使用最优 window 的预测结果画图
df_pred_best = best_result["df_pred"]
plt.figure(figsize=(10, 4))
plt.plot(df_pred_best.index, df_pred_best["close"], label="真实价")
plt.plot(df_pred_best.index, df_pred_best["pred"], label="预测价", c="r")
plt.title(f"LSTM 预测对比（best window={best_window}）")
plt.legend()
plt.savefig("predict.png")
plt.show()