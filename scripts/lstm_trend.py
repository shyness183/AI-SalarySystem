#!/usr/bin/env python3
"""
lstm_trend.py
输入：data/raw/ai_job_dataset.csv + ai_job_dataset1.csv
功能：按月聚合全球AI薪资均值，用LSTM预测未来薪资趋势
输出：
  - models/lstm_trend.keras        训练好的模型
  - models/lstm_scaler.pkl         归一化器（推理时用）
  - data/processed/monthly_avg.csv 月均薪资序列（供ECharts使用）
  - data/processed/lstm_forecast.csv 预测结果（供ECharts使用）
"""

import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')               # 无头模式，服务器/Docker环境不报错
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── 配置 ──────────────────────────────────────────────────────────────────
RAW_DIR        = "data/raw"
PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "models"
WINDOW_SIZE    = 3     # 用前3个月预测下1个月（数据只有16个点，窗口不宜过大）
FORECAST_STEPS = 3     # 向后预测3个月
EPOCHS         = 200
BATCH_SIZE     = 4
RANDOM_SEED    = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 1. 读取并聚合数据 ─────────────────────────────────────────────────────
print("=" * 55)
print("步骤1：加载数据并按月聚合薪资均值")
print("=" * 55)

df1 = pd.read_csv(f"{RAW_DIR}/ai_job_dataset.csv",  parse_dates=["posting_date"])
df2 = pd.read_csv(f"{RAW_DIR}/ai_job_dataset1.csv", parse_dates=["posting_date"])
df  = pd.concat([df1[["posting_date", "salary_usd"]],
                 df2[["posting_date", "salary_usd"]]], ignore_index=True)

df["month"] = df["posting_date"].dt.to_period("M")
monthly = (df.groupby("month")["salary_usd"]
             .agg(mean_salary="mean", count="count")
             .reset_index())
monthly["month_str"] = monthly["month"].astype(str)   # "2024-01" 格式
monthly = monthly.sort_values("month_str").reset_index(drop=True)

print(f"月度时间点数量: {len(monthly)}")
print(monthly[["month_str", "mean_salary", "count"]].to_string(index=False))

monthly.to_csv(f"{PROCESSED_DIR}/monthly_avg.csv", index=False)
print(f"\n月均薪资已保存至 {PROCESSED_DIR}/monthly_avg.csv")

# ── 2. 归一化 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤2：MinMax 归一化")
print("=" * 55)

values = monthly["mean_salary"].values.reshape(-1, 1).astype(float)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

joblib.dump(scaler, f"{MODEL_DIR}/lstm_scaler.pkl")
print(f"归一化器已保存至 {MODEL_DIR}/lstm_scaler.pkl")
print(f"原始薪资范围: [{values.min():.0f}, {values.max():.0f}] USD")

# ── 3. 构建滑动窗口样本 ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"步骤3：滑动窗口构建（window={WINDOW_SIZE}）")
print("=" * 55)

def make_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window, 0])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, WINDOW_SIZE)
print(f"样本数量: {len(X)}  （{len(scaled)}个时间点 - {WINDOW_SIZE}窗口）")

# 时序数据按前80%训练、后20%测试（不能随机打乱）
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"训练集: {len(X_train)} 个样本 | 测试集: {len(X_test)} 个样本")

# LSTM输入格式: (samples, timesteps, features)
X_train = X_train.reshape(-1, WINDOW_SIZE, 1)
X_test  = X_test.reshape(-1, WINDOW_SIZE, 1)

# ── 4. 构建 LSTM 模型 ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤4：构建 LSTM 模型")
print("=" * 55)

model = Sequential([
    LSTM(64, input_shape=(WINDOW_SIZE, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
], name="salary_trend_lstm")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",
              metrics=["mae"])
model.summary()

# ── 5. 训练 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤5：训练模型")
print("=" * 55)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

model.save(f"{MODEL_DIR}/lstm_trend.keras")
print(f"\n模型已保存至 {MODEL_DIR}/lstm_trend.keras")

# ── 6. 测试集评估 ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤6：测试集评估")
print("=" * 55)

y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"RMSE : {rmse:,.0f} USD")
print(f"MAE  : {mae:,.0f} USD")
print(f"MAPE : {mape:.2f}%")

# ── 7. 向未来递推预测 FORECAST_STEPS 个月 ────────────────────────────────
print("\n" + "=" * 55)
print(f"步骤7：递推预测未来 {FORECAST_STEPS} 个月")
print("=" * 55)

# 用最后 WINDOW_SIZE 个真实值作为起始窗口
last_window = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1).copy()
future_preds_scaled = []

for step in range(FORECAST_STEPS):
    pred = model.predict(last_window, verbose=0)[0, 0]
    future_preds_scaled.append(pred)
    # 滚动窗口：丢掉最旧一个，追加新预测值
    last_window = np.roll(last_window, -1, axis=1)
    last_window[0, -1, 0] = pred

future_preds = scaler.inverse_transform(
    np.array(future_preds_scaled).reshape(-1, 1)
).flatten()

# 推算未来月份标签
last_period = monthly["month"].iloc[-1]
future_months = []
for i in range(1, FORECAST_STEPS + 1):
    future_months.append(str(last_period + i))

print("未来预测结果：")
for m, v in zip(future_months, future_preds):
    print(f"  {m}: {v:,.0f} USD")

# ── 8. 保存完整预测 CSV ───────────────────────────────────────────────────
# 历史真实值
hist_df = monthly[["month_str", "mean_salary"]].copy()
hist_df.columns = ["month", "actual"]
hist_df["predicted"] = None
hist_df["type"] = "historical"

# 测试集对应的预测值（回填，便于可视化对比）
test_start_idx = split + WINDOW_SIZE        # 测试集预测对应的原始月份索引
for i, pred_val in enumerate(y_pred):
    idx = test_start_idx + i
    if idx < len(hist_df):
        hist_df.loc[idx, "predicted"] = pred_val
        hist_df.loc[idx, "type"] = "test_pred"

# 未来预测
future_df = pd.DataFrame({
    "month":     future_months,
    "actual":    [None] * FORECAST_STEPS,
    "predicted": future_preds.tolist(),
    "type":      ["forecast"] * FORECAST_STEPS
})

result_df = pd.concat([hist_df, future_df], ignore_index=True)
result_df.to_csv(f"{PROCESSED_DIR}/lstm_forecast.csv", index=False)
print(f"\n预测结果已保存至 {PROCESSED_DIR}/lstm_forecast.csv")

# ── 9. 可视化 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("步骤9：生成可视化图表")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Global AI Salary Trend — LSTM Forecast", fontsize=14, fontweight="bold")

# 图1：训练 Loss 曲线
ax1 = axes[0]
ax1.plot(history.history["loss"],     label="Train Loss", color="#4472C4", linewidth=1.5)
ax1.plot(history.history["val_loss"], label="Val Loss",   color="#ED7D31", linewidth=1.5)
ax1.set_title("Training & Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：预测结果对比
ax2 = axes[1]
all_months   = result_df["month"].tolist()
actual_vals  = result_df["actual"].astype(float)
pred_vals    = result_df["predicted"].astype(float)

# 历史真实曲线
hist_mask = result_df["type"].isin(["historical", "test_pred"])
ax2.plot(result_df[hist_mask]["month"].tolist(),
         actual_vals[hist_mask].tolist(),
         color="#4472C4", linewidth=2, label="Actual", marker="o", markersize=4)

# 测试集预测点
test_mask = result_df["type"] == "test_pred"
if test_mask.any():
    ax2.scatter(result_df[test_mask]["month"].tolist(),
                pred_vals[test_mask].tolist(),
                color="#70AD47", zorder=5, s=60, label="Test Pred", marker="^")

# 未来预测曲线
fore_mask = result_df["type"] == "forecast"
forecast_x = ([result_df[hist_mask]["month"].iloc[-1]] +
               result_df[fore_mask]["month"].tolist())
forecast_y = ([actual_vals[hist_mask].iloc[-1]] +
               pred_vals[fore_mask].tolist())
ax2.plot(forecast_x, forecast_y,
         color="#ED7D31", linewidth=2, linestyle="--", marker="s", markersize=5,
         label=f"Forecast (+{FORECAST_STEPS}m)")

ax2.set_title("Monthly Avg Salary: Actual vs Forecast")
ax2.set_xlabel("Month")
ax2.set_ylabel("Avg Salary (USD)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

chart_path = f"{MODEL_DIR}/lstm_forecast_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"可视化图表已保存至 {chart_path}")

# ── 10. 输出供 ECharts / Flask 使用的 JSON ────────────────────────────────
echarts_data = {
    "months":    result_df["month"].tolist(),
    "actual":    [round(v, 0) if pd.notna(v) else None for v in actual_vals],
    "predicted": [round(v, 0) if pd.notna(v) else None for v in pred_vals],
    "type":      result_df["type"].tolist(),
    "metrics": {
        "rmse": round(float(rmse), 2),
        "mae":  round(float(mae),  2),
        "mape": round(float(mape), 2)
    }
}

json_path = f"{PROCESSED_DIR}/lstm_echarts.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(echarts_data, f, ensure_ascii=False, indent=2)
print(f"ECharts JSON 已保存至 {json_path}")

print("\n" + "=" * 55)
print("LSTM 训练完成！")
print(f"  模型文件  : {MODEL_DIR}/lstm_trend.keras")
print(f"  归一化器  : {MODEL_DIR}/lstm_scaler.pkl")
print(f"  月均CSV   : {PROCESSED_DIR}/monthly_avg.csv")
print(f"  预测CSV   : {PROCESSED_DIR}/lstm_forecast.csv")
print(f"  ECharts   : {PROCESSED_DIR}/lstm_echarts.json")
print(f"  图表      : {MODEL_DIR}/lstm_forecast_chart.png")
print("=" * 55)
