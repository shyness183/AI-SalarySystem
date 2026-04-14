#!/usr/bin/env python3
# models/train_XGBoost.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib

# ------------------------- 配置 -------------------------
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 已经划分好，这里只是参考

# ------------------------- 1. 加载数据 -------------------------
print("加载训练/测试数据...")
X_train = pd.read_parquet(f"{PROCESSED_DIR}/X_train_w2v.parquet")
X_test = pd.read_parquet(f"{PROCESSED_DIR}/X_test_w2v.parquet")
y_train = pd.read_parquet(f"{PROCESSED_DIR}/y_train.parquet").squeeze()
y_test = pd.read_parquet(f"{PROCESSED_DIR}/y_test.parquet").squeeze()

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ------------------------- 2. 训练基础 XGBoost 模型 -------------------------
print("\n训练 XGBoost 回归模型...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ------------------------- 3. 预测与评估 -------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== 模型评估结果 (默认参数) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ------------------------- 4. 特征重要性分析 -------------------------
# XGBoost 提供了三种重要性类型：'weight', 'gain', 'cover'
importance_type = 'gain'  # 常用 'gain' 表示特征带来的平均增益
importance = model.get_booster().get_score(importance_type=importance_type)

# 将重要性转换为 DataFrame 并排序
importance_df = pd.DataFrame({
    'feature': list(importance.keys()),
    'importance': list(importance.values())
}).sort_values(by='importance', ascending=False)

print("\nTop 10 特征重要性 (gain):")
print(importance_df.head(10))

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.barh(importance_df.head(20)['feature'], importance_df.head(20)['importance'])
plt.xlabel('Importance (gain)')
plt.title('XGBoost Feature Importance (Top 20)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/feature_importance.png", dpi=150)
print(f"特征重要性图已保存至 {MODEL_DIR}/feature_importance.png")

# ------------------------- 5. 保存模型 -------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = f"{MODEL_DIR}/xgboost_base.pkl"
joblib.dump(model, model_path)
print(f"模型已保存至 {model_path}")