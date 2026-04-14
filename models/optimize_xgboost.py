#!/usr/bin/env python3
# models/optimize_xgboost.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from scipy.stats import uniform, randint

# ------------------------- 配置 -------------------------
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
RANDOM_STATE = 42
N_ITER = 50          # 随机搜索迭代次数
CV_FOLDS = 5

# ------------------------- 1. 加载数据 -------------------------
print("加载训练/测试数据...")
X_train = pd.read_parquet(f"{PROCESSED_DIR}/X_train_w2v.parquet")
X_test = pd.read_parquet(f"{PROCESSED_DIR}/X_test_w2v.parquet")
y_train = pd.read_parquet(f"{PROCESSED_DIR}/y_train.parquet").squeeze()
y_test = pd.read_parquet(f"{PROCESSED_DIR}/y_test.parquet").squeeze()

# ------------------------- 2. 定义参数分布 -------------------------
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# ------------------------- 3. 初始化模型和搜索 -------------------------
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=N_ITER,
    cv=kfold,
    scoring='neg_root_mean_squared_error',
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("开始随机搜索...")
random_search.fit(X_train, y_train)

print("\n最佳参数:")
print(random_search.best_params_)
print(f"最佳交叉验证 RMSE: {-random_search.best_score_:.4f}")

# ------------------------- 4. 在测试集上评估最佳模型 -------------------------
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== 优化后模型在测试集上的表现 ===")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# ------------------------- 5. 保存最佳模型 -------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = f"{MODEL_DIR}/xgboost_optimized.pkl"
joblib.dump(best_model, model_path)
print(f"优化模型已保存至 {model_path}")