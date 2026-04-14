import pandas as pd

X_train = pd.read_parquet('data/processed/X_train_w2v.parquet')
X_test = pd.read_parquet('data/processed/X_test_w2v.parquet')
y_train = pd.read_parquet('data/processed/y_train.parquet').squeeze()
y_test = pd.read_parquet('data/processed/y_test.parquet').squeeze()

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("X_train columns:", X_train.columns.tolist()[:10])  # 只显示前10列
print("First few y values:", y_train.head())