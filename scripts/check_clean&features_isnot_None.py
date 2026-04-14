#!/usr/bin/env python3
# scripts/check_data.py

import pandas as pd
import ast

# 1. 检查清洗后的数据
print("=" * 60)
print("检查清洗后的数据: data/interim/cleaned_jobs.parquet")
df_cleaned = pd.read_parquet("data/interim/cleaned_jobs.parquet")
print("清洗数据形状:", df_cleaned.shape)
print("清洗数据列:", df_cleaned.columns.tolist())

# 检查 skills_array 列
if 'skills_array' in df_cleaned.columns:
    print("\nskills_array 列的前5个值:")
    print(df_cleaned['skills_array'].head(10).tolist())
    # 统计空值
    null_count = df_cleaned['skills_array'].isnull().sum()
    print(f"skills_array 空值数量: {null_count}")
    # 统计空列表（如果是字符串或列表）
    if null_count == 0:
        # 尝试检查是否是列表类型
        sample = df_cleaned['skills_array'].iloc[0]
        if isinstance(sample, list):
            empty_list_count = df_cleaned['skills_array'].apply(lambda x: len(x) == 0).sum()
            print(f"skills_array 为空列表的数量: {empty_list_count}")
        elif isinstance(sample, str):
            # 可能是字符串表示的列表
            empty_list_count = df_cleaned['skills_array'].apply(lambda x: x == '[]' or x == '').sum()
            print(f"skills_array 为 '[]' 或空字符串的数量: {empty_list_count}")
else:
    print("skills_array 列不存在！")

# 2. 检查特征工程后的完整编码数据
print("\n" + "=" * 60)
print("检查特征工程后的完整编码数据: data/processed/df_encoded.parquet")
df_encoded = pd.read_parquet("data/processed/df_encoded.parquet")
print("df_encoded 形状:", df_encoded.shape)
print("df_encoded 列:", df_encoded.columns.tolist())

# 检查 skills_list 列
if 'skills_list' in df_encoded.columns:
    print("\nskills_list 列的前5个值:")
    print(df_encoded['skills_list'].head(10).tolist())
    null_count = df_encoded['skills_list'].isnull().sum()
    print(f"skills_list 空值数量: {null_count}")
    # 检查空列表
    empty_list_count = df_encoded['skills_list'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
    print(f"skills_list 为空列表的数量: {empty_list_count}")
else:
    print("skills_list 列不存在！")

# 3. 检查训练集 X_train.parquet
print("\n" + "=" * 60)
print("检查训练集 X_train.parquet")
X_train = pd.read_parquet("data/processed/X_train.parquet")
print("X_train 形状:", X_train.shape)
print("X_train 列:", X_train.columns.tolist())

if 'skills_list' in X_train.columns:
    print("\nX_train 中 skills_list 列的前5个值:")
    print(X_train['skills_list'].head(10).tolist())
    null_count = X_train['skills_list'].isnull().sum()
    print(f"skills_list 空值数量: {null_count}")
    empty_list_count = X_train['skills_list'].apply(lambda x: len(x) == 0 if isinstance(x, list) else True).sum()
    print(f"skills_list 为空列表的数量: {empty_list_count}")
else:
    print("X_train 中无 skills_list 列！")