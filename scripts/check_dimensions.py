#!/usr/bin/env python3
# scripts/check_dimensions.py

import pandas as pd

def check_dimensions():
    print("=" * 60)
    print("检查最终特征矩阵维度")
    print("=" * 60)

    # 加载数据
    X_train = pd.read_parquet('data/processed/X_train_w2v.parquet')
    X_test = pd.read_parquet('data/processed/X_test_w2v.parquet')

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")

    # 统计词向量特征列（以 skill_emb_ 开头）
    emb_cols = [col for col in X_train.columns if col.startswith('skill_emb_')]
    base_cols = [col for col in X_train.columns if not col.startswith('skill_emb_')]

    print(f"\n基础特征数量: {len(base_cols)}")
    print(f"词向量特征数量: {len(emb_cols)}")
    print(f"总特征数量: {len(base_cols) + len(emb_cols)}")

    # 列出所有基础特征列
    print("\n所有基础特征列:")
    for i, col in enumerate(base_cols):
        print(f"{i+1:3d}. {col}")

    # 列出所有词向量特征列
    print("\n所有词向量特征列:")
    for i, col in enumerate(emb_cols):
        print(f"{i+1:3d}. {col}")

    # 验证训练集和测试集列名是否一致
    if set(X_train.columns) == set(X_test.columns):
        print("\n✓ 训练集和测试集列名完全一致")
    else:
        print("\n✗ 训练集和测试集列名不一致！")
        train_only = set(X_train.columns) - set(X_test.columns)
        test_only = set(X_test.columns) - set(X_train.columns)
        if train_only:
            print(f"训练集独有列: {train_only}")
        if test_only:
            print(f"测试集独有列: {test_only}")

    # 验证维度是否与预期一致（可选，根据实际修改）
    expected_base = 53   # 基础特征数量（根据你的特征工程脚本确定）
    expected_emb = 100   # 词向量维度
    if len(base_cols) == expected_base and len(emb_cols) == expected_emb:
        print(f"\n✓ 特征维度符合预期：{expected_base} + {expected_emb} = {expected_base + expected_emb}")
    else:
        print(f"\n✗ 特征维度与预期不符！预期基础特征 {expected_base}，实际 {len(base_cols)}；预期词向量 {expected_emb}，实际 {len(emb_cols)}。")

if __name__ == "__main__":
    check_dimensions()