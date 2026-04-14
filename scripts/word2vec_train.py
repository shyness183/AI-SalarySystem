#!/usr/bin/env python3
# scripts/word2vec_train.py

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
import ast
from tqdm import tqdm

# ------------------------- model的参数配置 -------------------------
PROCESSED_DIR = "data/processed" # 词向量维度，可根据效果调整
WORD2VEC_DIM = 100   # 出现次数少于该值的技能将被忽略
MIN_WORD_COUNT = 5 # Word2Vec 上下文窗口大小，通常为 5-10
WINDOW = 5
SEED = 42  #锁死随机数，确保答辩时候，结果可复现

# ------------------------- 1. 加载数据 -------------------------
print("加载 df_encoded 完整数据...")
df_encoded = pd.read_parquet(f"{PROCESSED_DIR}/df_encoded.parquet")
print(f"df_encoded 形状: {df_encoded.shape}")

print("加载训练集和测试集（用于获取索引）...")
X_train = pd.read_parquet(f"{PROCESSED_DIR}/X_train.parquet")
X_test = pd.read_parquet(f"{PROCESSED_DIR}/X_test.parquet")
print(f"X_train 形状: {X_train.shape}, X_test 形状: {X_test.shape}")

# 确保技能列表列是列表格式
def safe_to_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    elif isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return []

df_encoded['skills_list'] = df_encoded['skills_list'].apply(safe_to_list)

# ------------------------- 2. 根据索引提取技能列表 -------------------------
print("根据索引提取训练集和测试集的技能列表...")
train_indices = X_train.index
test_indices = X_test.index

train_skills_raw = df_encoded.loc[train_indices, 'skills_list'].tolist()
test_skills_raw = df_encoded.loc[test_indices, 'skills_list'].tolist()

# 清洗：确保每个技能字符串去除空格，并过滤空字符串
train_skills = [[skill.strip() for skill in skills if skill.strip()] for skills in train_skills_raw]
test_skills = [[skill.strip() for skill in skills if skill.strip()] for skills in test_skills_raw]

# 检查数据情况
non_empty_train = sum(1 for skills in train_skills if len(skills) > 0)
total_skills_train = sum(len(skills) for skills in train_skills)
print(f"训练集样本数: {len(train_skills)}")
print(f"非空技能列表样本数: {non_empty_train}")
print(f"总技能出现次数: {total_skills_train}")

if total_skills_train == 0:
    raise ValueError("训练集中没有任何技能数据，请检查原始数据或特征工程步骤。")

# ------------------------- 3. 训练 Word2Vec 模型 -------------------------
print("初始化 Word2Vec 模型...")
w2v_model = Word2Vec(
    vector_size=WORD2VEC_DIM,
    window=WINDOW,
    min_count=MIN_WORD_COUNT,
    workers=4,
    seed=SEED,
    epochs=10
)

print("构建词汇表...")
w2v_model.build_vocab(train_skills)
print(f"词汇表大小: {len(w2v_model.wv)}")

print("训练 Word2Vec 模型...")
w2v_model.train(train_skills, total_examples=len(train_skills), epochs=w2v_model.epochs)

# 保存模型
w2v_model.save(f"{PROCESSED_DIR}/skills_word2vec.model")
print(f"模型已保存至 {PROCESSED_DIR}/skills_word2vec.model")

word_vectors = w2v_model.wv

# ------------------------- 4. 生成技能平均向量函数 -------------------------
def get_skill_avg_vector(skills_list, word_vectors, dim=WORD2VEC_DIM):
    vectors = []
    for skill in skills_list:
        if skill in word_vectors:
            vectors.append(word_vectors[skill])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# ------------------------- 5. 为训练集和测试集生成特征 -------------------------
print("为训练集生成词向量特征...")
train_embeddings = np.array([get_skill_avg_vector(skills, word_vectors) for skills in tqdm(train_skills)])
print("为测试集生成词向量特征...")
test_embeddings = np.array([get_skill_avg_vector(skills, word_vectors) for skills in tqdm(test_skills)])

emb_cols = [f'skill_emb_{i}' for i in range(WORD2VEC_DIM)]
train_emb_df = pd.DataFrame(train_embeddings, columns=emb_cols, index=train_indices)
test_emb_df = pd.DataFrame(test_embeddings, columns=emb_cols, index=test_indices)

# ------------------------- 6. 拼接到原特征矩阵 -------------------------
# 删除原来的 skills_list 列（如果存在）
if 'skills_list' in X_train.columns:
    X_train = X_train.drop(columns=['skills_list'])
if 'skills_list' in X_test.columns:
    X_test = X_test.drop(columns=['skills_list'])

X_train_w2v = pd.concat([X_train, train_emb_df], axis=1)
X_test_w2v = pd.concat([X_test, test_emb_df], axis=1)

print(f"新训练集形状: {X_train_w2v.shape}")
print(f"新测试集形状: {X_test_w2v.shape}")

X_train_w2v.to_parquet(f"{PROCESSED_DIR}/X_train_w2v.parquet")
X_test_w2v.to_parquet(f"{PROCESSED_DIR}/X_test_w2v.parquet")
print("新特征文件已保存。")