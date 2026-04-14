#!/usr/bin/env python3
"""
特征工程脚本
输入：data/interim/cleaned_jobs.parquet（清洗后的数据）
输出：data/processed/ 下的训练/测试特征矩阵和目标变量
说明：本脚本完成基础特征工程，包括技能列表处理、有序编码、日期特征、独热编码等。
      技能词向量特征将在独立脚本（word2vec_features.py）中生成，拼接到本脚本输出的特征上。
      标准化和特征筛选为可选步骤，可根据需要开启。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

# 设置显示所有列
pd.set_option('display.max_columns', None)

# 1. 加载清洗后的数据
data_path = "data/interim/cleaned_jobs.parquet"
df = pd.read_parquet(data_path)

print("数据形状:", df.shape)
print("列名:", df.columns.tolist())

# 2. 处理技能列表：保留原列 skills_array，新建列 skills_list（Python列表格式）
# 确保转换为 Python 列表，避免 numpy 数组问题
def to_python_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, str):
        return ast.literal_eval(arr)
    elif isinstance(arr, list):
        return arr
    else:
        return []

df['skills_list'] = df['skills_array'].apply(to_python_list)
# 检查转换结果
print("技能列示例:\n", df['skills_list'].head())

# 3. 检查缺失值
print("\n缺失值统计:\n", df.isnull().sum())

# 4. 目标变量：薪资对数变换
df['log_salary'] = np.log(df['salary_usd'])
# 可视化验证（可选）
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].hist(df['salary_usd'], bins=50, edgecolor='k')
axes[0].set_title('Salary (original)')
axes[1].hist(df['log_salary'], bins=50, edgecolor='k')
axes[1].set_title('Log Salary')
plt.tight_layout()
plt.show()

# 5. 有序分类变量编码
print("\n有序变量唯一值:")
print("经验水平:\n", df['experience_level'].value_counts())
print("学历要求:\n", df['education_required'].value_counts())
print("公司规模:\n", df['company_size'].value_counts())

exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
df['exp_level_encoded'] = df['experience_level'].map(exp_map)

edu_map = {
    'Bachelor': 0,
    'Master': 1,
    'PhD': 2,
    'Associate': 3,
    'Unknown': 0
}
df['edu_level_encoded'] = df['education_required'].map(edu_map)

size_map = {'S': 0, 'M': 1, 'L': 2, 'unknown': 0}
df['size_encoded'] = df['company_size'].map(size_map)

# 6. 日期特征提取
df['posting_date'] = pd.to_datetime(df['posting_date'])
df['application_deadline'] = pd.to_datetime(df['application_deadline'])

df['post_year'] = df['posting_date'].dt.year
df['post_month'] = df['posting_date'].dt.month
df['post_quarter'] = df['posting_date'].dt.quarter
df['deadline_year'] = df['application_deadline'].dt.year
df['deadline_month'] = df['application_deadline'].dt.month
df['days_to_deadline'] = (df['application_deadline'] - df['posting_date']).dt.days.clip(lower=0)

print("\n招聘天数统计:\n", df['days_to_deadline'].describe())

# 7. 技能数量特征
df['skill_count'] = df['skills_list'].apply(len)
print("\n技能数量统计:\n", df['skill_count'].describe())

# 8. 分类变量基数检查
print("\n分类变量基数:")
print("行业唯一值数量:", df['industry'].nunique())
print("国家唯一值数量:", df['country'].nunique())
print("雇佣类型唯一值数量:", df['employment_type'].nunique())

# 9. 独热编码
df_encoded = pd.get_dummies(df, columns=['industry', 'employment_type', 'country'],
                             prefix=['ind', 'emp', 'ctry'], dummy_na=False)
print("\n编码后数据形状:", df_encoded.shape)

# 10. 丢弃无关列，准备特征矩阵 X 和目标 y
cols_to_drop = [
    'job_id', 'job_title', 'required_skills', 'skills_array',
    'posting_date', 'application_deadline',
    'experience_level', 'education_required', 'company_size',
    'employee_residence', 'salary_currency', 'company_name', 'company_location',
    'salary_usd', 'log_salary'   # log_salary 是目标，不放入特征
]
X = df_encoded.drop(columns=cols_to_drop, errors='ignore')
y = df_encoded['log_salary']

print("\n特征矩阵形状:", X.shape)
print("特征列（前20个）:\n", X.columns[:20].tolist())

# 11. 可选：数值特征标准化（此处跳过）

# 12. 可选：特征筛选（此处跳过）

# 13. 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 14. 保存数据前，确保 skills_list 列为 Python 列表（再次转换，避免保存问题）
# 注意：划分后 X_train 和 X_test 中的 skills_list 可能还是 numpy 数组（如果原数据是 array），需要处理
for df_split in [X_train, X_test]:
    if 'skills_list' in df_split.columns:
        df_split['skills_list'] = df_split['skills_list'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

os.makedirs('data/processed', exist_ok=True)

X_train.to_parquet('data/processed/X_train.parquet')
X_test.to_parquet('data/processed/X_test.parquet')
y_train.to_frame().to_parquet('data/processed/y_train.parquet')
y_test.to_frame().to_parquet('data/processed/y_test.parquet')

# 保存完整编码数据（可选）
df_encoded.to_parquet('data/processed/df_encoded.parquet')

print("\n特征工程数据保存完成！")

# 15. 后续步骤提示
print("\n" + "="*50)
print("后续建模建议：")
print("1. 使用 scripts/word2vec_train.py 生成技能词向量，并拼接到 X_train/X_test 上。")
print("2. 使用 XGBoost 训练回归模型，评估性能并分析特征重要性。")
print("3. 根据特征重要性可进一步筛选特征，优化模型。")
print("="*50)