import pandas as pd
import numpy as np
import os

# 读取 Parquet 数据
data_path = "data\interim\cleaned_jobs.parquet"
df = pd.read_parquet(data_path)

print(f"数据维度: {df.shape}")
print("前5行:\n", df.head())
print("列信息:\n", df.info())

# EDA: 薪资分布
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.histplot(df['salary_usd'], bins=50, kde=True)
plt.title('Salary Distribution (USD)')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# 薪资与经验等级、公司规模、行业的关系
# 按经验等级分组统计薪资
exp_order = ['EN', 'MI', 'SE', 'EX']  # 定义顺序
df['experience_level'] = pd.Categorical(df['experience_level'], categories=exp_order, ordered=True)
plt.figure(figsize=(10,5))
sns.boxplot(x='experience_level', y='salary_usd', data=df)
plt.title('Salary by Experience Level')
plt.show()

# 公司规模（S, M, L）
plt.figure(figsize=(10,5))
sns.boxplot(x='company_size', y='salary_usd', data=df)
plt.title('Salary by Company Size')
plt.show()

# 行业（取前10高频行业）
top_industries = df['industry'].value_counts().head(10).index
df_top = df[df['industry'].isin(top_industries)]
plt.figure(figsize=(12,6))
sns.boxplot(x='industry', y='salary_usd', data=df_top)
plt.xticks(rotation=45)
plt.title('Salary by Industry (Top 10)')
plt.show()

# 技能数量与薪资关系

# skills_array 是 Spark 清洗时生成的列表，如果 Pandas 读取后是字符串形式，需转换
import ast
df['skills_array'] = df['skills_array'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['skill_count'] = df['skills_array'].apply(len)

plt.figure(figsize=(10,5))
sns.scatterplot(x='skill_count', y='salary_usd', data=df, alpha=0.3)
plt.title('Skill Count vs Salary')
plt.show()

#   薪资与远程比例

plt.figure(figsize=(10,5))
sns.boxplot(x='remote_ratio', y='salary_usd', data=df)
plt.title('Salary by Remote Ratio')
plt.show()

