#!/usr/bin/env python3
"""
scripts/tableau_export.py
生成 Tableau 专用 CSV 文件
输出到 data/tableau/ 目录
运行：python scripts/tableau_export.py
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "data/tableau"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 读取清洗后数据 ────────────────────────────────────────────────────────
print("读取数据...")
try:
    df = pd.read_parquet("data/interim/cleaned_jobs.parquet")
    print(f"  从 parquet 读取成功，共 {len(df)} 行")
except Exception as e:
    print(f"  parquet 读取失败({e})，改用原始 CSV...")
    df1 = pd.read_csv("data/raw/ai_job_dataset.csv")
    df2 = pd.read_csv("data/raw/ai_job_dataset1.csv")
    cols = ['job_title','salary_usd','experience_level','employment_type',
            'company_location','company_size','remote_ratio','required_skills',
            'education_required','years_experience','industry',
            'posting_date','benefits_score']
    df = pd.concat([df1[cols], df2[cols]], ignore_index=True)
    df['company_location'] = df['company_location'].str.upper().str.strip()
    print(f"  原始 CSV 读取成功，共 {len(df)} 行")

# 统一国家列名
if 'country' not in df.columns:
    df['country'] = df['company_location'].str.upper().str.strip()

# ── 表1：主数据表（用于大部分图表）──────────────────────────────────────
print("\n生成表1：主数据表...")
df_main = df[[
    'job_title', 'salary_usd', 'experience_level', 'employment_type',
    'company_size', 'remote_ratio', 'education_required',
    'years_experience', 'industry', 'posting_date', 'benefits_score', 'country'
]].copy()

# 经验等级中文标签（Tableau显示用）
exp_label = {'EN': 'Entry (EN)', 'MI': 'Mid (MI)', 'SE': 'Senior (SE)', 'EX': 'Expert (EX)'}
df_main['exp_label'] = df_main['experience_level'].map(exp_label).fillna('Unknown')
df_main['exp_order'] = df_main['experience_level'].map({'EN':1,'MI':2,'SE':3,'EX':4})

# 公司规模标签
size_label = {'S': 'Small (S)', 'M': 'Medium (M)', 'L': 'Large (L)'}
df_main['size_label'] = df_main['company_size'].map(size_label).fillna('Unknown')
df_main['size_order'] = df_main['company_size'].map({'S':1,'M':2,'L':3})

# 发布年月
df_main['posting_date'] = pd.to_datetime(df_main['posting_date'], errors='coerce')
df_main['post_year']  = df_main['posting_date'].dt.year
df_main['post_month'] = df_main['posting_date'].dt.month
df_main['post_ym']    = df_main['posting_date'].dt.to_period('M').astype(str)

df_main.to_csv(f"{OUTPUT_DIR}/tableau_main.csv", index=False, encoding='utf-8-sig')
print(f"  已保存: {OUTPUT_DIR}/tableau_main.csv  ({len(df_main)} 行)")

# ── 表2：国家薪资汇总（用于地图）────────────────────────────────────────
print("\n生成表2：国家薪资汇总...")
# 国家名映射为 Tableau 地图能识别的标准英文名
country_name_map = {
    'UNITED STATES':  'United States',
    'UNITED KINGDOM': 'United Kingdom',
    'SOUTH KOREA':    'South Korea',
    'CHINA':          'China',
    'INDIA':          'India',
    'GERMANY':        'Germany',
    'CANADA':         'Canada',
    'FRANCE':         'France',
    'AUSTRALIA':      'Australia',
    'JAPAN':          'Japan',
    'SWEDEN':         'Sweden',
    'NORWAY':         'Norway',
    'DENMARK':        'Denmark',
    'FINLAND':        'Finland',
    'SWITZERLAND':    'Switzerland',
    'AUSTRIA':        'Austria',
    'IRELAND':        'Ireland',
    'ISRAEL':         'Israel',
    'NETHERLANDS':    'Netherlands',
    'SINGAPORE':      'Singapore',
}
df_country = (df_main.groupby('country')['salary_usd']
              .agg(avg_salary='mean', median_salary='median',
                   job_count='count', max_salary='max')
              .reset_index())
df_country['country_name'] = df_country['country'].map(country_name_map).fillna(df_country['country'].str.title())
df_country['avg_salary']    = df_country['avg_salary'].round(0)
df_country['median_salary'] = df_country['median_salary'].round(0)
df_country = df_country.sort_values('avg_salary', ascending=False)

df_country.to_csv(f"{OUTPUT_DIR}/tableau_country.csv", index=False, encoding='utf-8-sig')
print(f"  已保存: {OUTPUT_DIR}/tableau_country.csv  ({len(df_country)} 行)")
print(df_country[['country_name','avg_salary','job_count']].to_string(index=False))

# ── 表3：经验等级 × 薪资分位数（用于箱线图）────────────────────────────
print("\n生成表3：经验等级薪资分位数...")
df_exp = (df_main.groupby(['experience_level','exp_label','exp_order'])['salary_usd']
          .describe(percentiles=[.25,.5,.75])
          .reset_index())
df_exp.columns = ['experience_level','exp_label','exp_order',
                  'count','mean','std','min','q25','median','q75','max']
df_exp = df_exp.sort_values('exp_order')
df_exp.to_csv(f"{OUTPUT_DIR}/tableau_exp_salary.csv", index=False, encoding='utf-8-sig')
print(f"  已保存: {OUTPUT_DIR}/tableau_exp_salary.csv")
print(df_exp[['exp_label','mean','median','q25','q75']].to_string(index=False))

# ── 表4：行业薪资汇总（用于条形图）──────────────────────────────────────
print("\n生成表4：行业薪资汇总...")
df_industry = (df_main.groupby('industry')['salary_usd']
               .agg(avg_salary='mean', job_count='count')
               .reset_index()
               .sort_values('avg_salary', ascending=False))
df_industry['avg_salary'] = df_industry['avg_salary'].round(0)
df_industry.to_csv(f"{OUTPUT_DIR}/tableau_industry.csv", index=False, encoding='utf-8-sig')
print(f"  已保存: {OUTPUT_DIR}/tableau_industry.csv")
print(df_industry.to_string(index=False))

# ── 表5：月均薪资趋势（用于趋势折线）────────────────────────────────────
print("\n生成表5：月均薪资趋势...")
df_trend = (df_main.dropna(subset=['post_ym'])
            .groupby('post_ym')['salary_usd']
            .agg(avg_salary='mean', job_count='count')
            .reset_index()
            .sort_values('post_ym'))
df_trend['avg_salary'] = df_trend['avg_salary'].round(0)
df_trend.to_csv(f"{OUTPUT_DIR}/tableau_trend.csv", index=False, encoding='utf-8-sig')
print(f"  已保存: {OUTPUT_DIR}/tableau_trend.csv  ({len(df_trend)} 行)")

# ── 表6：技能频次（用于条形图）───────────────────────────────────────────
print("\n生成表6：技能频次统计...")
skill_list = []
for row in df['required_skills'].dropna():
    for s in str(row).split(','):
        s = s.strip()
        if s:
            skill_list.append(s)
df_skills = (pd.Series(skill_list)
             .value_counts()
             .reset_index())
df_skills.columns = ['skill', 'count']
df_skills.to_csv(f"{OUTPUT_DIR}/tableau_skills.csv", index=False, encoding='utf-8-sig')
print(f"  已保存: {OUTPUT_DIR}/tableau_skills.csv  ({len(df_skills)} 个技能)")
print(df_skills.head(10).to_string(index=False))

print("\n" + "="*50)
print("Tableau 数据导出完成！")
print(f"所有文件在: {OUTPUT_DIR}/")
print("  tableau_main.csv     — 主数据（薪资预测散点/箱线）")
print("  tableau_country.csv  — 国家汇总（地图）")
print("  tableau_exp_salary.csv — 经验等级统计（箱线图）")
print("  tableau_industry.csv — 行业汇总（条形图）")
print("  tableau_trend.csv    — 月均趋势（折线图）")
print("  tableau_skills.csv   — 技能频次（条形图）")
print("="*50)