#!/usr/bin/env python3
# scripts/clean_data.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, split, trim, when, upper, size, row_number, lit,
    concat, lpad
)
from pyspark.sql.types import DoubleType, IntegerType

# 初始化 SparkSession
spark = SparkSession.builder \
    .appName("AIJobsDataCleaning") \
    .getOrCreate()

# 定义数据目录（容器内路径）
RAW_DIR = "/opt/spark/data/raw"
OUTPUT_DIR = "/opt/spark/data/interim"

# 1. 读取两个 CSV 文件
print("正在读取 ai_job_dataset.csv ...")
df1 = spark.read.csv(f"{RAW_DIR}/ai_job_dataset.csv", header=True, inferSchema=True)

print("正在读取 ai_job_dataset1.csv ...")
df2 = spark.read.csv(f"{RAW_DIR}/ai_job_dataset1.csv", header=True, inferSchema=True)

# 2. 统一列名（确保两个 DataFrame 具有完全相同的列）
#    两个文件都包含 salary_usd，我们保留它，忽略 salary_local
common_columns = [
    "job_id", "job_title",
    col("salary_usd").cast(DoubleType()),
    "salary_currency",
    "experience_level", "employment_type",
    "company_location", "company_size", "employee_residence",
    "remote_ratio", "required_skills", "education_required",
    col("years_experience").cast(IntegerType()),
    "industry", "posting_date", "application_deadline",
    col("job_description_length").cast(IntegerType()),
    col("benefits_score").cast(DoubleType()),
    "company_name"
]

df1 = df1.select(*common_columns)
df2 = df2.select(*common_columns)

# 3. 合并数据集
print("合并数据集...")
df = df1.unionByName(df2, allowMissingColumns=False)
print(f"合并后总行数: {df.count()}, 总列数: {len(df.columns)}")

# 4. 重新生成 job_id（连续编号，格式 AI00001）
print("重新生成 job_id ...")
window_spec = Window.orderBy(lit(1))  # 按当前顺序生成行号
df = df.withColumn("row_num", row_number().over(window_spec))
df = df.withColumn("new_job_id", concat(lit("AI"), lpad(df["row_num"].cast("string"), 5, "0")))
df = df.drop("job_id", "row_num").withColumnRenamed("new_job_id", "job_id")
# 将 job_id 移到第一列
df = df.select("job_id", *[c for c in df.columns if c != "job_id"])

# 5. 检查缺失值（统计每列缺失数量）
print("缺失值统计（每列缺失数）:")
for c in df.columns:
    null_count = df.filter(col(c).isNull()).count()
    if null_count > 0:
        print(f"  列 {c}: 缺失 {null_count} 行")

# 6. 处理缺失值
print("处理缺失值...")
# 薪资缺失直接删除（无法预测）
df = df.dropna(subset=["salary_usd"])
# required_skills 缺失填充为空字符串
df = df.fillna({"required_skills": ""})
# company_size 缺失填充为 'unknown'
df = df.fillna({"company_size": "unknown"})
# 其他分类字段填充 'Unknown'
categorical_cols = ["experience_level", "employment_type", "education_required", "industry", "company_name"]
for col_name in categorical_cols:
    df = df.fillna({col_name: "Unknown"})

# 7. 过滤异常薪资（负值或超过 1,000,000 USD）
print("过滤异常薪资（负值或超过 1,000,000 USD）...")
df = df.filter((col("salary_usd") > 0) & (col("salary_usd") < 1000000))

# 8. 将 required_skills 拆分为数组
print("将 required_skills 拆分为数组...")
df = df.withColumn("skills_array", split(col("required_skills"), ",\\s*"))

# 9. 国家名称标准化：转为大写并去除空格
print("标准化国家名称...")
df = df.withColumn("country", upper(trim(col("company_location"))))

# 10. 处理日期字段（转换为日期类型）
df = df.withColumn("posting_date", col("posting_date").cast("date"))
df = df.withColumn("application_deadline", col("application_deadline").cast("date"))

# 11. 写入清洗后的数据（Parquet 格式）
print(f"正在写入 Parquet 到 {OUTPUT_DIR}/cleaned_jobs.parquet ...")
df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/cleaned_jobs.parquet")



print("清洗完成！")
spark.stop()