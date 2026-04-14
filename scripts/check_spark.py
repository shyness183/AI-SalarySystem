from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ValidateCleanedData").getOrCreate()
df = spark.read.parquet("data/interim/cleaned_jobs.parquet")

print("数据总行数:", df.count())
print("列数:", len(df.columns))
print("列名:", df.columns)
df.printSchema()
df.show(5, truncate=False)

from pyspark.sql.functions import col, sum as spark_sum

# 计算每列的缺失值数量
null_counts = df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0].asDict()
print("每列缺失值数量:")
for col_name, count in null_counts.items():
    print(f"  {col_name}: {count}")

# 特别检查薪资列应无缺失
print("薪资列是否有缺失？", df.filter(col("salary_usd").isNull()).count() == 0)

print("薪资 <= 0 的记录数:", df.filter(col("salary_usd") <= 0).count())
print("薪资 >= 1,000,000 的记录数:", df.filter(col("salary_usd") >= 1000000).count())


df.describe("salary_usd").show()

df.select("company_location", "country").distinct().show(20, truncate=False)

df.select("required_skills", "skills_array").show(10, truncate=False)
df.select("job_id").show(10)
# 检查 job_id 是否从 AI00001 开始递增

# 随机抽取 5 条记录
df.sample(0.001, seed=42).show(5, vertical=True)