"""
职位类型分类 + 薪资对比分析
运行：python scripts/job_title_classify.py
输出：
  - data/processed/job_category_salary.csv
  - models/shap_figures/job_category_salary.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 职位归类规则（关键词匹配，优先级从高到低）
JOB_CATEGORIES = {
    "NLP / LLM":      ["nlp", "natural language", "text mining", "llm",
                        "large language", "bert", "gpt", "language model",
                        "speech", "translation"],
    "Computer Vision": ["computer vision", "cv engineer", "image", "video",
                        "visual", "opencv", "object detection", "segmentation",
                        "3d", "lidar"],
    "MLOps / Platform":["mlops", "ml platform", "ml infrastructure", "ai platform",
                        "model deployment", "devops", "data platform",
                        "data engineer", "pipeline"],
    "Research / Science":["research scientist", "research engineer", "ai researcher",
                          "scientist", "r&d", "phd", "principal scientist"],
    "Data Science":    ["data scientist", "data analyst", "analytics",
                        "business intelligence", "bi analyst", "quantitative"],
    "General ML / AI": ["machine learning", "deep learning", "ai engineer",
                        "ml engineer", "artificial intelligence", "neural network",
                        "algorithm engineer"],
}

def classify_job(title: str) -> str:
    title_lower = title.lower()
    for category, keywords in JOB_CATEGORIES.items():
        if any(kw in title_lower for kw in keywords):
            return category
    return "Other AI Roles"

def main():
    # 读取数据
    try:
        df = pd.read_parquet("data/interim/cleaned_jobs.parquet")
    except:
        import glob
        csvs = glob.glob("data/raw/*.csv")
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    df["job_category"] = df["job_title"].fillna("").apply(classify_job)

    # 统计各类别薪资
    stats = df.groupby("job_category")["salary_usd"].agg(
        count="count",
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        std="std"
    ).reset_index()
    stats = stats[stats["count"] >= 50]  # 过滤样本量不足的类别
    stats = stats.sort_values("median", ascending=False)

    # 保存CSV
    os.makedirs("data/processed", exist_ok=True)
    stats.to_csv("data/processed/job_category_salary.csv", index=False)
    print("职位分类统计：")
    print(stats[["job_category", "count", "mean", "median"]].to_string(index=False))

    # 生成箱线图
    os.makedirs("models/shap_figures", exist_ok=True)
    categories = stats["job_category"].tolist()
    box_data = [df[df["job_category"] == c]["salary_usd"].dropna().tolist()
                for c in categories]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(box_data, patch_artist=True, vert=True,
                    showfliers=False, notch=False)

    colors = ["#4361ee", "#3a86ff", "#7209b7", "#f72585", "#4cc9f0", "#06d6a0", "#ffd166"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Annual Salary (USD)", fontsize=12)
    ax.set_title("AI Job Salary Distribution by Role Category", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.grid(axis="y", alpha=0.3)

    # 在每个箱子上方标注中位数
    for i, (cat, row) in enumerate(stats.iterrows(), 1):
        ax.text(i, row["q75"] + 2000, f'${row["median"]/1000:.0f}K',
                ha="center", va="bottom", fontsize=9, color="#333")

    plt.tight_layout()
    plt.savefig("models/shap_figures/job_category_salary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图表已保存：models/shap_figures/job_category_salary.png")

if __name__ == "__main__":
    main()
