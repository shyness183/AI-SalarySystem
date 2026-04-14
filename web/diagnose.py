"""
诊断脚本 —— 放到 web/ 目录下运行：python diagnose.py
逐步检查每个加载步骤，找到500错误的根本原因
"""
import os, sys
print("=" * 55)
print("Flask 500 错误诊断")
print("=" * 55)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
PROC_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")

print(f"\n[路径]")
print(f"  BASE_DIR  = {BASE_DIR}")
print(f"  MODEL_DIR = {os.path.abspath(MODEL_DIR)}")
print(f"  PROC_DIR  = {os.path.abspath(PROC_DIR)}")

# ── 1. 检查文件是否存在 ───────────────────────────────────────────────────
print(f"\n[文件存在检查]")
files = {
    "XGBoost模型":    os.path.join(MODEL_DIR, "xgboost_optimized.pkl"),
    "LSTM模型":       os.path.join(MODEL_DIR, "lstm_trend.keras"),
    "LSTM归一化器":   os.path.join(MODEL_DIR, "lstm_scaler.pkl"),
    "Word2Vec模型":   os.path.join(PROC_DIR,  "skills_word2vec.model"),
    "月均CSV":        os.path.join(PROC_DIR,  "monthly_avg.csv"),
    "预测CSV":        os.path.join(PROC_DIR,  "lstm_forecast.csv"),
}
all_ok = True
for name, path in files.items():
    exists = os.path.exists(path)
    flag   = "✅" if exists else "❌ 缺失"
    print(f"  {flag}  {name}: {path}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n⚠️  有文件缺失，请先确认路径正确，或重新运行训练脚本生成对应文件。")

# ── 2. 逐个加载测试 ───────────────────────────────────────────────────────
print(f"\n[逐步加载测试]")

# joblib
print("\n→ 加载 XGBoost...")
try:
    import joblib
    m = joblib.load(os.path.join(MODEL_DIR, "xgboost_optimized.pkl"))
    print(f"  ✅ 成功  类型: {type(m)}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# tensorflow
print("\n→ 加载 LSTM (tensorflow)...")
try:
    import tensorflow as tf
    lm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_trend.keras"))
    print(f"  ✅ 成功  输入shape: {lm.input_shape}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# lstm scaler
print("\n→ 加载 LSTM scaler...")
try:
    import joblib
    sc = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
    print(f"  ✅ 成功  类型: {type(sc)}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# word2vec
print("\n→ 加载 Word2Vec...")
try:
    from gensim.models import Word2Vec
    w2v = Word2Vec.load(os.path.join(PROC_DIR, "skills_word2vec.model"))
    print(f"  ✅ 成功  词汇量: {len(w2v.wv)}  示例: {list(w2v.wv.index_to_key[:6])}")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# pandas csv
print("\n→ 读取 CSV 数据...")
try:
    import pandas as pd
    df1 = pd.read_csv(os.path.join(PROC_DIR, "monthly_avg.csv"))
    df2 = pd.read_csv(os.path.join(PROC_DIR, "lstm_forecast.csv"))
    print(f"  ✅ monthly_avg: {len(df1)} 行  lstm_forecast: {len(df2)} 行")
except Exception as e:
    print(f"  ❌ 失败: {e}")

# ── 3. 检查 templates 目录 ────────────────────────────────────────────────
print(f"\n[Templates 目录检查]")
tpl_dir = os.path.join(BASE_DIR, "templates")
if os.path.isdir(tpl_dir):
    files_in = os.listdir(tpl_dir)
    print(f"  templates/ 目录存在，包含: {files_in}")
    for f in ["base.html","index.html","predict.html","trend.html","advice.html"]:
        flag = "✅" if f in files_in else "❌ 缺失"
        print(f"    {flag} {f}")
else:
    print(f"  ❌ templates/ 目录不存在: {tpl_dir}")

# ── 4. 模拟 predict GET 请求 ──────────────────────────────────────────────
print(f"\n[模拟 render_template 调用]")
try:
    sys.path.insert(0, BASE_DIR)
    from flask import Flask
    app = Flask(__name__)
    with app.app_context():
        from flask import render_template
        # 最简单的渲染测试
        result = None
        form_data = {}
        ALL_SKILLS_TEST = ["Python","SQL","TensorFlow"]
        html = render_template("predict.html",
                               result=result,
                               form_data=form_data,
                               all_skills=ALL_SKILLS_TEST,
                               w2v_ready=False)
        print(f"  ✅ render_template 成功，HTML长度: {len(html)} 字符")
except Exception as e:
    print(f"  ❌ render_template 失败: {e}")

print("\n" + "=" * 55)
print("诊断完成。请把以上输出发给我，我来定位具体问题。")
print("=" * 55)