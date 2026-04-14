"""
web/app.py  v3
修复：Word2Vec 路径自动探测 + 启动时验证词汇表
"""
import os, datetime, json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import (Flask, render_template, request, jsonify, Response,
                   redirect, url_for, flash)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import httpx as _httpx
    _httpx_ok = True
except ImportError:
    _httpx_ok = False

try:
    import anthropic as _anthropic
    _anthropic_ok = True
except ImportError:
    _anthropic_ok = False
try:
    import shap as shap_lib
    _shap_ok = True
except ImportError:
    _shap_ok = False

try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    from stable_baselines3 import PPO as _PPO
    from rl_negotiation_env import SalaryNegotiationEnv as _SalaryNegotiationEnv
    _rl_PPO = _PPO
    _rl_Env = _SalaryNegotiationEnv
    _rl_ok = True
except Exception as _e:
    _rl_ok = False
    print(f"  [WARN] RL依赖未就绪: {_e}")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "salary-system-dev-key-2025")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 数据库配置 ─────────────────────────────────────────────────────────────
_root_dir = os.path.dirname(BASE_DIR)
_data_dir = os.path.join(_root_dir, "data")
os.makedirs(_data_dir, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(_data_dir, "salary_system.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "请先登录"


# ── 数据库模型 ─────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(64),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    predictions   = db.relationship("Prediction", backref="user", lazy=True)
    chats         = db.relationship("ChatHistory", backref="user", lazy=True)

    def set_password(self, pwd):
        self.password_hash = generate_password_hash(pwd)

    def check_password(self, pwd):
        return check_password_hash(self.password_hash, pwd)


class Prediction(db.Model):
    __tablename__    = "predictions"
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    input_params     = db.Column(db.Text)          # JSON 字符串
    predicted_salary = db.Column(db.Integer)
    percentile       = db.Column(db.Float)
    created_at       = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class ChatHistory(db.Model):
    __tablename__   = "chat_history"
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    prediction_id   = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=True)
    role            = db.Column(db.String(16))   # 'user' | 'assistant'
    content         = db.Column(db.Text)
    created_at      = db.Column(db.DateTime, default=datetime.datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()

# Jinja2 自定义 filter
@app.template_filter("from_json")
def from_json_filter(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

# Docker容器里 WORKDIR=/app，模型挂载在 /app/models
# 本地运行时 app.py 在 web/，模型在上一级的 models/
if os.path.exists(os.path.join(BASE_DIR, "models")):
    # Docker环境：模型就在 /app/models
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
else:
    # 本地环境：模型在项目根目录的 models/
    ROOT_DIR  = os.path.dirname(BASE_DIR)
    MODEL_DIR = os.path.join(ROOT_DIR, "models")
    PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")

# ── 完整153维特征列 ────────────────────────────────────────────────────────
BASE_FEATURES = [
    'remote_ratio','years_experience','job_description_length','benefits_score',
    'exp_level_encoded','edu_level_encoded','size_encoded',
    'post_year','post_month','post_quarter','deadline_year','deadline_month',
    'days_to_deadline','skill_count',
    'ind_Automotive','ind_Consulting','ind_Education','ind_Energy','ind_Finance',
    'ind_Gaming','ind_Government','ind_Healthcare','ind_Manufacturing','ind_Media',
    'ind_Real Estate','ind_Retail','ind_Technology','ind_Telecommunications','ind_Transportation',
    'emp_CT','emp_FL','emp_FT','emp_PT',
    'ctry_AUSTRALIA','ctry_AUSTRIA','ctry_CANADA','ctry_CHINA','ctry_DENMARK',
    'ctry_FINLAND','ctry_FRANCE','ctry_GERMANY','ctry_INDIA','ctry_IRELAND',
    'ctry_ISRAEL','ctry_JAPAN','ctry_NETHERLANDS','ctry_NORWAY','ctry_SINGAPORE',
    'ctry_SOUTH KOREA','ctry_SWEDEN','ctry_SWITZERLAND','ctry_UNITED KINGDOM','ctry_UNITED STATES',
]
EMB_FEATURES = [f'skill_emb_{i}' for i in range(100)]
ALL_FEATURES = BASE_FEATURES + EMB_FEATURES   # 153维

EXP_MAP  = {'EN':0,'MI':1,'SE':2,'EX':3}
EDU_MAP  = {'Associate':3,'Bachelor':0,'Master':1,'PhD':2,'Unknown':0}
SIZE_MAP = {'S':0,'M':1,'L':2,'unknown':0}
W2V_DIM  = 100

# 词汇表里全部24个技能词（小写），供前端展示
ALL_SKILLS = [
    "Python", "SQL", "TensorFlow", "Kubernetes", "PyTorch", "Scala",
    "Linux", "Git", "Java", "GCP", "Hadoop", "R", "Tableau",
    "Computer Vision", "Data Visualization", "Spark", "MLOps", "Azure",
    "Deep Learning", "NLP", "AWS", "Mathematics", "Docker", "Statistics"
]

print("正在加载模型...")
xgb_model  = joblib.load(os.path.join(MODEL_DIR, "xgboost_optimized.pkl"))
print("  [OK] XGBoost 模型")
lstm_model  = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_trend.keras"))
print("  [OK] LSTM 模型")
lstm_scaler = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
print("  [OK] LSTM 归一化器")

# SHAP explainer（TreeExplainer 对 XGBoost 速度最快）
shap_explainer = None
if _shap_ok:
    try:
        shap_explainer = shap_lib.TreeExplainer(xgb_model)
        print("  [OK] SHAP Explainer 初始化完成")
    except Exception as e:
        print(f"  [WARN] SHAP 初始化失败: {e}")
else:
    print("  [WARN] shap 库未安装，跳过 SHAP 功能")
# Word2Vec —— 多路径探测
w2v_wv = None
W2V_SEARCH = [
    os.path.join(PROC_DIR,  "skills_word2vec.model"),
    os.path.join(MODEL_DIR, "skills_word2vec.model"),
    
]
for p in W2V_SEARCH:
    if os.path.exists(p):
        try:
            from gensim.models import Word2Vec
            w2v_wv = Word2Vec.load(p).wv
            print(f"  [OK] Word2Vec 加载自 {p}")
            print(f"       词汇量: {len(w2v_wv)}  示例: {list(w2v_wv.index_to_key[:6])}")
            # 快速验证
            ok = sum(1 for s in ALL_SKILLS if s in w2v_wv)
            print(f"       内置技能词命中: {ok}/{len(ALL_SKILLS)}")
        except Exception as e:
            print(f"  [WARN] 加载失败 {p}: {e}")
            w2v_wv = None
        break
else:
    print(f"  [WARN] Word2Vec 模型未找到，已搜索: {W2V_SEARCH}")
    print("         技能词向量将全部为零，预测仍然有效")

monthly_df  = pd.read_csv(os.path.join(PROC_DIR, "monthly_avg.csv"))
forecast_df = pd.read_csv(os.path.join(PROC_DIR, "lstm_forecast.csv"))

# 用于分位数计算的原始薪资数据（保留 salary_usd / country / experience_level）
_interim_dir = os.path.join(os.path.dirname(PROC_DIR), "interim")
_cleaned_path = os.path.join(_interim_dir, "cleaned_jobs.parquet")
if os.path.exists(_cleaned_path):
    salary_ref_df = pd.read_parquet(_cleaned_path,
                                    columns=["salary_usd", "country", "experience_level"])
    print(f"  [OK] 薪资参照数据加载完成，共 {len(salary_ref_df):,} 条")
else:
    salary_ref_df = None
    print("  [WARN] cleaned_jobs.parquet 未找到，分位数功能不可用")

print(f"  [OK] 数据文件加载完成，特征维度={len(ALL_FEATURES)}\n")


# ── 工具函数 ──────────────────────────────────────────────────────────────
# ── SHAP 特征名映射 ──────────────────────────────────────────────────────
_FEAT_MAP = {
    'exp_level_encoded':'Experience Level','years_experience':'Years of Experience',
    'size_encoded':'Company Size','edu_level_encoded':'Education Required',
    'remote_ratio':'Remote Ratio','skill_count':'Skill Count',
    'job_description_length':'JD Length','benefits_score':'Benefits Score',
    'days_to_deadline':'Days to Deadline','post_month':'Post Month',
    'post_quarter':'Post Quarter','post_year':'Post Year',
    'deadline_month':'Deadline Month','deadline_year':'Deadline Year',
}
def _map_feat(name):
    if name in _FEAT_MAP: return _FEAT_MAP[name]
    if name.startswith('ctry_'):     return 'Country: ' + name[5:].title()
    if name.startswith('ind_'):      return 'Industry: ' + name[4:]
    if name.startswith('emp_'):      return 'EmpType: ' + name[4:]
    if name.startswith('skill_emb_'):return f'SkillVec[{name[10:]}]'
    return name


def skill_str_to_vector(skill_str: str):
    """
    把逗号分隔的技能字符串转为100维词向量 + 返回匹配情况。
    词汇表实际存储的是原始大小写（如 Python, TensorFlow, SQL）。
    匹配策略（优先级依次降低）：
      1. 原始输入直接命中
      2. 首字母大写（python -> Python）
      3. 全大写（sql -> SQL）
      4. 大小写不敏感（建 lower->原词 映射兜底）
    """
    if w2v_wv is None or not skill_str.strip():
        return np.zeros(W2V_DIM), [], []

    # 建立 lower -> 词汇表原始词 的映射（兜底用）
    lower_to_vocab = {token.lower(): token for token in w2v_wv.key_to_index}

    raw_skills = [s.strip() for s in skill_str.split(',') if s.strip()]
    matched, unmatched, vecs = [], [], []

    for s in raw_skills:
        # 依次尝试四种形式
        candidates = [
            s,                          # 原始输入：python / Python / PYTHON
            s.title(),                  # 首字母大写：Python / Deep Learning
            s.upper(),                  # 全大写：SQL / NLP / GCP / AWS
            lower_to_vocab.get(s.lower())  # 兜底映射
        ]
        hit = None
        for c in candidates:
            if c and c in w2v_wv:
                hit = c
                break
        if hit:
            vecs.append(w2v_wv[hit])
            matched.append(f"{s}→{hit}" if hit != s else s)
        else:
            unmatched.append(s)

    vec = np.max(vecs, axis=0) if vecs else np.zeros(W2V_DIM)
    return vec, matched, unmatched


def build_input(form: dict):
    now = datetime.datetime.now()
    row = {f: 0 for f in ALL_FEATURES}

    row.update({
        'remote_ratio':           int(form.get('remote_ratio', 0)),
        'years_experience':       int(form.get('years_experience', 3)),
        'job_description_length': int(form.get('job_description_length', 1200)),
        'benefits_score':         float(form.get('benefits_score', 7.0)),
        'exp_level_encoded':      EXP_MAP.get(form.get('experience_level','MI'), 1),
        'edu_level_encoded':      EDU_MAP.get(form.get('education_required','Bachelor'), 0),
        'size_encoded':           SIZE_MAP.get(form.get('company_size','M'), 1),
        'post_year':    now.year,
        'post_month':   now.month,
        'post_quarter': (now.month - 1) // 3 + 1,
        'deadline_year':  now.year if now.month < 12 else now.year + 1,
        'deadline_month': now.month + 1 if now.month < 12 else 1,
        'days_to_deadline': 30,
        'skill_count':  int(form.get('skill_count', 3)),
    })

    ind_col  = f"ind_{form.get('industry', 'Technology')}"
    emp_col  = f"emp_{form.get('employment_type', 'FT')}"
    ctry_col = f"ctry_{form.get('country', 'UNITED STATES').upper().strip()}"
    for col in (ind_col, emp_col, ctry_col):
        if col in row:
            row[col] = 1

    vec, matched, unmatched = skill_str_to_vector(form.get('skills', ''))
    for i, v in enumerate(vec):
        row[f'skill_emb_{i}'] = float(v)

    return pd.DataFrame([row])[ALL_FEATURES], matched, unmatched


# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def index():
    return render_template("index.html")


def get_job_category_data():
    csv_path = os.path.join(PROC_DIR, "job_category_salary.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    return [
        {
            "category": row["job_category"],
            "median": round(row["median"], 0),
            "mean": round(row["mean"], 0),
            "count": int(row["count"]),
            "q25": round(row["q25"], 0),
            "q75": round(row["q75"], 0),
        }
        for _, row in df.iterrows()
    ]


@app.route("/api/dashboard")
def api_dashboard():
    hist = {"months": monthly_df["month_str"].tolist(),
            "actuals": monthly_df["mean_salary"].round(0).tolist()}
    lstm_data = {
        "months":    forecast_df["month"].tolist(),
        "actual":    [round(float(v),0) if pd.notna(v) else None
                      for v in pd.to_numeric(forecast_df["actual"],    errors="coerce")],
        "predicted": [round(float(v),0) if pd.notna(v) else None
                      for v in pd.to_numeric(forecast_df["predicted"], errors="coerce")],
        "type":      forecast_df["type"].tolist()
    }
    fi = [
        {"feature":"经验等级","gain":105.79}, {"feature":"工作年限","gain":46.91},
        {"feature":"国家:瑞士","gain":16.36}, {"feature":"国家:丹麦","gain":10.67},
        {"feature":"国家:挪威","gain":9.84},  {"feature":"国家:美国","gain":7.16},
        {"feature":"公司规模","gain":6.55},   {"feature":"国家:中国","gain":6.54},
        {"feature":"国家:印度","gain":5.72},  {"feature":"国家:韩国","gain":3.04},
    ]
    country_salaries = [
        {"name": "United States",   "value": 135000},
        {"name": "Switzerland",     "value": 148000},
        {"name": "Norway",          "value": 132000},
        {"name": "Denmark",         "value": 128000},
        {"name": "Singapore",       "value": 128000},
        {"name": "Canada",          "value": 110000},
        {"name": "Finland",         "value": 112000},
        {"name": "Sweden",          "value": 118000},
        {"name": "Israel",          "value": 118000},
        {"name": "Germany",         "value": 105000},
        {"name": "Netherlands",     "value": 103000},
        {"name": "Ireland",         "value": 101000},
        {"name": "Australia",       "value": 108000},
        {"name": "United Kingdom",  "value": 98000},
        {"name": "Austria",         "value": 96000},
        {"name": "France",          "value": 92000},
        {"name": "Japan",           "value": 82000},
        {"name": "South Korea",     "value": 75000},
        {"name": "China",           "value": 72000},
        {"name": "India",           "value": 58000},
    ]
    return jsonify({"hist":hist,"lstm":lstm_data,"feature_importance":fi,
                    "country_salaries": country_salaries,
                    "job_categories": get_job_category_data(),
                    "metrics":{"xgb":{"rmse":0.1864,"r2":0.8721},
                               "lstm":{"mape":1.17,"mae":1402}}})


@app.route("/predict", methods=["GET","POST"])
def predict():
    result, form_data = None, {}
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            input_df, matched, unmatched = build_input(form_data)
            log_pred    = float(xgb_model.predict(input_df)[0])
            salary_pred = int(np.exp(log_pred))

            skill_str   = form_data.get('skills','').strip()
            input_count = len([s for s in skill_str.split(',') if s.strip()]) if skill_str else 0

            if not skill_str:
                skill_note = ("info",
                    "未输入技能。可从下方技能标签中选择，技能词向量将纳入预测。")
            elif w2v_wv is None:
                skill_note = ("warn",
                    "Word2Vec 模型文件未找到（已搜索 data/processed/ 和 models/ 目录），"
                    "请确认 skills_word2vec.model 路径正确。当前技能维度为零，其余特征仍有效。")
            elif matched and not unmatched:
                skill_note = ("ok",
                    f"全部 {len(matched)} 个技能均已命中词汇表"
                    f"（{', '.join(matched)}），词向量已纳入预测。")
            elif matched:
                skill_note = ("ok",
                    f"命中 {len(matched)}/{input_count} 个技能"
                    f"（✅ {', '.join(matched)}），"
                    f"未命中：{', '.join(unmatched)}（不在词汇表中，不影响已命中部分）。")
            else:
                # 全部未命中 —— 给出最可能的原因和建议
                similar = []
                for u in unmatched[:3]:
                    cands = [s for s in ALL_SKILLS if u.lower() in s or s in u.lower()]
                    if cands:
                        similar.append(u + ' -> ' + cands[0])
                hint = ("；".join(similar) + "。") if similar else \
                       "请从下方技能标签直接点击选择，确保名称完全一致。"
                skill_note = ("warn",
                    f"输入了 {input_count} 个技能，全部未命中词汇表，技能维度为零。"
                    f"{hint}")

            result = {
                "salary":        f"${salary_pred:,}",
                "salary_low":    f"${int(salary_pred*0.83):,}",
                "salary_high":   f"${int(salary_pred*1.17):,}",
                "salary_raw":    salary_pred,
                "log_pred":      round(log_pred, 4),
                "skill_note":    skill_note,
                "low_gain_note": ("根据模型特征重要性（gain），学历、雇佣类型、福利评分、"
                                  "职位描述长度等字段的Gain值较低，这些字段变化时预测结果"
                                  "变动幅度小是正常现象，并非系统错误。"
                                  "影响最大的特征依次是：经验等级 > 工作年限 > 所在国家。"),
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("predict.html",
                           result=result,
                           form_data=form_data,
                           all_skills=ALL_SKILLS,
                           w2v_ready=(w2v_wv is not None))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """薪资预测 JSON 接口，供前端 AJAX 调用"""
    data = request.get_json() or {}
    try:
        input_df, matched, unmatched = build_input(data)
        log_pred    = float(xgb_model.predict(input_df)[0])
        salary_pred = int(np.exp(log_pred))

        skill_str   = data.get("skills", "").strip()
        input_count = len([s for s in skill_str.split(",") if s.strip()]) if skill_str else 0

        if not skill_str:
            skill_note = ["info", "未输入技能。可从技能标签中选择，技能词向量将纳入预测。"]
        elif w2v_wv is None:
            skill_note = ["warn", "Word2Vec 模型未找到，当前技能维度为零，其余特征仍有效。"]
        elif matched and not unmatched:
            skill_note = ["ok", f"全部 {len(matched)} 个技能已命中词汇表（{', '.join(matched)}）。"]
        elif matched:
            skill_note = ["ok",
                          f"命中 {len(matched)}/{input_count} 个技能（{', '.join(matched)}），"
                          f"未命中：{', '.join(unmatched)}。"]
        else:
            skill_note = ["warn",
                          f"输入了 {input_count} 个技能，全部未命中词汇表，技能维度为零。"]

        resp = {
            "salary":      f"${salary_pred:,}",
            "salary_low":  f"${int(salary_pred * 0.83):,}",
            "salary_high": f"${int(salary_pred * 1.17):,}",
            "salary_raw":  salary_pred,
            "log_pred":    round(log_pred, 4),
            "skill_note":  skill_note,
        }

        # 登录用户自动保存预测历史
        if current_user.is_authenticated:
            try:
                rec = Prediction(
                    user_id=current_user.id,
                    input_params=json.dumps(data, ensure_ascii=False),
                    predicted_salary=salary_pred,
                )
                db.session.add(rec)
                db.session.commit()
                resp["prediction_id"] = rec.id
            except Exception:
                db.session.rollback()

        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/trend")
def trend():
    return render_template("trend.html")


@app.route("/api/trend")
def api_trend():
    return jsonify({
        "months":    forecast_df["month"].tolist(),
        "actual":    [round(float(v),0) if pd.notna(v) else None
                      for v in pd.to_numeric(forecast_df["actual"],    errors="coerce")],
        "predicted": [round(float(v),0) if pd.notna(v) else None
                      for v in pd.to_numeric(forecast_df["predicted"], errors="coerce")],
        "type":      forecast_df["type"].tolist(),
        "upper_bound": [
            round(float(v)*1.02, 0) if pd.notna(v) and t == "forecast" else None
            for v, t in zip(pd.to_numeric(forecast_df["predicted"], errors="coerce"),
                            forecast_df["type"])
        ],
        "lower_bound": [
            round(float(v)*0.98, 0) if pd.notna(v) and t == "forecast" else None
            for v, t in zip(pd.to_numeric(forecast_df["predicted"], errors="coerce"),
                            forecast_df["type"])
        ],
        "metrics":   {"rmse":1428,"mae":1402,"mape":1.17},
        "forecast_summary": [{"month":"2025-05","salary":119787},
                              {"month":"2025-06","salary":119616},
                              {"month":"2025-07","salary":120292}]
    })


@app.route("/advice")
def advice():
    return render_template("advice.html")


@app.route("/api/advice", methods=["POST"])
def api_advice():
    data        = request.get_json()
    exp_level   = data.get("experience_level","MI")
    country     = data.get("country","UNITED STATES").upper()
    industry    = data.get("industry","Technology")
    skill_count = int(data.get("skill_count",3))
    current_sal = int(data.get("current_salary",80000))

    exp_advice = {
        "EN":"您当前处于入门级，建议深耕1-2项核心技能（如Python、机器学习），争取2年内晋升至中级。",
        "MI":"您处于中级阶段，可考虑主导项目、积累团队协作经验，向高级工程师方向发展。",
        "SE":"高级阶段薪资提升空间较大，可向专家级（EX）进发，或转型为技术负责人/架构师。",
        "EX":"您已处于专家级，薪资天花板主要受公司规模和地区影响，可考虑北欧/北美高薪市场。"
    }
    benchmarks = {
        "UNITED STATES":135000,"SWITZERLAND":148000,"NORWAY":132000,"DENMARK":128000,
        "CANADA":110000,"GERMANY":105000,"UNITED KINGDOM":98000,"CHINA":72000,
        "INDIA":58000,"SOUTH KOREA":75000,"JAPAN":82000,"AUSTRALIA":108000,
        "FRANCE":92000,"SWEDEN":118000,"FINLAND":112000,
    }
    bm  = benchmarks.get(country, 100000)
    gap = current_sal - bm
    geo = (f"您的薪资高于{country}地区基准（${bm:,}）约${gap:,}，竞争力较强。" if gap > 10000
           else f"您的薪资低于{country}地区基准（${bm:,}）约${abs(gap):,}，建议通过晋级或跳槽提升。" if gap < -10000
           else f"您的薪资与{country}地区基准（${bm:,}）基本持平。")

    skill_map = {
        "Technology":["LLM/大模型应用","MLOps/模型部署","云计算（AWS/GCP）"],
        "Finance":["量化分析","风险建模","Python量化库"],
        "Healthcare":["医疗AI合规","NLP病历分析","联邦学习"],
        "Consulting":["数据可视化（Tableau）","项目管理（PMP）","商业分析"],
    }
    skills = skill_map.get(industry, ["Python","机器学习","数据可视化"])
    sk = (f"技能储备偏少（{skill_count}项），建议补充：{' / '.join(skills[:2])}" if skill_count < 4
          else f"技能储备充足，建议深化：{' / '.join(skills)}")

    neg = ("您处于高价值区间，议价空间约15-25%。"
           if exp_level in ["SE","EX"] and country in ["UNITED STATES","SWITZERLAND","NORWAY"]
           else "入门级薪资弹性较小，建议优先积累经验，2-3年后是最佳跳槽窗口。"
           if exp_level == "EN"
           else "建议提供竞品offer作为参考，议价空间约10-15%。")

    return jsonify({
        "exp_advice":exp_advice.get(exp_level,""),"geo_advice":geo,
        "skill_advice":sk,"negotiate":neg,
        "next_level":{"EN":"MI","MI":"SE","SE":"EX","EX":None}.get(exp_level),
        "benchmark_salary":f"${bm:,}","your_salary":f"${current_sal:,}"
    })


@app.route("/api/shap", methods=["POST"])
def api_shap():
    """单样本 SHAP 解释，返回 Top 8 特征贡献"""
    if shap_explainer is None:
        return jsonify({"error": "SHAP 未初始化，请安装 shap 库"}), 500
    data = request.get_json()
    try:
        input_df, _, _ = build_input(data)
        sv   = shap_explainer.shap_values(input_df)[0]
        base = float(shap_explainer.expected_value)

        top_idx  = np.argsort(np.abs(sv))[::-1][:8]
        features = [_map_feat(ALL_FEATURES[i]) for i in top_idx]
        values   = [round(float(sv[i]), 4)     for i in top_idx]

        return jsonify({
            "features":    features,
            "shap_values": values,
            "base_value":  round(base, 4),
            "prediction":  round(float(np.sum(sv) + base), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/salary-percentile")
def api_salary_percentile():
    """
    计算预测薪资在同国家+同经验等级样本中的百分位排名。
    参数（query string）：
      salary    — 预测薪资（美元，整数）
      country   — 国家名，大写，如 UNITED STATES
      exp_level — 经验等级 EN / MI / SE / EX
    """
    if salary_ref_df is None:
        return jsonify({"error": "参照数据未加载，分位数功能不可用"}), 503

    try:
        salary    = float(request.args.get("salary", 0))
        country   = request.args.get("country", "").strip().upper()
        exp_level = request.args.get("exp_level", "").strip().upper()
    except ValueError:
        return jsonify({"error": "参数格式错误"}), 400

    if not country or not exp_level:
        return jsonify({"error": "缺少必要参数 country 或 exp_level"}), 400

    subset = salary_ref_df[
        (salary_ref_df["country"] == country) &
        (salary_ref_df["experience_level"] == exp_level)
    ]["salary_usd"]

    n = len(subset)
    MIN_SAMPLES = 10
    if n < MIN_SAMPLES:
        return jsonify({
            "enough": False,
            "message": f"样本量不足（{n} 条），无法计算分位数",
            "sample_size": n
        })

    percentile = float((subset < salary).mean() * 100)
    return jsonify({
        "enough":     True,
        "percentile": round(percentile, 1),
        "sample_size": n,
        "median":     round(float(subset.median()), 0),
        "p25":        round(float(subset.quantile(0.25)), 0),
        "p75":        round(float(subset.quantile(0.75)), 0),
    })


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=request.form.get("remember") == "on")
            next_page = request.args.get("next") or url_for("index")
            return redirect(next_page)
        error = "用户名或密码错误"
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        if not username or not email or not password:
            error = "请填写所有必填项"
        elif password != confirm:
            error = "两次密码不一致"
        elif len(password) < 6:
            error = "密码长度至少 6 位"
        elif User.query.filter_by(username=username).first():
            error = "用户名已存在"
        elif User.query.filter_by(email=email).first():
            error = "邮箱已注册"
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for("index"))
    return render_template("register.html", error=error)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/profile")
@login_required
def profile():
    preds = (Prediction.query
             .filter_by(user_id=current_user.id)
             .order_by(Prediction.created_at.desc())
             .limit(20).all())
    chats = (ChatHistory.query
             .filter_by(user_id=current_user.id)
             .order_by(ChatHistory.created_at.desc())
             .limit(10).all())
    return render_template("profile.html",
                           predictions=preds,
                           recent_chats=chats)


@app.route("/api/chat/save", methods=["POST"])
def api_chat_save():
    """登录用户保存对话记录（由前端在 SSE 结束后调用）"""
    if not current_user.is_authenticated:
        return jsonify({"ok": False})
    data = request.get_json() or {}
    try:
        prediction_id = data.get("prediction_id")
        for msg in data.get("messages", []):
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                rec = ChatHistory(
                    user_id=current_user.id,
                    prediction_id=prediction_id,
                    role=role,
                    content=content,
                )
                db.session.add(rec)
        db.session.commit()
        return jsonify({"ok": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/stock-data")
def api_stock_data():
    """获取 AI 相关股票行情，内存缓存 10 分钟"""
    now = datetime.datetime.utcnow()
    cache = getattr(app, "_stock_cache", None)
    if cache and (now - cache["ts"]).seconds < 600:
        return jsonify(cache["data"])

    STOCKS = [
        ("NVDA",   "英伟达"),
        ("MSFT",   "微软"),
        ("GOOGL",  "谷歌"),
        ("META",   "Meta"),
        ("BIDU",   "百度"),
    ]
    result = []
    try:
        import yfinance as yf
        for symbol, name in STOCKS:
            try:
                t     = yf.Ticker(symbol)
                info  = t.fast_info
                price = round(float(info.last_price), 2)
                prev  = round(float(info.previous_close), 2)
                chg   = round(price - prev, 2)
                pct   = round(chg / prev * 100, 2) if prev else 0
                result.append({"symbol": symbol, "name": name,
                                "price": price, "change": chg,
                                "change_pct": pct, "up": chg >= 0})
            except Exception:
                result.append({"symbol": symbol, "name": name,
                                "price": None, "change": 0,
                                "change_pct": 0, "up": True})
    except ImportError:
        pass

    app._stock_cache = {"ts": now, "data": result}
    return jsonify(result)


@app.route("/api/validate-key", methods=["POST"])
def api_validate_key():
    """向目标 API 发一个最小请求，验证 Key 是否有效"""
    data     = request.get_json() or {}
    api_key  = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()
    model    = data.get("model", "").strip()

    if not api_key:
        return jsonify({"ok": False, "error": "API Key 不能为空"}), 400

    # Anthropic 分支
    if "anthropic.com" in base_url:
        if not _anthropic_ok:
            return jsonify({"ok": False, "error": "服务端未安装 anthropic 库"}), 500
        try:
            client = _anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

    # OpenAI-compatible 分支
    if not _httpx_ok:
        return jsonify({"ok": False, "error": "服务端未安装 httpx 库"}), 500
    try:
        resp = _httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=15
        )
        if resp.status_code in (200, 201):
            return jsonify({"ok": True})
        body = resp.text[:200]
        return jsonify({"ok": False, "error": f"HTTP {resp.status_code}: {body}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


def _build_system_prompt(profile: dict) -> str:
    return f"""你是一位专业的AI就业薪资顾问，拥有全球AI行业薪资数据库的访问权限。

用户职业信息：
- 经验等级：{profile.get('exp_level', '未填写')}
- 国家：{profile.get('country', '未填写')}
- 工作年限：{profile.get('years', '未填写')} 年
- 技能：{profile.get('skills', '未填写')}
- 行业：{profile.get('industry', '未填写')}
- 系统预测薪资：${profile.get('salary', '待预测')}

全球AI薪资基准（2025年最新数据）：
- 全球月均 $119,000 | 美国 $135,000 | 瑞士 $148,000 | 挪威 $132,000
- 中国 $72,000 | 印度 $58,000 | 德国 $105,000 | 英国 $98,000
- 最大薪资影响因素：工作年限 > 经验等级 > 所在国家（SHAP分析结论）
- XGBoost 模型 R²=0.8721，预测误差约 ±17%

回答要求：
- 用中文，语言简洁直接，不超过200字
- 使用 Markdown 格式：重点用 **加粗**，分点用 `-`，分段清晰
- 不要废话开场（如"好的"、"当然"、"非常感谢"），直接给结论
- 数字要具体，给出美元金额或百分比"""


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """AI就业顾问流式对话接口（SSE）"""
    data         = request.get_json() or {}
    api_key      = data.get("api_key", "").strip()
    base_url     = data.get("base_url", "").strip()
    model        = data.get("model", "").strip()
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])
    profile      = data.get("profile", {})

    if not api_key:
        return jsonify({"error": "请先在设置中配置 API Key"}), 400
    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    system_prompt = _build_system_prompt(profile)

    # Anthropic 原生 SDK
    if "anthropic.com" in base_url:
        if not _anthropic_ok:
            return jsonify({"error": "服务端未安装 anthropic 库"}), 500

        def generate_anthropic():
            try:
                client = _anthropic.Anthropic(api_key=api_key)
                messages = [m for m in history if m.get("role") in ("user", "assistant")]
                messages.append({"role": "user", "content": user_message})
                with client.messages.stream(
                    model=model or "claude-sonnet-4-6",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for text in stream.text_stream:
                        chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate_anthropic(), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    # OpenAI-compatible
    if not _httpx_ok:
        return jsonify({"error": "服务端未安装 httpx 库"}), 500

    messages = [{"role": "system", "content": system_prompt}]
    messages += [m for m in history if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": user_message})

    def generate_openai():
        try:
            with _httpx.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "stream": True},
                timeout=60
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate_openai(), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/settings")
def settings():
    return render_template("settings.html")


# ── RL 谈判模型懒加载 ──────────────────────────────────────────────────────
_rl_model = None

def _load_rl_model():
    global _rl_model
    if _rl_model is not None:
        return _rl_model
    if not _rl_ok:
        return None
    rl_path = os.path.join(MODEL_DIR, "rl_negotiation")
    if os.path.exists(rl_path + ".zip"):
        _rl_model = _rl_PPO.load(rl_path)
    return _rl_model


@app.route("/negotiate")
def negotiate_page():
    return render_template("negotiate.html")


@app.route("/api/negotiate", methods=["POST"])
def api_negotiate():
    try:
        return _do_negotiate()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _do_negotiate():
    global _rl_model
    # 每次请求重新尝试加载（支持训练完成后热加载）
    _rl_model = None
    model = _load_rl_model()
    if model is None:
        return jsonify({"error": "RL模型未加载，请先运行 python scripts/train_rl_negotiation.py"}), 500

    data          = request.get_json()
    market_salary = float(data.get("market_salary", 119000))
    exp_level     = int(data.get("exp_level", 1))
    company_size  = int(data.get("company_size", 1))

    env = _rl_Env(market_salary, exp_level, company_size)
    obs, _ = env.reset()

    rounds       = []
    action_names = ["坚持报价", "小幅退让 (-3%)", "大幅退让 (-8%)"]
    advice_map   = {
        0: "当前出价合理，坚持有助于争取更高薪资。",
        1: "适度退让表现诚意，有助于推动谈判。",
        2: "大幅退让适合快速促成，但可能低于市场价值。"
    }

    for _ in range(env.max_rounds):
        action, _ = model.predict(obs, deterministic=True)
        action    = int(action)
        obs, reward, done, _, info = env.step(action)

        rounds.append({
            "round":           info["round"],
            "action":          action_names[action],
            "advice":          advice_map[action],
            "candidate_offer": info["candidate_offer"],
            "hr_offer":        info["hr_offer"],
            "final_salary":    info["final_salary"],
            "done":            done,
        })
        if done:
            break

    final     = rounds[-1]["final_salary"]
    vs_market = (final / market_salary - 1) * 100

    return jsonify({
        "rounds":        rounds,
        "final_salary":  final,
        "market_salary": market_salary,
        "vs_market":     round(vs_market, 1),
        "hr_budget":     info["hr_budget"],
        "result_text": (
            f"最终成交价 ${final:,}，"
            f"{'高于' if vs_market > 0 else '低于'}市场基准 {abs(vs_market):.1f}%。"
            + ("表现优秀！" if vs_market > 5 else
               "接近市场水平。" if vs_market > -5 else
               "建议下次提高初始报价。")
        )
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)