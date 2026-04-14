# AI 薪资预测系统

> 
> 基于 30,000 条全球 AI 岗位数据，集成机器学习预测、强化学习谈判、AI 就业顾问的一体化薪资分析平台

---

## 项目概览

本系统以全球 AI 行业招聘数据为基础，构建了从数据清洗、特征工程、模型训练到 Web 交互的完整链路，提供以下核心能力：

| 模块      | 技术                 | 功能                                            |
| ------- | ------------------ | --------------------------------------------- |
| 薪资预测    | XGBoost + Word2Vec | 输入职位信息，预测年薪及置信区间                              |
| 趋势分析    | LSTM 时间序列          | 历史薪资走势 + 未来 3 个月预测                            |
| 职位分类分析  | 关键词规则分类            | 8 类 AI 职位薪资对比（中位数/均值/分位数）                     |
| 薪资谈判模拟  | PPO 强化学习           | 与 RL 智能体多轮谈判，学习最优报价策略                         |
| AI 就业顾问 | LLM 流式对话           | 基于用户档案的个性化职业建议（支持 DeepSeek / Claude / OpenAI） |
| 可解释性分析  | SHAP               | 单次预测的特征贡献可视化                                  |

---

## 技术栈

**后端**

- Python 3.12 · Flask 2.3.3 · Flask-Login · Flask-SQLAlchemy
- XGBoost 1.7.6 · TensorFlow 2.16.1 · Stable-Baselines3 · Gymnasium
- Gensim Word2Vec · SHAP · Scikit-learn · yfinance

**前端**

- Jinja2 模板 · ECharts 5.4.3 · marked.js · 原生 JavaScript

**数据处理**

- Apache Spark（数据清洗）· Pandas · PyArrow（Parquet 格式）

**存储与部署**

- SQLite（用户/预测/聊天记录）· Docker + Docker Compose

---

## 目录结构

```
salary-system/
├── data/
│   ├── raw/                    # 原始 CSV 数据集（30,000 条 AI 岗位）
│   ├── interim/                # Spark 清洗后的 Parquet 数据
│   ├── processed/              # 特征矩阵、Word2Vec 模型、职位分类统计
│   └── salary_system.db        # SQLite 数据库
├── models/
│   ├── xgboost_optimized.pkl   # XGBoost 薪资预测模型（R²=0.8721）
│   ├── lstm_trend.keras        # LSTM 趋势预测模型（MAPE=1.17%）
│   ├── rl_negotiation.zip      # PPO 强化学习谈判模型
│   └── shap_figures/           # SHAP 可视化图表
├── scripts/
│   ├── clean_data.py           # Spark 数据清洗
│   ├── features_engineering.py # 特征工程（153 维特征）
│   ├── word2vec_train.py       # 技能词向量训练（100 维）
│   ├── lstm_trend.py           # LSTM 训练与预测
│   ├── shap_analysis.py        # SHAP 特征解释
│   ├── job_title_classify.py   # 职位关键词分类
│   ├── rl_negotiation_env.py   # RL 谈判环境（Gymnasium）
│   └── train_rl_negotiation.py # PPO 模型训练
├── web/
│   ├── app.py                  # Flask 主应用
│   ├── requirements.txt        # Python 依赖
│   └── templates/              # HTML 模板
├── Dockerfile
└── docker-compose.yml
```

---

## 快速启动

### 方式一：Docker（推荐，无需配置 Python 环境）

前提：已有训练好的模型文件（`models/` 目录非空）

```bash
docker compose up -d
```

访问 http://localhost:5000

### 方式二：本地运行

```bash
# 安装依赖
pip install -r web/requirements.txt
pip install "stable-baselines3[extra]>=2.3.0" "gymnasium>=0.29.0"

# 启动 Flask
cd web
python app.py
```

访问 http://localhost:5000

---

## 数据与模型训练流程

> 以下步骤仅在首次搭建或重新训练时需要执行，Docker 部署用户**无需**重跑。

```bash
# 1. 数据清洗（需要 Spark 环境）
python scripts/clean_data.py

# 2. 特征工程
python scripts/features_engineering.py

# 3. 训练 Word2Vec 技能词向量
python scripts/word2vec_train.py

# 4. 训练 XGBoost 薪资预测模型
python models/train_XGBoost.py

# 5. 训练 LSTM 趋势预测模型
python scripts/lstm_trend.py

# 6. 生成职位分类薪资统计
python scripts/job_title_classify.py

# 7. 训练 RL 谈判模型（约 2 分钟）
python scripts/train_rl_negotiation.py
```

---

## 主要功能说明

### 数据看板 `/dashboard`

四个 Tab：**全球概览**（世界地图 + AI 股票行情）· **薪资趋势**（历史走势 + LSTM 预测）· **特征分析**（Top 10 特征重要性）· **职位分类分析**（8 类 AI 职位横向薪资对比）

### 薪资预测 `/predict`

输入：经验等级、工作年限、国家、公司规模、行业、远程比例、技能列表
输出：预测年薪（±17% 置信区间）+ 技能词向量匹配状态 + SHAP 特征贡献分析
登录用户的预测记录自动存入数据库。

### 薪资趋势 `/trend`

基于 LSTM 模型的月均薪资时间序列分析，含历史数据回溯与未来 3 个月预测区间。

### 职业建议 `/advice`

- **AI 建议**：根据经验等级、国家、行业、技能数，输出经验晋升 / 地区对标 / 技能提升 / 谈判策略四维建议
- **AI 对话**：基于用户档案的流式对话（SSE），支持 DeepSeek / Claude / OpenAI，API Key 存储在浏览器 localStorage，不上传服务端

### 薪资谈判模拟 `/negotiate`

基于 PPO 算法（Stable-Baselines3）训练的强化学习智能体，模拟 HR 谈判博弈：

- **状态空间（5 维）**：当前轮次 / 双方出价 / 经验等级 / 公司规模
- **动作空间（3 个）**：坚持报价 / 小幅退让 -3% / 大幅退让 -8%
- **奖励函数**：成交薪资 / 市场基准 - 1（鼓励高薪成交，惩罚拖延）
- 训练步数 500,000，测试场景成交价约为市场基准的 105%

### 用户系统

Flask-Login + SQLAlchemy，支持注册/登录/退出/个人档案，密码 Werkzeug 哈希存储。

---

## 路由总览

| 路由                  | 方法       | 说明                |
| ------------------- | -------- | ----------------- |
| `/`                 | GET      | 首页（项目介绍）          |
| `/dashboard`        | GET      | 数据看板              |
| `/predict`          | GET/POST | 薪资预测              |
| `/trend`            | GET      | 趋势分析              |
| `/advice`           | GET      | 职业建议 + AI 对话      |
| `/negotiate`        | GET      | 薪资谈判模拟            |
| `/settings`         | GET      | AI 设置（API Key 配置） |
| `/login`            | GET/POST | 登录                |
| `/register`         | GET/POST | 注册                |
| `/logout`           | GET      | 退出登录              |
| `/profile`          | GET      | 个人档案（需登录）         |
| `/api/dashboard`    | GET      | 看板数据接口            |
| `/api/predict`      | POST     | 预测接口              |
| `/api/trend`        | GET      | 趋势数据接口            |
| `/api/advice`       | POST     | 建议生成接口            |
| `/api/shap`         | POST     | SHAP 解释接口         |
| `/api/chat`         | POST     | AI 对话流式接口（SSE）    |
| `/api/negotiate`    | POST     | RL 谈判接口           |
| `/api/stock-data`   | GET      | AI 股票行情（10 分钟缓存）  |
| `/api/validate-key` | POST     | API Key 验证        |

---

## 模型性能

| 模型           | 指标    | 数值          |
| ------------ | ----- | ----------- |
| XGBoost 薪资预测 | R²    | 0.8721      |
| XGBoost 薪资预测 | RMSE  | 0.1864      |
| LSTM 趋势预测    | MAPE  | 1.17%       |
| LSTM 趋势预测    | MAE   | $1,402      |
| PPO 谈判模型     | 测试成交价 | 市场基准 × 105% |

**Top 3 薪资影响因素**（SHAP 分析）：经验等级（Gain 105.79）> 工作年限（46.91）> 所在国家（瑞士 16.36）

---

## 环境变量

| 变量                     | 默认值                          | 说明                       |
| ---------------------- | ---------------------------- | ------------------------ |
| `SECRET_KEY`           | `salary-system-dev-key-2025` | Flask Session 密钥，生产环境请替换 |
| `FLASK_ENV`            | `production`                 | 运行模式                     |
| `TF_CPP_MIN_LOG_LEVEL` | `3`                          | 抑制 TensorFlow 日志         |

---

## 注意事项

- **NumPy 版本**：TensorFlow 2.16.1 要求 `numpy<2.0`，当前锁定 1.26.4，与 stable-baselines3 兼容
- **RL 模型**：模型文件与训练时的 NumPy 版本绑定，切换 NumPy 版本后需重新运行 `train_rl_negotiation.py`
- **API Key**：AI 对话功能的 Key 仅存于用户浏览器 localStorage，服务端不持久化
- **数据库**：SQLite 文件位于 `data/salary_system.db`，Docker 挂载 `./data` 目录确保持久化

---

## 开发者
steaker-shyness183
