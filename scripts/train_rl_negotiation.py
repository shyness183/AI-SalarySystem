"""
训练薪资谈判RL模型
运行：python scripts/train_rl_negotiation.py
输出：models/rl_negotiation.zip
耗时：约10-20分钟（CPU）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from rl_negotiation_env import SalaryNegotiationEnv
import numpy as np

def make_env(market_salary=None):
    """随机化市场薪资，提升泛化能力"""
    def _init():
        salary  = market_salary or np.random.choice(
            [58000, 72000, 92000, 105000, 119000, 132000, 148000])
        exp     = np.random.randint(0, 4)
        size    = np.random.randint(0, 3)
        return SalaryNegotiationEnv(salary, exp, size)
    return _init

# 路径：兼容从项目根目录或 scripts/ 内运行
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)

# 创建向量化训练环境（8个并行）
vec_env = make_vec_env(make_env(), n_envs=8)

# PPO模型
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=os.path.join(ROOT_DIR, "models", "rl_logs"),
)

# 评估回调
eval_env = make_vec_env(make_env(market_salary=119000), n_envs=4)
eval_cb  = EvalCallback(eval_env,
                        best_model_save_path=os.path.join(ROOT_DIR, "models"),
                        log_path=os.path.join(ROOT_DIR, "models", "rl_logs"),
                        eval_freq=10000, n_eval_episodes=20,
                        deterministic=True, verbose=1)

print("开始训练（约500,000步）...")
model.learn(total_timesteps=500_000, callback=eval_cb, progress_bar=True)

save_path = os.path.join(ROOT_DIR, "models", "rl_negotiation")
model.save(save_path)
print(f"训练完成！模型已保存：{save_path}.zip")

# 简单测试
print("\n=== 测试谈判场景（市场基准$119,000） ===")
test_env = SalaryNegotiationEnv(119000, exp_level=2, company_size=1)
obs, _ = test_env.reset()
for _ in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = test_env.step(action)
    action_name = ["坚持", "小幅退让", "大幅退让"][action]
    print(f"  Round {info['round']}: 求职者出价 ${info['candidate_offer']:,} "
          f"({action_name}) | HR出价 ${info['hr_offer']:,}")
    if done:
        print(f"  成交价：${info['final_salary']:,} "
              f"（市场基准的{info['final_salary']/119000*100:.1f}%）")
        break
