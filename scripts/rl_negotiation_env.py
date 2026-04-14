"""
薪资谈判强化学习环境
状态空间：[当前轮次, 上次出价(归一化), 市场基准(归一化), 经验等级, 公司预算估计]
动作空间：Discrete(3) → 0=坚持, 1=小幅退让(-3%), 2=大幅退让(-8%)
奖励：成交薪资 / 市场基准 - 1（超出基准为正，低于为负）
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SalaryNegotiationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market_salary=100000, exp_level=1, company_size=1):
        super().__init__()
        self.market_salary   = market_salary    # 市场基准薪资
        self.exp_level       = exp_level        # 0-3
        self.company_size    = company_size     # 0=小 1=中 2=大
        self.max_rounds      = 6                # 最多6轮谈判
        self.hr_budget_ratio = self._calc_budget_ratio()

        # 动作空间：0=坚持, 1=小幅退让, 2=大幅退让
        self.action_space = spaces.Discrete(3)

        # 状态空间：[轮次/6, 当前出价/市场, HR出价/市场, 经验等级/3, 公司规模/2]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.5, 0.5, 0, 0], dtype=np.float32),
            high=np.array([1, 2.0, 2.0, 1, 1], dtype=np.float32)
        )

    def _calc_budget_ratio(self):
        """HR预算上限 = 市场基准 × 系数（受公司规模和经验等级影响）"""
        size_factor = {0: 0.95, 1: 1.05, 2: 1.20}[self.company_size]
        exp_factor  = {0: 0.90, 1: 1.00, 2: 1.10, 3: 1.25}.get(self.exp_level, 1.0)
        # 加入随机扰动模拟真实HR预算不确定性
        noise = np.random.uniform(0.95, 1.05)
        return size_factor * exp_factor * noise

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.round       = 0
        # 求职者初始出价：市场基准的110%-130%
        self.candidate_offer = self.market_salary * np.random.uniform(1.10, 1.30)
        # HR初始出价：市场基准的80%-95%
        self.hr_offer        = self.market_salary * np.random.uniform(0.80, 0.95)
        self.hr_budget       = self.market_salary * self.hr_budget_ratio
        self.done            = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.round / self.max_rounds,
            self.candidate_offer / self.market_salary,
            self.hr_offer / self.market_salary,
            self.exp_level / 3,
            self.company_size / 2,
        ], dtype=np.float32)

    def step(self, action):
        self.round += 1

        # 求职者根据动作调整出价
        if action == 0:   # 坚持
            pass
        elif action == 1: # 小幅退让
            self.candidate_offer *= 0.97
        elif action == 2: # 大幅退让
            self.candidate_offer *= 0.92

        # HR策略：如果求职者出价接近预算则接受，否则小幅上调
        if self.candidate_offer <= self.hr_budget:
            # HR接受，谈判成功
            final_salary = (self.candidate_offer + self.hr_offer) / 2
            reward = (final_salary / self.market_salary) - 1.0
            self.done = True
        else:
            # HR小幅上调出价（每轮提高2-5%）
            self.hr_offer *= np.random.uniform(1.02, 1.05)
            self.hr_offer  = min(self.hr_offer, self.hr_budget * 0.95)
            reward = -0.01  # 每多谈一轮轻微负向（鼓励尽快成交）
            final_salary = None

        # 超过最大轮次：以HR当前出价成交
        if self.round >= self.max_rounds and not self.done:
            final_salary = self.hr_offer
            reward = (final_salary / self.market_salary) - 1.0
            self.done = True

        info = {
            "round": self.round,
            "candidate_offer": round(self.candidate_offer),
            "hr_offer": round(self.hr_offer),
            "final_salary": round(final_salary) if final_salary else None,
            "hr_budget": round(self.hr_budget),
        }
        return self._get_obs(), reward, self.done, False, info
