"""
reward/reward_func.py

强化学习奖励函数（Reward Function）

本文件实现基于“成本-效果分析（CEA）”的 reward 计算，
核心思想是：

    reward_t
    = 健康收益（避免的感染 × QALY）
    - λ ×（医疗成本 + 封控导致的生产力损失）

其中：
- 健康收益与医疗成本均基于“相对于 baseline 的增量”
- baseline 轨迹在离线阶段已计算并保存
"""

import os
import numpy as np
import torch

from reward.qaly import QALY_PER_E
from reward.cost import (
    MEDICAL_COST_PER_E,
    LOCKDOWN_COST_PER_PERSON_DAY,
)

# ======================================================
# 1. baseline 数据加载
# ======================================================

# baseline_new_E.npy 的路径（与你的 baseline 脚本保持一致）
BASELINE_PATH = "results/baseline_new_E.npy"
_BASELINE_CACHE = None

# reward/reward_func.py

import os
import numpy as np
import torch

BASELINE_PATH = "results/baseline_new_E.npy"
_BASELINE_CACHE = None


def get_baseline_new_E():
    global _BASELINE_CACHE

    if _BASELINE_CACHE is None:
        if not os.path.exists(BASELINE_PATH):
            raise FileNotFoundError(
                f"未找到 baseline_new_E 文件: {BASELINE_PATH}，"
                f"请先运行 baseline 仿真脚本。"
            )
        _BASELINE_CACHE = np.load(BASELINE_PATH)

    return _BASELINE_CACHE

# ======================================================
# 2. 核心 reward 计算函数
# ======================================================

def compute_reward(
    prev_state: torch.Tensor,
    curr_state: torch.Tensor,
    action: torch.Tensor | None,
    t: int,
    wtp: float,
) -> torch.Tensor:
    """
    计算单个时间步的 reward（支持 batch）

    参数:
        prev_state: torch.Tensor
            上一时刻状态 [B, N, 6] (S, E, I, H, R, D)

        curr_state: torch.Tensor
            当前时刻状态 [B, N, 6]

        action: torch.Tensor 或 None
            本时间步采取的动作（OD 缩放矩阵）
            shape: [N, N] 或 [B, N, N]
            若为 None，表示无干预

        t: int
            当前时间步索引（用于读取 baseline_new_E）

        wtp: float
            willingness-to-pay（单位：万元 / QALY）

    返回:
        reward: torch.Tensor
            shape: [B]
    """

    device = curr_state.device
    batch_size = curr_state.shape[0]

    # ==================================================
    # 1. 计算 policy 下的新增感染数 newE_policy
    # ==================================================
    # 使用 S 的减少量来估计新增感染（稳健、无需改 env）

    S_prev = prev_state[..., 0].sum(dim=1)  # [B]
    S_curr = curr_state[..., 0].sum(dim=1)  # [B]

    newE_policy = torch.clamp(S_prev - S_curr, min=0.0)  # [B]

    # ==================================================
    # 2. 读取 baseline 的新增感染数
    # ==================================================
    baseline_new_E = get_baseline_new_E()
    
    if t >= len(baseline_new_E):
        # 若超过 baseline 长度，默认 baseline 为 0
        baseline_newE = 0.0
    else:
        baseline_newE = baseline_new_E[t]

    baseline_newE = torch.full(
        (batch_size,),
        float(baseline_newE),
        device=device,
    )  # [B]

    # ==================================================
    # 3. 计算“避免的新增感染数”
    # ==================================================

    delta_newE = baseline_newE - newE_policy  # [B]

    # ==================================================
    # 4. 健康收益（QALY）
    # ==================================================

    health_gain = delta_newE * QALY_PER_E  # [B]

    # ==================================================
    # 5. 医疗成本（期望，按新增感染）
    # ==================================================

    medical_cost = delta_newE * MEDICAL_COST_PER_E  # [B]

    # ==================================================
    # 6. 封控 / 隔离导致的生产力损失
    # ==================================================

    if action is None:
        lockdown_cost = torch.zeros(batch_size, device=device)
    else:
        # action 越小，封控越强
        # 这里用 (1 - action) 的平均值近似封控强度
        if action.dim() == 2:
            # [N, N] -> [1, N, N]
            act = action.unsqueeze(0)
        else:
            act = action

        lockdown_intensity = 1.0 - act  # [B, N, N]

        # 用平均封控强度 × 总人口 近似“被限制出行的人数”
        avg_lockdown_ratio = lockdown_intensity.mean(dim=(1, 2))  # [B]

        # 近似：每个时间步被封控的人数
        # 注意：现在用的封控成本是一个 “轻量近似版本”，后续更改为基于 OD × population 的精确人次
        num_restricted = avg_lockdown_ratio

        lockdown_cost = (
            num_restricted
            * LOCKDOWN_COST_PER_PERSON_DAY
        )  # [B]

    # ==================================================
    # 7. 合成最终 reward
    # ==================================================

    # λ = 1 / WTP
    lambda_cost = 1.0 / wtp

    reward = health_gain - lambda_cost * (medical_cost + lockdown_cost)

    return reward


