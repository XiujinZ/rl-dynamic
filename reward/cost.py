"""
reward/cost.py

成本（Cost）相关的工具模块。

本文件主要完成两件事：
1）离线计算：每新增 1 个 E（感染）所对应的期望医疗成本
2）定义：每封控 1 人 · 1 天 所造成的生产力损失成本

所有成本单位统一为：万元（10,000 CNY）

设计原则：
- 所有复杂计算在离线阶段完成
- 训练 / 仿真阶段只使用“固化常数”
- 与 env / RL 完全解耦
"""

from dataclasses import dataclass
import numpy as np


# ======================================================
# 1. 成本参数容器
# ======================================================

@dataclass
class CostParams:
    """
    成本相关参数（用于离线标定）

    """

    # ---------- 概率参数 ----------
    # 感染后进入住院状态的概率 P(H | E)
    hospitalization_prob: float = 0.059

    # 住院后死亡的条件概率 P(D | H)
    death_prob: float = 0.139

    # ---------- 医疗费用（单位：万元） ----------
    # 住院但最终未死亡个体的平均住院费用
    cost_hosp_survive: float = 1.54

    # 住院并最终死亡个体的 ICU 费用
    cost_icu_death: float = 7.88

    # ---------- 生产力损失 ----------
    # 每封控 1 人 · 1 天 造成的生产力损失（万元）
    cost_lockdown_per_person_day: float = 0.0308


# ======================================================
# 2. 医疗成本计算：每 1 个 E 的期望医疗费用
# ======================================================

def _compute_expected_medical_cost_per_E(params: CostParams) -> float:
    """
    计算：每新增 1 个 E（感染）带来的期望医疗成本

    计算逻辑（全概率公式）：
        E[C_med | E] =
            P(H|E) × [
                (1 - P(D|H)) × 住院费用
                + P(D|H) × ICU 费用
            ]

    参数:
        params: CostParams，成本与概率参数

    返回:
        expected_medical_cost: float
            每 1 个感染的期望医疗成本（单位：万元）
    """

    p_h = params.hospitalization_prob
    p_d = params.death_prob

    expected_medical_cost = (
        p_h
        * (
            (1.0 - p_d) * params.cost_hosp_survive
            + p_d * params.cost_icu_death
        )
    )

    return float(expected_medical_cost)


# ======================================================
# 3. 固化后的成本常数（训练时直接使用）
# ======================================================

# 每新增 1 个 E（感染）对应的期望医疗成本（万元）
MEDICAL_COST_PER_E: float = _compute_expected_medical_cost_per_E(CostParams())

# 每封控 1 人 · 1 天 的生产力损失成本（万元）
LOCKDOWN_COST_PER_PERSON_DAY: float = CostParams().cost_lockdown_per_person_day


# ======================================================
# 4. 离线校准 / 参数检查入口
# ======================================================

if __name__ == "__main__":
    """
    仅在手动运行本文件时执行，用于：
    - 打印当前参数下的期望医疗成本
    - 做 sanity check
    """

    params = CostParams()

    medical_cost = _compute_expected_medical_cost_per_E(params)

    print("【成本参数离线校准】")
    print(f"每新增 1 个感染的期望医疗成本: {medical_cost:.4f} 万元")
    print(f"每封控 1 人·天的生产力损失: "
          f"{params.cost_lockdown_per_person_day:.4f} 万元")
