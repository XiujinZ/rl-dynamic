"""
reward/qaly.py

计算健康损失（QALY loss）的工具函数。
核心用途：计算「每新增 1 个 E（感染）」带来的期望 QALY 损失（YLD + YLL）。

设计原则：
- 所有复杂计算在离线阶段完成
- 训练 / 仿真阶段只使用“固化常数”
- 与 env / RL 完全解耦
"""

from dataclasses import dataclass, field
import numpy as np


# ======================================================
# 1. 参数容器
# ======================================================

@dataclass
class QALYParams:
    # ---------- 年龄结构 ----------
    age_structure: np.ndarray = field(
        default_factory=lambda: np.array(
            [3.44, 6.35, 52.68, 21.25, 16.28]
        ) / 100.0
    )

    # ---------- 概率参数 ----------
    hospitalization_prob: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.00025, 0.02672, 0.09334, 0.15465]
        )
    )

    death_prob: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.039, 0.121, 0.03, 0.105, 0.227]
        )
    )

    # ---------- 健康效用 ----------
    u_norm: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.97, 0.97, 0.96, 0.93, 0.85]
        )
    )

    # ---------- 标量（这些没问题） ----------
    u_mild: float = 0.6
    u_severe: float = 0.43

    delta_mild: float = 5.6
    delta_hosp_mild: float = 5.9
    delta_hosp_severe: float = 11.0

    discounted_life_expectancy: np.ndarray = field(
        default_factory=lambda: np.array(
            [30.38, 29.48, 25.84, 18.33, 7.79]
        )
    )

# ======================================================
# 2. 核心函数：每 1 个 E 的期望 QALY 损失
# ======================================================

def compute_expected_qaly_loss_per_E(
    params: QALYParams = QALYParams(),
) -> float:
    """
    计算：每新增 1 个 E（感染）带来的期望 QALY 损失（YLD + YLL）

    返回:
        qaly_loss_per_E : float
            单位：年（QALY）
    """

    # --------- 展开参数 ---------
    age = np.asarray(params.age_structure)
    p_h = np.asarray(params.hospitalization_prob)
    p_d = np.asarray(params.death_prob)
    u_norm = np.asarray(params.u_norm)
    le = np.asarray(params.discounted_life_expectancy)

    u_mild = params.u_mild
    u_severe = params.u_severe

    d_mild = params.delta_mild / 365.0
    d_hosp_mild = params.delta_hosp_mild / 365.0
    d_hosp_severe = params.delta_hosp_severe / 365.0

    # --------- 校验 ---------
    assert np.isclose(age.sum(), 1.0), "age_structure 必须加和为 1"
    assert (
        age.shape
        == p_h.shape
        == p_d.shape
        == u_norm.shape
        == le.shape
    ), "所有按年龄分层的数组长度必须一致"

    # ==================================================
    # YLD（Years Lived with Disability）
    # ==================================================

    # 1) 非住院（轻症）
    yld_mild = (
        age
        * (1.0 - p_h)
        * (u_norm - u_mild)
        * d_mild
    )

    # 2) 住院但未死亡
    yld_hosp = (
        age
        * p_h
        * (1.0 - p_d)
        * (
            (u_norm - u_mild) * d_hosp_mild
            + (u_norm - u_severe) * d_hosp_severe
        )
    )

    total_yld = yld_mild.sum() + yld_hosp.sum()

    # ==================================================
    # YLL（Years of Life Lost）
    # ==================================================

    yll = age * p_h * p_d * le
    total_yll = yll.sum()

    return float(total_yld + total_yll)

# ======================================================
# 3. 固化后的“模型常数”（训练中直接用）
# ======================================================

#: Expected QALY loss per avoided infection (E)
#: Calibrated offline using demographic and clinical data
QALY_PER_E: float = 0.1


# ======================================================
# 4. 离线校准入口（只在你想重新算时用）
# ======================================================

if __name__ == "__main__":
    params = QALYParams()
    qaly = compute_expected_qaly_loss_per_E(params)
    print(f"[Calibration] Expected QALY loss per infection: {qaly:.5f}")

