import sys
import os

# 把项目根目录加入 sys.path
PROJECT_ROOT = "/home/zxj/my_pytorch_project"
sys.path.append(PROJECT_ROOT)

"""
测试 SEIRHD 环境在真实上海 OD 数据下的传播情况。
使用 UniformPolicy 控制 OD 流量。
随机挑选 10 个感染者作为初始感染源。
输出总感染曲线，并可选绘制感染/住院/死亡趋势。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from env.env_SEIRHD import SEIRHDEpidemicEnv
from env.env_func_seirhd import SEIRHDParams
from utils.utils_data_loader import load_od_matrices, load_residents
from policy.uniform_policy import UniformControlPolicy  # 假设您已经定义了 UniformControlPolicy


# ===============================
# 1. 数据加载
# ===============================
OD_DIR = "./data/OD_matrices"
RESIDENT_PATH = "./data/residents.npy"

od_mats = load_od_matrices(OD_DIR)
residents = load_residents(RESIDENT_PATH)
N = residents.shape[0]
T_total = od_mats.shape[0]
print(f"[INFO] 数据加载完毕：N={N}, T_total={T_total} 小时")

# ===============================
# 2. 定义 SEIRHD 参数
# ===============================
params = SEIRHDParams(
    beta=1.8/24,    # 传染率
    sigma=0.833/24,    # 潜伏期 → 感染的速率
    gamma_I=0.179/24,  # 感染者恢复率
    eta_I=0.169/24,   # 感染者住院率
    p_H=0.059,      # 感染者转住院概率
    gamma_H=0.128/24, # 住院者恢复率
    gamma_D=0.091/24, # 住院者死亡率
    p_D=0.139,      # 住院者死亡概率
    dt=1.0,       # 时间步（1小时）
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 3. 初始化环境
# ===============================
env = SEIRHDEpidemicEnv(
    od_mats=od_mats,
    residents=residents,
    epi_params=params,
    device=device,
    max_steps=T_total,  # 31天×24小时
)

# ===============================
# 4. 随机生成初始感染者
# ===============================
batch_size = 1
initial_I = torch.zeros(batch_size, N)

# 总人口总数
total_pop = residents.sum()
# 随机选取 10 个感染个体所在格子
infected_indices = random.sample(range(N), 10)
for idx in infected_indices:
    # 每个感染者所在格子的数量 = 1 人
    initial_I[0, idx] = 1.0

print(f"[INFO] 初始感染网格: {infected_indices}")

obs = env.reset(batch_size=batch_size, initial_infected=initial_I.to(device))

# ===============================
# 5. 初始化 UniformPolicy 策略
# ===============================
uniform_policy = UniformControlPolicy(310,alpha=0.1)  # alpha 取值示例，您可以根据需要调整

# ===============================
# 6. 仿真传播（使用 UniformPolicy 控制流量）
# ===============================
T_sim = T_total  # 31天 * 24小时
total_I, total_H, total_D = [], [], []
baseline_new_E = []

for t in range(T_sim):
    prev_state = env.state
    
    # 使用 UniformPolicy 获取 action
    action = uniform_policy.act(obs, t)
    
    # 环境执行一步，并返回状态、奖励、结束标志等信息
    obs, reward, done, info = env.step(action=action)
    
    # ---- 存量曲线（可视化用）----
    total_I.append(info["total_I"][0])
    total_H.append(info["total_H"][0])
    total_D.append(info["total_D"][0])

    # ---- baseline 新增感染数（reward 核心）----
    S_prev = prev_state[..., 0].sum().item()
    S_curr = env.state[..., 0].sum().item()
    newE = max(S_prev - S_curr, 0.0)
    baseline_new_E.append(newE)
    
    if t % 24 == 0:
        print(
            f"[Day {t//24:02d}] "
            f"newE={newE:.1f}, "
            f"I={info['total_I'][0]:.0f}, "
            f"H={info['total_H'][0]:.0f}, "
            f"D={info['total_D'][0]:.0f}"
        )

    if done.any():
        print(f"[END] 仿真在 t={t} 小时结束")
        break

# ===============================
# 7. 结果可视化（I(t)/N 并保存）
# ===============================

# 转成 numpy
total_I = np.array(total_I)
total_H = np.array(total_H)
total_D = np.array(total_D)

total_pop = residents.sum()

I_ratio = total_I / total_pop
H_ratio = total_H / total_pop
D_ratio = total_D / total_pop

# 时间轴（小时 & 天）
t_hours = np.arange(len(I_ratio))
t_days = t_hours / 24.0

plt.figure(figsize=(8, 5))

plt.plot(t_days, I_ratio, label="I / N", linewidth=2)
plt.plot(t_days, H_ratio, label="H / N", linestyle="--")
plt.plot(t_days, D_ratio, label="D / N", linestyle=":")

plt.xlabel("Time (days)")
plt.ylabel("Population fraction")
plt.title("SEIRHD Simulation (Fractions)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# 保存路径（推荐放到 results/）
os.makedirs("figure", exist_ok=True)
save_path = "figure/SEIRHD_I_fraction.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"[DONE] 曲线已保存至 {save_path}")

# ===============================
# 8. 保存计算Reward所需的baseline新增E序列
# ===============================
baseline_new_E = np.array(baseline_new_E)
np.save("data/baseline_new_E.npy", baseline_new_E)

print(f"[DONE] baseline_new_E 已保存，长度 = {len(baseline_new_E)}")
