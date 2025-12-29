# env_func_seirhd.py
import torch
import math
from typing import Optional, Tuple

class SEIRHDParams:
    """
    存储 SEIRHD 模型的流行病学参数。

    所有速率都是按时间步（每小时）计算的。
    """
    def __init__(
        self,
        beta: float,      # S->E 传染率（每次接触 I 时感染的概率）
        sigma: float,     # E->I 转化率（暴露到感染的速率）
        gamma_I: float,   # I->R 恢复率（感染者恢复的速率）
        eta_I: float,     # I->H 住院率（感染者转为住院的速率）
        p_H: float,       # I->H 的概率（感染者转为住院的概率）
        gamma_H: float,   # H->R 恢复率（住院者恢复的速率）
        gamma_D: float,   # H->D 死亡率（住院者死亡的速率）
        p_D: float,       # H->D 死亡的概率（住院者死亡的概率）
        dt: float = 1.0,  # 时间步长（例如每小时为 1）
        deterministic: bool = True,  # 是否采用确定性（True）还是随机过程（False）
    ):
        self.beta = beta
        self.sigma = sigma
        self.gamma_I = gamma_I
        self.eta_I = eta_I
        self.p_H = p_H
        self.gamma_H = gamma_H
        self.gamma_D = gamma_D
        self.p_D = p_D
        self.dt = dt
        self.deterministic = deterministic


def init_state_from_residents(
    residents: torch.Tensor,          # [N] 每个网格的初始常驻居民人数
    batch_size: int = 1,
    initial_infected: Optional[torch.Tensor] = None  # [B, N] 初始感染者分布
) -> torch.Tensor:
    """
    根据常驻居民数和初始感染者数初始化 SEIRHD 状态。
    
    返回初始状态 [B, N, 6]，包括 S, E, I, H, R, D 六个 compartment。
    """
    # 初始化每个网格的 S（常驻居民数）
    S0 = residents.unsqueeze(0).expand(batch_size, -1)  # [B, N]
    E0 = torch.zeros_like(S0)  # [B, N], 暴露者
    I0 = torch.zeros_like(S0)  # [B, N], 感染者
    H0 = torch.zeros_like(S0)  # [B, N], 住院者
    R0 = torch.zeros_like(S0)  # [B, N], 康复者
    D0 = torch.zeros_like(S0)  # [B, N], 死亡者

    # 如果有初始感染者，将 S 扣除感染者数量
    if initial_infected is not None:
        initial_infected = initial_infected.to(S0.device, dtype=S0.dtype)
        assert initial_infected.shape == S0.shape
        I0 = initial_infected
        S0 = (S0 - initial_infected).clamp(min=0.0)  # 确保 S0 不为负数

    # 将所有 compartment 堆叠成一个 [B, N, 6] 的状态张量
    state0 = torch.stack([S0, E0, I0, H0, R0, D0], dim=-1)  # [B, N, 6]
    return state0


def seirhd_local_step(
    state: torch.Tensor,           # [B, N, 6] (S,E,I,H,R,D)
    params: "SEIRHDParams",
    allow_H_infectious: bool = False,  # 如果想让 H 也有一定传染性，可以后面开这个开关
) -> torch.Tensor:
    """
    单个时间步的本地 SEIRHD 更新（不含流动），确定性版本。

    state:  [B, N, 6]，分别为 S,E,I,H,R,D
    params: SEIRHDParams 实例（包含 beta, sigma, gamma_I, eta_I, p_H, gamma_H, gamma_D, p_D, dt 等）

    返回:
        next_state: [B, N, 6]
    """
    device = state.device
    dtype = state.dtype

    # 将最后一维拆开：每个都是 [B, N]
    S, E, I, H, R, D = state.unbind(dim=-1)

    # ---- 0. 计算“参与接触的人口总数” ----
    # 这里 D 不参与接触，所以不计入 N_active
    N_active = (S + E + I + H + R).clamp(min=1.0)  # 防止除零

    # ---- 1. S -> E (感染) ----
    # force of infection: λ = β * I / N_active
    if allow_H_infectious:
        # 如果住院者也有一定传染性，可以改成 I+αH，这里先简单用 I+H 作示例
        lambda_inf = params.beta * (I + H) / N_active
    else:
        lambda_inf = params.beta * I / N_active

    # 单步从 S 变成 E 的概率：1 - exp(-λ dt)，逐格不同
    prob_SE = 1.0 - torch.exp(-lambda_inf * params.dt)
    prob_SE = prob_SE.clamp(0.0, 1.0)

    new_E_from_S = S * prob_SE   # [B,N]，本步从 S 转到 E 的人数

    # ---- 2. E -> I ----
    # 这里 σ 是常数，所以概率对所有格子一样
    prob_EI = 1.0 - math.exp(-params.sigma * params.dt)
    prob_EI = max(0.0, min(prob_EI, 1.0))

    new_I_from_E = E * prob_EI   # [B,N]

    # ---- 3. I -> R / H ----
    # 总离开率 = γ_I + η_I
    total_I_rate = params.gamma_I + params.eta_I
    prob_I_out = 1.0 - math.exp(-total_I_rate * params.dt)
    prob_I_out = max(0.0, min(prob_I_out, 1.0))

    new_I_out = I * prob_I_out   # 本步离开 I 的总人数
    new_H_from_I = new_I_out * params.p_H          # 离开 I 中去 H 的部分
    new_R_from_I = new_I_out * (1.0 - params.p_H)  # 离开 I 中直接恢复 R 的部分

    # ---- 4. H -> R / D ----
    total_H_rate = params.gamma_H + params.gamma_D
    prob_H_out = 1.0 - math.exp(-total_H_rate * params.dt)
    prob_H_out = max(0.0, min(prob_H_out, 1.0))

    new_H_out = H * prob_H_out
    new_D_from_H = new_H_out * params.p_D
    new_R_from_H = new_H_out * (1.0 - params.p_D)

    # ---- 5. 更新各个 compartment（确保不为负）----
    S_next = (S - new_E_from_S).clamp(min=0.0)
    E_next = (E + new_E_from_S - new_I_from_E).clamp(min=0.0)
    I_next = (I + new_I_from_E - new_R_from_I - new_H_from_I).clamp(min=0.0)
    H_next = (H + new_H_from_I - new_R_from_H - new_D_from_H).clamp(min=0.0)
    R_next = (R + new_R_from_I + new_R_from_H).clamp(min=0.0)
    D_next = (D + new_D_from_H).clamp(min=0.0)

    next_state = torch.stack(
        [S_next, E_next, I_next, H_next, R_next, D_next],
        dim=-1,
    ).to(device=device, dtype=dtype)

    return next_state


def mobility_step_torch(
    state: torch.Tensor,         # [B, N, C]，这里 C=6: (S,E,I,H,R,D)
    od_mat: torch.Tensor,        # [N, N] 或 [B, N, N] 的概率矩阵
    mobile_mask: torch.Tensor,   # [C] bool，哪些 compartment 随 OD 流动
    renormalize: bool = True,
) -> torch.Tensor:
    """
    人口流动步骤：根据 OD 概率矩阵把各个 compartment 在网格间迁移。

    参数:
        state: [B, N, C] 当前状态（C 通常是 6: S,E,I,H,R,D）
        od_mat:
            - [N, N]: 所有 batch 共享同一个 OD 矩阵
            - [B, N, N]: 每个 batch 有自己的 OD 矩阵
        mobile_mask: [C] bool，True 表示该 compartment 会参与流动
        renormalize: 是否对每个 OD 行做一次归一化（更稳妥）

    返回:
        moved_state: [B, N, C] 进行完流动后的状态
    """
    assert state.dim() == 3, "state 应为 [B, N, C] 张量"
    B, N, C = state.shape
    assert mobile_mask.shape[0] == C, "mobile_mask 长度必须等于状态维 C"

    device = state.device
    dtype = state.dtype

    # ---- 1. 准备 batched OD 矩阵 ----
    od_mat = od_mat.to(device=device, dtype=dtype)

    if od_mat.dim() == 2:
        # [N, N] -> 扩展成 [B, N, N]
        od = od_mat.unsqueeze(0).expand(B, -1, -1)   # [B, N, N]
    elif od_mat.dim() == 3:
        assert od_mat.shape[0] == B and od_mat.shape[1] == N and od_mat.shape[2] == N, \
            "od_mat 若为 3D，应为 [B, N, N]"
        od = od_mat
    else:
        raise ValueError("od_mat 维度必须是 2 或 3")

    # 可选：对每一行做一次归一化，避免数值误差导致行和 ≠ 1
    if renormalize:
        row_sum = od.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # [B, N, 1]
        od = od / row_sum

    # ---- 2. 对每个 compartment 做 batched 矩阵乘 ----
    moved = state.clone()

    # 将 C 维拆开，方便分别处理
    comps = state.unbind(dim=-1)  # 得到 C 个 [B, N] 张量

    moved_comps = []
    for c_idx, comp in enumerate(comps):
        if not bool(mobile_mask[c_idx]):
            # 不移动的 compartment（如 H、D），保持不变
            moved_comps.append(comp)
            continue

        # comp: [B, N] -> [B, 1, N]，与 [B, N, N] 的 od 做 batched matmul
        comp_expanded = comp.unsqueeze(1)        # [B, 1, N]
        moved_comp = torch.bmm(comp_expanded, od)  # [B, 1, N]
        moved_comp = moved_comp.squeeze(1)       # [B, N]

        moved_comps.append(moved_comp)

    # 把所有 compartment 堆回 [B, N, C]
    moved_state = torch.stack(moved_comps, dim=-1).to(device=device, dtype=dtype)

    return moved_state


def seirhd_step_with_mobility(
    state: torch.Tensor,               # [B, N, 6] 当前 S,E,I,H,R,D
    od_mat: torch.Tensor,              # [N, N] 或 [B, N, N] 当前时间步的 OD 概率矩阵
    params: "SEIRHDParams",
    mobile_mask: torch.Tensor,         # [6] bool，哪些 compartment 会流动
    action: Optional[torch.Tensor] = None,  # 可选：RL 动作，对 OD 做缩放控制
    allow_H_infectious: bool = False, # H 是否也具有传染性
) -> torch.Tensor:
    """
    单个时间步的完整更新：先按 OD 做人口流动，再做本地 SEIRHD 传播。

    参数:
        state: [B, N, 6]  当前状态 (S,E,I,H,R,D)
        od_mat:
            - [N, N]: 所有 batch 共享一个 OD 矩阵
            - [B, N, N]: 每个 batch 有自己的 OD
        params: SEIRHDParams 流行病学参数
        mobile_mask: [6] bool，True 表示该 compartment 会随 OD 流动
        action:
            - None：不做出行控制
            - [N, N] 或 [B, N, N]：0~1 之间的缩放因子，对 od_mat 逐元素相乘
        allow_H_infectious:
            - True: 住院者 H 也有传染性（在 λ=β(I+H)/N 中参与）
            - False: 只由 I 产生传染力

    返回:
        next_state: [B, N, 6]
    """
    device = state.device
    dtype = state.dtype
    B, N, C = state.shape
    assert C == 6, "state 的最后一维必须是 6（S,E,I,H,R,D）"
    assert mobile_mask.shape[0] == C

    # -------- 1. 构造 batched OD 矩阵 --------
    od_mat = od_mat.to(device=device, dtype=dtype)

    if od_mat.dim() == 2:
        # [N,N] -> [B,N,N]
        od = od_mat.unsqueeze(0).expand(B, -1, -1)
    elif od_mat.dim() == 3:
        assert od_mat.shape[0] == B and od_mat.shape[1] == N and od_mat.shape[2] == N
        od = od_mat
    else:
        raise ValueError("od_mat 必须是 [N,N] 或 [B,N,N] 形状")

    # -------- 2. 结合动作 action 调整 OD（出行干预）--------
    """
        关键约定（修复点）：
        - action 只作用于 OD 的“非对角线”(跨网格出行)；
        - 被缩减掉的出行概率补回到对角线（留在原地）；
        - 处理完后每一行仍为概率分布（行和=1），因此不再做会“抵消控制效果”的行归一化。
    """
    if action is not None:
        act = action.to(device=device, dtype=dtype)
        if act.dim() == 2:
            # [N,N] -> [B,N,N]
            act = act.unsqueeze(0).expand(B, -1, -1)
        elif act.dim() == 3:
            assert act.shape == od.shape, "action 为 3D 时，形状须等于 [B,N,N]"
        else:
            raise ValueError("action 必须是 [N,N] 或 [B,N,N]")
        
        # mask：对角线=1，非对角线=0（float）
        eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)  # [1,N,N]
        off = 1.0 - eye                                              # [1,N,N]

        # (1) 取出“出行部分”（非对角线）
        od_off = od * off  # 对角线变 0，只有跨区出行概率

        # (2) 只对“出行部分”施加控制（缩放）
        # 即使 act 是全矩阵，对角线也会被 off 抹掉
        od_off = od_off * act

        # (3) 把缩减掉的概率补回对角线（留在原地）
        off_sum = od_off.sum(dim=-1, keepdim=True)   # [B,N,1]
        diag = (1.0 - off_sum).clamp(min=0.0)        # [B,N,1]

        # 得到新 od：非对角=缩放后的出行；对角=补回后的留本地
        od = od_off + eye * diag

        # 理论上行和应为 1（数值误差很小），这里做一次轻量校正避免漂移
        row_sum2 = od.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        od = od / row_sum2
        
        # 逐元素缩放（假设动作值在 [0,1]，代表不同 OD 边的保留比例）
        # od = od * act
        
        # 手动调整对角线元素
        # for i in range(od.shape[0]):
        #     od[i, i] = 1 - od[i, :].sum() + od[i, i]

    # -------- 3. 人口流动（OD 搬家）--------
    # 用我们之前写好的流动函数
    state_after_move = mobility_step_torch(
        state=state,
        od_mat=od,               # 这里已经是 [B,N,N]
        mobile_mask=mobile_mask,
        renormalize=False,       # 已经手动归一化过
    )

    # -------- 4. 本地 SEIRHD 传播 --------
    next_state = seirhd_local_step(
        state_after_move,
        params=params,
        allow_H_infectious=allow_H_infectious,
    )

    return next_state

