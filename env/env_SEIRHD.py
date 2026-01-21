# env_SEIRHD.py
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch import Tensor

from env.env_func_seirhd import (
    SEIRHDParams,
    init_state_from_residents,
    seirhd_step_with_mobility,
)

from reward.reward_func import compute_reward

class SEIRHDEpidemicEnv:
    """
    类似 Gym 的 SEIRHD + OD 流动 环境。

    - 内部状态 state: [B, N, 6]  (S, E, I, H, R, D)
    - 动作 action: 用来缩放 OD 矩阵的 [N, N] 或 [B, N, N]（例如 0~1 的限流系数）
    - 观测 obs: 目前直接使用 state 本身（[B, N, 6]），之后你可以改成别的形式
    """

    def __init__(
        self,
        od_mats: np.ndarray,           # [T_total, N, N] 的 OD 概率矩阵序列
        residents: np.ndarray,         # [N] 各网格常驻人口数
        epi_params: SEIRHDParams,      # SEIRHD 参数
        device: str = "cpu",
        mobile_compartments=("S", "E", "I", "R"),  # 默认 S,E,I,R 流动；H,D 不流动
        max_steps: Optional[int] = None,           # 最长仿真步数；默认用 T_total
        wtp: float = 1.0, 
    ):
        """
        od_mats:     np.array [T_total, N, N]，每个时间步的移动概率矩阵
        residents:   np.array [N]，各网格常驻人口数，用作初始 S
        epi_params:  模型参数（beta, sigma, gamma_I, eta_I, p_H, gamma_H, gamma_D, p_D, dt）
        device:      "cpu" 或 "cuda"
        mobile_compartments: 哪些 compartment 会随着 OD 流动
        max_steps:   最多仿真多少步（None 表示用 od_mats 的长度）
        """
        self.device = torch.device(device)
        self.params = epi_params
        
        self.wtp = wtp

        # ====== 1. 处理 OD 矩阵 ======
        od_mats_t = torch.as_tensor(od_mats, dtype=torch.float32, device=self.device)
        assert od_mats_t.dim() == 3, "od_mats 必须是 [T_total, N, N]"
        self.T_total, self.N, _ = od_mats_t.shape
        self.od_mats = od_mats_t  # [T_total, N, N]

        # ====== 2. 处理居民数 ======
        residents_t = torch.as_tensor(residents, dtype=torch.float32, device=self.device)
        assert residents_t.shape[0] == self.N, "residents 长度必须等于 N"
        self.residents = residents_t  # [N]

        # ====== 3. 哪些 compartment 流动 ======
        comp_names = ["S", "E", "I", "H", "R", "D"]
        mobile_mask = torch.zeros(6, dtype=torch.bool, device=self.device)
        for name in mobile_compartments:
            assert name in comp_names, f"未知 compartment 名称: {name}"
            idx = comp_names.index(name)
            mobile_mask[idx] = True
        self.mobile_mask = mobile_mask  # [6] bool

        # ====== 4. 时间与仿真控制 ======
        self.max_steps = max_steps if max_steps is not None else self.T_total

        # 运行时状态
        self.state: Optional[Tensor] = None   # [B, N, 6]
        self.batch_size: int = 1
        self.t: int = 0  # 当前时间步

    # ============================
    # 内部工具函数
    # ============================

    def _get_current_od(self) -> Tensor:
        """
        返回当前时间步对应的 OD 矩阵 [N, N]。

        这里用 (t % T_total) 周期性循环 OD。
        也可以改成：t >= T_total 时终止，而不是循环。
        """
        idx = self.t % self.T_total
        return self.od_mats[idx]  # [N, N]

    def _get_observation(self) -> Tensor:
        """
        返回给 RL agent 的观测。

        暂时直接用内部 state （[B, N, 6]），
        后续可以自己定义：例如只给 I,H，或者拼接 GNN 特征等。
        """
        return self.state

    # ============================
    # Gym 风格接口：reset / step
    # ============================

    def reset(
        self,
        batch_size: int = 1,
        initial_infected: Optional[Tensor] = None,  # [B, N] 或 None
    ) -> Tensor:
        """
        重置环境，初始化 state，并返回初始观测 obs。

        默认:
            S_0(i) = residents[i]
            E,I,H,R,D = 0
        若提供 initial_infected，则从 S 扣对应人数到 I。
        """
        self.batch_size = batch_size
        self.t = 0

        if initial_infected is not None:
            # 确保形状正确
            assert initial_infected.shape == (batch_size, self.N), \
                f"initial_infected 形状应为 [B, N] = [{batch_size}, {self.N}]"
            initial_infected_t = initial_infected.to(self.device, dtype=torch.float32)
        else:
            initial_infected_t = None

        # [B,N,6]
        state0 = init_state_from_residents(
            residents=self.residents,
            batch_size=batch_size,
            initial_infected=initial_infected_t,
        )
        self.state = state0.to(self.device)

        return self._get_observation()

    def step(
        self,
        action: Optional[Tensor] = None,  # [N,N] 或 [B,N,N]，用于缩放 OD（限流措施）
        compute_reward: bool = True, 
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        """
        推进一个时间步：

        1. 取出当前 OD 矩阵
        2. 调用 seirhd_step_with_mobility 做流动 + 本地 SEIRHD
        3. 更新时间步 t
        4. 计算 reward, done, info
        5. 返回 (obs, reward, done, info)
        """
        assert self.state is not None, "请先调用 reset() 再 step()."

        # 0. 保存旧状态
        prev_state = self.state.clone()
        
        # 1. 当前时间步的 OD
        od_t = self._get_current_od()  # [N, N]

        # 2. 流动 + 本地传播（环境推进）
        next_state = seirhd_step_with_mobility(
            state=self.state,
            od_mat=od_t,
            params=self.params,
            mobile_mask=self.mobile_mask,
            action=action,
            allow_H_infectious=False,  # 之后可以改成 True 做实验
        )

        self.state = next_state
        self.t += 1

        # 3. 计算reward
        if compute_reward:
            reward = compute_reward(
                prev_state=prev_state,
                curr_state=self.state,
                action=action,
                t=self.t - 1,
                wtp=self.wtp,
            )
        else:
            reward = torch.zeros((self.batch_size,), device=self.device)
        
        # 4. done
        done = torch.full(
            (self.batch_size,),
            self.t >= self.max_steps,
            dtype=torch.bool,
            device=self.device,
        )
        
        # 5. info
        info = {
            "t": self.t,
            "reward": reward.detach().cpu().numpy(),
        }

        return self.state, reward, done, info
    
        # obs = self._get_observation()
        # reward, done, info = self._compute_reward_and_done()
        # return obs, reward, done, info


    # ============================
    # （新增）OD -> Flow 相关工具
    # ============================

    def _normalize_od(self, od: Tensor, eps: float = 1e-9) -> Tensor:
        """
        对 OD 做行归一化，保证每行和为 1（数值稳定）。
        od: [N,N]
        """
        row_sum = od.sum(dim=-1, keepdim=True).clamp(min=eps)
        return od / row_sum

    def _apply_action_to_od(self, od: Tensor, action: Optional[Tensor]) -> Tensor:
        """
        把 action 施加到 OD 上，逻辑与 seirhd_step_with_mobility 保持一致：
        - action 只作用于非对角线（跨区出行）
        - 缩减掉的概率补回对角线（留在原地）
        - 最后轻量行归一化

        输入：
          od:     [N,N]（假设已经行归一化过）
          action: [N,N] or None（0~1）
        输出：
          od_new: [N,N]
        """
        if action is None:
            return od

        N = od.shape[0]
        act = action.to(device=od.device, dtype=od.dtype)

        if act.dim() != 2 or act.shape != (N, N):
            raise ValueError(f"action 必须是 [N,N]，当前是 {tuple(act.shape)}")

        act = act.clamp(0.0, 1.0)

        eye = torch.eye(N, device=od.device, dtype=od.dtype)  # [N,N]
        off = 1.0 - eye

        # 1) 只对非对角线施加控制
        od_off = od * off
        od_off = od_off * act  # act 的对角线即使有值也会被 off 抹掉

        # 2) 补回对角线，使每行和为 1
        off_sum = od_off.sum(dim=-1, keepdim=True)  # [N,1]
        diag = (1.0 - off_sum).clamp(min=0.0)       # [N,1]
        od_new = od_off + eye * diag                # [N,N]

        # 3) 轻量归一化，防数值漂移
        od_new = self._normalize_od(od_new)

        return od_new

    def _compute_outflow(self, state: Tensor) -> Tensor:
        """
        计算每个节点本步会流动的人数 outflow_i。
        state: [B,N,6]
        return: outflow: [B,N]
        """
        # mobile_mask: [6] bool
        # 把会流动的 comp 加总
        # state[..., mask] -> [B,N,num_mobile] 然后 sum
        mobile = state[..., self.mobile_mask]  # [B,N,K]
        outflow = mobile.sum(dim=-1)           # [B,N]
        return outflow

    def get_current_flow(
        self,
        action: Optional[Tensor] = None,
        controlled: bool = False,
    ) -> Tensor:
        """
        给 encoder 用的 flow 获取接口。

        - 默认 controlled=False：返回当前时刻的“原始 OD”对应的 flow（选 action 前最常用）
        - 如果 controlled=True：会把传入的 action 作用到 OD 后，再计算 flow（可选研究设置）

        返回：
          flow: [B,N,N]  (人数流量矩阵)
        """
        assert self.state is not None, "请先 reset() 再调用 get_current_flow()"

        B, N, _ = self.state.shape

        # 1) 当前时刻 OD 概率
        od = self._get_current_od()          # [N,N]
        od = self._normalize_od(od)          # 行归一

        # 2) 是否使用控制后的 OD
        if controlled:
            od = self._apply_action_to_od(od, action)

        # 3) 计算 outflow（会流动的总人数）
        outflow = self._compute_outflow(self.state)  # [B,N]

        # 4) flow_ij = outflow_i * od_ij
        # od: [N,N] -> [1,N,N] -> [B,N,N]
        flow = outflow.unsqueeze(-1) * od.unsqueeze(0).expand(B, -1, -1)

        return flow


    # def _compute_reward_and_done(self) -> Tuple[Tensor, Tensor, Dict]:
    #     """
    #     根据当前 state 计算 reward 与 done。

    #     当前是一个占位版本：（已废弃）
    #     - reward: 惩罚 I + H（感染者 + 住院者越多越差）
    #     - done:
    #         * 达到 max_steps
    #     """
    #     assert self.state is not None, "state 为空，请先 reset()。"

    #     S, E, I, H, R, D = self.state.unbind(dim=-1)  # 每个都是 [B, N]

    #     total_I = I.sum(dim=1)  # [B]
    #     total_H = H.sum(dim=1)  # [B]
    #     total_D = D.sum(dim=1)  # [B]

    #     # 简单 reward：感染+住院越多越差
    #     reward = -(total_I + total_H)  # [B]

    #     # 终止条件 1：时间步到头
    #     done_time = torch.full(
    #         (self.batch_size,),
    #         self.t >= self.max_steps,
    #         dtype=torch.bool,
    #         device=self.device,
    #     )  # [B]

    #     # 终止条件 2：疫情几乎结束（感染+住院≈0）（已删除）
    #     # done_epi = (total_I + total_H < 1.0)  # [B] bool

    #     done = done_time  # [B] bool

    #     info = {
    #         "total_S": S.sum(dim=1).detach().cpu().numpy(),
    #         "total_E": E.sum(dim=1).detach().cpu().numpy(),
    #         "total_I": total_I.detach().cpu().numpy(),
    #         "total_H": total_H.detach().cpu().numpy(),
    #         "total_R": R.sum(dim=1).detach().cpu().numpy(),
    #         "total_D": total_D.detach().cpu().numpy(),
    #         "t": self.t,
    #     }

    #     return reward, done, info