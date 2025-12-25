"""
policy/rl_policy.py

RL Policy

将 RL actor 网络的输出（节点级动作）
映射为 OD 层面的流动控制矩阵。
"""

import torch
from .base import Policy


class RLPolicy(Policy):
    """
    RL 策略包装器

    actor 网络输出：
        node_action ∈ [0,1]^{B×N}

    OD 动作构造规则：
        action_ij = f(node_i, node_j)
    """

    def __init__(
        self,
        actor,
        num_nodes: int,
        edge_mapping: str = "mul"
    ):
        """
        参数:
            actor:
                RL actor 网络（如 SAC 的 policy 网络）
                输入 obs，输出 node_action ∈ [0,1]^{B×N}

            num_nodes: int
                区域/节点数量 N

            edge_mapping: str
                节点动作到 OD 动作的映射方式
                - "mul":   a_ij = a_i * a_j
                - "min":   a_ij = min(a_i, a_j)
        """
        self.actor = actor
        self.N = num_nodes
        self.edge_mapping = edge_mapping

    def act(self, obs, t, info=None):
        """
        使用 actor 网络生成动作
        """
        with torch.no_grad():
            # actor 输出节点级动作 [B,N]
            node_action = self.actor(obs)  # 你后续保证输出在 [0,1]

        # 目前默认 batch_size = 1
        a = node_action[0]  # [N]

        if self.edge_mapping == "mul":
            action = a.unsqueeze(1) * a.unsqueeze(0)
        elif self.edge_mapping == "min":
            action = torch.minimum(
                a.unsqueeze(1),
                a.unsqueeze(0)
            )
        else:
            raise ValueError(f"未知 edge_mapping: {self.edge_mapping}")

        return action.clamp(0.0, 1.0)
