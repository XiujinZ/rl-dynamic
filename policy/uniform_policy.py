"""
policy/uniform_policy.py

全局统一封控策略：
对所有区域间流动施加相同的缩放比例。
"""

import torch
from .base import Policy


class UniformControlPolicy(Policy):
    """
    全局统一封控策略

    action_ij = alpha, 对所有 OD 边一致
    """

    def __init__(self, num_nodes: int, alpha: float):
        """
        参数:
            num_nodes: int
                区域/网格数量 N

            alpha: float
                流动保留比例 ∈ [0,1]
                - alpha = 1.0 -> 无封控
                - alpha = 0.0 -> 完全封控
        """
        assert 0.0 <= alpha <= 1.0, "alpha 必须在 [0,1] 内"
        self.N = num_nodes
        self.alpha = float(alpha)

    def act(self, obs, t, info=None):
        """
        返回统一缩放的 OD action 矩阵 [N,N]
        """
        device = obs.device
        dtype = obs.dtype

        action = torch.full(
            (self.N, self.N),
            self.alpha,
            device=device,
            dtype=dtype
        )
        return action
