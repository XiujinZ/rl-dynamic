"""
policy/base.py

Policy 抽象基类。
所有策略（手工 / RL）都必须实现同一个接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch


class Policy(ABC):
    """
    Policy 抽象基类

    约定：
    - 输入 obs: torch.Tensor
    - 输出 action:
        * None                -> 不干预
        * torch.Tensor        -> OD 缩放矩阵，取值范围 [0,1]
    """

    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        t: int,
        info: Optional[Dict[str, Any]] = None
    ) -> Optional[torch.Tensor]:
        """
        根据当前观测 obs 和时间步 t，生成动作 action。

        参数:
            obs: torch.Tensor
                当前观测，形状通常为 [B, N, 6]

            t: int
                当前时间步（可选使用）

            info: dict 或 None
                额外信息（可选）

        返回:
            action:
                - None：不进行流动控制
                - Tensor：[N,N] 或 [B,N,N]，元素 ∈ [0,1]
        """
        raise NotImplementedError
