# agent/encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "relu"

    # 图聚合相关
    add_self_loop: bool = True     # 给 W 加对角，确保能看到自己
    row_normalize: bool = True     # 行归一化，让 W@X 更像加权平均
    eps: float = 1e-8


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLPFlowGraphEncoder(nn.Module):
    """
    使用 flow（人数流量矩阵）做邻居聚合的 MLP 图编码器。

    输入:
      - X:    [B, N, F]   节点特征
      - flow: [B, N, N]   人数流量 F_ij（从 i 到 j）
    输出:
      - H:    [B, N, h]   节点 embedding

    关键：
      - 边权用 W_ij = F_ij / N_i （按出发节点人口归一）
      - 可选：再做一次行归一化
    """

    def __init__(
        self,
        node_feat_dim: int,
        residents: torch.Tensor,           # [N]
        cfg: Optional[EncoderConfig] = None,
    ):
        super().__init__()
        assert residents.dim() == 1, "residents 必须是 [N]"
        self.register_buffer("residents", residents.float())
        self.cfg = cfg if cfg is not None else EncoderConfig()

        act = _act(self.cfg.activation)

        in_dim = 2 * node_feat_dim
        h = self.cfg.hidden_dim

        layers = []
        dims = [in_dim] + [h] * self.cfg.num_layers
        for i in range(self.cfg.num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
        self.mlp = nn.Sequential(*layers)

        self.output_dim = h

    def _build_W(self, flow: torch.Tensor, N: int) -> torch.Tensor:
        """
        flow: [B,N,N] 人数流量
        return W: [B,N,N] 归一化后的边权
        """
        cfg = self.cfg
        assert flow.dim() == 3 and flow.shape[1] == N and flow.shape[2] == N, \
            f"flow 必须是 [B,N,N]，got {tuple(flow.shape)}"

        B = flow.shape[0]
        device, dtype = flow.device, flow.dtype

        # 1) W_ij = F_ij / N_i
        Ni = self.residents[:N].to(device=device, dtype=dtype).clamp(min=cfg.eps)  # [N]
        W = flow / Ni.view(1, N, 1)  # [B,N,N]

        # 2) self-loop（保证每个节点保留自己的信息通道）
        if cfg.add_self_loop:
            eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)  # [1,N,N]
            W = W + eye

        # 3) 行归一化（把它变成“加权平均”的权重）
        if cfg.row_normalize:
            row_sum = W.sum(dim=-1, keepdim=True).clamp(min=cfg.eps)  # [B,N,1]
            W = W / row_sum

        return W

    def forward(self, X: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        X:    [B,N,F]
        flow: [B,N,N]
        """
        assert X.dim() == 3, "X 必须是 [B,N,F]"
        B, N, F = X.shape
        assert flow.shape[0] == B and flow.shape[1] == N and flow.shape[2] == N, \
            "flow 必须与 X 的 batch 和 N 对齐"

        # 构造边权矩阵 W
        W = self._build_W(flow=flow, N=N)  # [B,N,N]

        # 邻居聚合：nbr_i = Σ_j W_ij X_j
        nbr = torch.bmm(W, X)  # [B,N,F]

        # 拼接自身 + 邻居，再 MLP
        Z = torch.cat([X, nbr], dim=-1)    # [B,N,2F]
        Z = Z.view(B * N, 2 * F)           # [B*N,2F]
        H = self.mlp(Z).view(B, N, -1)     # [B,N,h]

        return H
