# agent/critic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CriticConfig:
    hidden_dim: int = 128          # MLP hidden
    num_layers: int = 2            # MLP depth (>=2 is common)
    activation: str = "relu"       # relu/gelu/silu
    aggregator: str = "sum"        # "sum" or "mean"
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


class NodeQNetwork(nn.Module):
    """
    单个 Q 网络：输入 (H, action) 输出 Q_total。

    输入:
      - H:      [B, N, h]     node embeddings (encoder output)
      - action: [B, N] or [B, N, 1]   node-wise action

    输出:
      - Q: [B]  每个 batch 一个标量 Q 值
    """

    def __init__(self, node_dim: int, cfg: Optional[CriticConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else CriticConfig()
        act = _act(self.cfg.activation)

        # per-node input is concat([h_i, a_i]) => dim = node_dim + 1
        in_dim = node_dim + 1
        hidden = self.cfg.hidden_dim

        layers = []
        d = in_dim
        for _ in range(self.cfg.num_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(act)
            d = hidden
        layers.append(nn.Linear(d, 1))  # per-node q_i
        self.mlp = nn.Sequential(*layers)

        if self.cfg.aggregator not in ("sum", "mean"):
            raise ValueError(f"aggregator must be 'sum' or 'mean', got {self.cfg.aggregator}")

    def forward(self, H: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert H.dim() == 3, "H 必须是 [B,N,h]"
        B, N, h = H.shape

        if action.dim() == 2:
            assert action.shape == (B, N), f"action 应为 [B,N], got {tuple(action.shape)}"
            a = action.unsqueeze(-1)  # [B,N,1]
        elif action.dim() == 3:
            assert action.shape == (B, N, 1), f"action 应为 [B,N,1], got {tuple(action.shape)}"
            a = action
        else:
            raise ValueError("action 必须是 [B,N] 或 [B,N,1]")

        # concat per-node
        x = torch.cat([H, a], dim=-1)        # [B,N,h+1]
        x = x.view(B * N, h + 1)             # [B*N,h+1]
        q_node = self.mlp(x).view(B, N, 1)   # [B,N,1]

        # aggregate to global Q
        if self.cfg.aggregator == "sum":
            Q = q_node.sum(dim=1)            # [B,1]
        else:
            Q = q_node.mean(dim=1)           # [B,1]

        return Q.squeeze(-1)                 # [B]


class TwinCritic(nn.Module):
    """
    SAC 标准 Twin Critic：两个独立的 Q 网络 Q1, Q2

    forward 返回:
      Q1: [B]
      Q2: [B]
    """

    def __init__(self, node_dim: int, cfg: Optional[CriticConfig] = None):
        super().__init__()
        self.q1 = NodeQNetwork(node_dim=node_dim, cfg=cfg)
        self.q2 = NodeQNetwork(node_dim=node_dim, cfg=cfg)

    def forward(self, H: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(H, action), self.q2(H, action)
