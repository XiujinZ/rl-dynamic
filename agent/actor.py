# agent/actor.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ActorConfig:
    hidden_dim: int = 64
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    activation: str = "relu"


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class NodeGaussianActor(nn.Module):
    """
    Node-wise Gaussian policy for SAC.

    输入:
        H: [B, N, h]   (encoder 输出)

    输出:
        action:    [B, N]    ∈ (0,1)
        log_prob:  [B]       （整张图的 log_prob）
    """

    def __init__(self, node_dim: int, cfg: ActorConfig):
        super().__init__()
        self.cfg = cfg

        act = _act(cfg.activation)

        self.net = nn.Sequential(
            nn.Linear(node_dim, cfg.hidden_dim),
            act,
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            act,
        )

        self.mu_head = nn.Linear(cfg.hidden_dim, 1)
        self.log_std_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self,
        H: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [B, N, h]

        return:
            action:   [B, N]   in (0,1)
            log_prob: [B]      (sum over nodes)
        """
        B, N, _ = H.shape

        x = self.net(H)                     # [B,N,h]
        mu = self.mu_head(x)               # [B,N,1]
        log_std = self.log_std_head(x)     # [B,N,1]

        log_std = torch.clamp(
            log_std,
            self.cfg.log_std_min,
            self.cfg.log_std_max
        )
        std = torch.exp(log_std)

        # --------- 采样 ----------
        if deterministic:
            z = mu
        else:
            eps = torch.randn_like(mu)
            z = mu + eps * std

        # tanh squash
        a_tanh = torch.tanh(z)             # (-1,1)
        action = (a_tanh + 1.0) / 2.0      # → (0,1)

        # --------- log prob ----------
        log_prob = None
        if with_logprob:
            # Gaussian log prob
            logp = -0.5 * (
                ((z - mu) / (std + 1e-8)) ** 2
                + 2 * log_std
                + torch.log(torch.tensor(2 * torch.pi))
            )

            # tanh 修正项
            logp -= torch.log(1 - a_tanh.pow(2) + 1e-6)

            # 对每个节点求和 → [B]
            logp = logp.sum(dim=-1).sum(dim=-1)

            log_prob = logp

        return action.squeeze(-1), log_prob
