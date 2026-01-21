# agent/feature_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FeatureConfig:
    """
    控制 feature_builder 的配置。
    """
    use_log_pop_ratio: bool = True         # 是否加入 log(N_i / N_total)
    include_prev_action: bool = True       # 是否加入 prev_action_i
    use_delta_E: bool = True               # 是否加入 ΔE_i
    clamp_delta_E_nonneg: bool = True      # ΔE 是否截断为非负（可选）
    eps: float = 1e-8                      # 防止除零


class FeatureBuilder(nn.Module):
    """
    将环境的 raw state(obs) 构造成 per-node 的特征向量 X。

    输入:
      - obs_t:   [B, N, 6]  (S,E,I,H,R,D)  人数
      - obs_tm1: [B, N, 6]  (可选) 上一时刻人数，用于计算 ΔE
      - prev_action: [B, N] (可选) 上一时刻节点动作
      - residents: [N]      每个节点人口 N_i（常驻人口）
    输出:
      - X: [B, N, F]   per-node 特征
    """

    def __init__(self, residents: torch.Tensor, cfg: Optional[FeatureConfig] = None):
        super().__init__()
        assert residents.dim() == 1, "residents 必须是 [N]"

        self.register_buffer("residents", residents.float())  # [N]，随模型搬到同 device
        self.cfg = cfg if cfg is not None else FeatureConfig()

        # 预先计算 N_total（全局总人口），用于 log(N_i / N_total)
        N_total = self.residents.sum().clamp(min=self.cfg.eps)
        self.register_buffer("N_total", N_total)

    @property
    def feature_dim(self) -> int:
        """
        返回输出特征维度 F（方便后续 encoder/policy 初始化）。
        固定顺序：
          0-5: S/N, E/N, I/N, H/N, R/N, D/N
          + ΔE
          + prev_action
          + log_pop_ratio
        """
        dim = 6
        if self.cfg.use_delta_E:
            dim += 1
        if self.cfg.include_prev_action:
            dim += 1
        if self.cfg.use_log_pop_ratio:
            dim += 1
        return dim

    def forward(
        self,
        obs_t: torch.Tensor,
        obs_tm1: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        obs_t: [B, N, 6]
        obs_tm1: [B, N, 6] or None
        prev_action: [B, N] or None
        """
        assert obs_t.dim() == 3 and obs_t.size(-1) == 6, "obs_t 必须是 [B, N, 6]"
        B, N, C = obs_t.shape
        assert C == 6

        # residents: [N] -> [1, N, 1]，用于对每个 compartment 做比例化
        Ni = self.residents[:N].view(1, N, 1).to(dtype=obs_t.dtype, device=obs_t.device)
        Ni_safe = Ni.clamp(min=self.cfg.eps)

        # ---------- 1) 基础比例特征：S/N, E/N, I/N, H/N, R/N, D/N ----------
        frac = obs_t / Ni_safe  # [B, N, 6]
        feats = [frac]          # 先放入 6 维

        # ---------- 2) ΔE 特征（绝对量，不除 N_i）----------
        if self.cfg.use_delta_E:
            if obs_tm1 is None:
                # 第 0 步没有上一帧时，ΔE 置 0
                delta_E = torch.zeros((B, N), dtype=obs_t.dtype, device=obs_t.device)
            else:
                assert obs_tm1.shape == obs_t.shape, "obs_tm1 形状必须与 obs_t 相同"
                # E 在 index=1（S=0, E=1, I=2, H=3, R=4, D=5）
                delta_E = obs_t[..., 1] - obs_tm1[..., 1]  # [B, N]

            if self.cfg.clamp_delta_E_nonneg:
                # 可选：只保留新增暴露，忽略减少（更贴近“新增感染压力”）
                delta_E = delta_E.clamp(min=0.0)

            feats.append(delta_E.unsqueeze(-1))  # [B, N, 1]

        # ---------- 3) prev_action 特征 ----------
        if self.cfg.include_prev_action:
            if prev_action is None:
                prev_action_feat = torch.zeros((B, N), dtype=obs_t.dtype, device=obs_t.device)
            else:
                assert prev_action.shape == (B, N), "prev_action 必须是 [B, N]"
                prev_action_feat = prev_action.to(dtype=obs_t.dtype, device=obs_t.device)
            feats.append(prev_action_feat.unsqueeze(-1))  # [B, N, 1]

        # ---------- 4) log(N_i / N_total) 规模特征 ----------
        if self.cfg.use_log_pop_ratio:
            # log_pop_ratio: [N] -> [1, N] -> [B, N]
            log_pop_ratio = torch.log(self.residents[:N] / self.N_total).to(
                dtype=obs_t.dtype, device=obs_t.device
            )  # [N]
            log_pop_ratio = log_pop_ratio.view(1, N).expand(B, N)  # [B, N]
            feats.append(log_pop_ratio.unsqueeze(-1))  # [B, N, 1]

        # 拼接成 [B, N, F]
        X = torch.cat(feats, dim=-1)

        return X
