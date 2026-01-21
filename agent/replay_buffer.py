# agent/replay_buffer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ReplayBufferConfig:
    capacity: int = 200_000
    obs_dim: int = 6          # (S,E,I,H,R,D)
    dtype: torch.dtype = torch.float32


class ReplayBuffer:
    """
    Replay buffer for SAC (off-policy).

    Stores transitions:
      (obs, flow, action, reward, done, next_obs, next_flow)

    Shapes:
      obs:       [B, N, 6]
      flow:      [B, N, N]
      action:    [B, N]        (node-wise action, in [0,1])
      reward:    [B]
      done:      [B]           (bool or float)
      next_obs:  [B, N, 6]
      next_flow: [B, N, N]
    """

    def __init__(self, N: int, cfg: Optional[ReplayBufferConfig] = None):
        self.cfg = cfg if cfg is not None else ReplayBufferConfig()
        self.N = int(N)
        C = self.cfg.capacity
        dt = self.cfg.dtype

        # Pre-allocate on CPU for memory efficiency
        self.obs = torch.zeros((C, self.N, self.cfg.obs_dim), dtype=dt, device="cpu")
        self.flow = torch.zeros((C, self.N, self.N), dtype=dt, device="cpu")
        self.action = torch.zeros((C, self.N), dtype=dt, device="cpu")

        self.reward = torch.zeros((C,), dtype=dt, device="cpu")
        self.done = torch.zeros((C,), dtype=dt, device="cpu")  # store as 0/1 float

        self.next_obs = torch.zeros((C, self.N, self.cfg.obs_dim), dtype=dt, device="cpu")
        self.next_flow = torch.zeros((C, self.N, self.N), dtype=dt, device="cpu")

        self.ptr = 0
        self.size = 0
        self.capacity = C

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,        # [B,N,6]
        flow: torch.Tensor,       # [B,N,N]
        action: torch.Tensor,     # [B,N]
        reward: torch.Tensor,     # [B]
        done: torch.Tensor,       # [B] bool/float
        next_obs: torch.Tensor,   # [B,N,6]
        next_flow: torch.Tensor,  # [B,N,N]
    ) -> None:
        """
        Add a batch of transitions. Each batch element is stored as one transition.
        """
        assert obs.dim() == 3 and obs.shape[1] == self.N and obs.shape[2] == self.cfg.obs_dim
        assert flow.dim() == 3 and flow.shape[1] == self.N and flow.shape[2] == self.N
        assert action.dim() == 2 and action.shape[1] == self.N
        B = obs.shape[0]
        assert flow.shape[0] == B and action.shape[0] == B
        assert next_obs.shape == obs.shape
        assert next_flow.shape == flow.shape
        assert reward.shape[0] == B
        assert done.shape[0] == B

        # Move to CPU & correct dtype
        obs = obs.detach().to("cpu", dtype=self.cfg.dtype)
        flow = flow.detach().to("cpu", dtype=self.cfg.dtype)
        action = action.detach().to("cpu", dtype=self.cfg.dtype)
        reward = reward.detach().to("cpu", dtype=self.cfg.dtype)

        # done as float(0/1)
        if done.dtype == torch.bool:
            done_f = done.to("cpu", dtype=self.cfg.dtype)
        else:
            done_f = done.detach().to("cpu", dtype=self.cfg.dtype).clamp(0.0, 1.0)

        next_obs = next_obs.detach().to("cpu", dtype=self.cfg.dtype)
        next_flow = next_flow.detach().to("cpu", dtype=self.cfg.dtype)

        # Write one by one (simple & robust). If you need speed later we can vectorize.
        for i in range(B):
            idx = self.ptr

            self.obs[idx].copy_(obs[i])
            self.flow[idx].copy_(flow[i])
            self.action[idx].copy_(action[i])

            self.reward[idx].copy_(reward[i])
            self.done[idx].copy_(done_f[i])

            self.next_obs[idx].copy_(next_obs[i])
            self.next_flow[idx].copy_(next_flow[i])

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device | str) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.

        Returns a dict of tensors on `device`:
          obs: [B,N,6]
          flow: [B,N,N]
          action: [B,N]
          reward: [B]
          done: [B]   (0/1 float)
          next_obs: [B,N,6]
          next_flow: [B,N,N]
        """
        assert self.size > 0, "ReplayBuffer is empty."
        B = int(batch_size)
        idx = torch.randint(0, self.size, (B,), device="cpu")

        batch = {
            "obs": self.obs[idx].to(device),
            "flow": self.flow[idx].to(device),
            "action": self.action[idx].to(device),
            "reward": self.reward[idx].to(device),
            "done": self.done[idx].to(device),
            "next_obs": self.next_obs[idx].to(device),
            "next_flow": self.next_flow[idx].to(device),
        }
        return batch
