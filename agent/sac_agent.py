# agent/sac_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class SACAgent:
    def __init__(
        self,
        encoder,
        actor,
        critic,
        critic_target,
        replay_buffer,
        device="cuda",
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        auto_entropy=True,
        target_entropy=None,
    ):
        self.device = torch.device(device)

        self.encoder = encoder.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = critic_target.to(self.device)

        self.replay_buffer = replay_buffer

        # -------- SAC hyperparameters --------
        self.gamma = gamma
        self.tau = tau

        # -------- entropy --------
        self.auto_entropy = auto_entropy
        self.log_alpha = torch.tensor(
            [torch.log(torch.tensor(alpha))],
            requires_grad=True,
            device=self.device,
        )

        if target_entropy is None:
            # heuristic: -|A|
            self.target_entropy = -actor.action_dim
        else:
            self.target_entropy = target_entropy

        # -------- optimizers --------
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # init target
        self.critic_target.load_state_dict(self.critic.state_dict())

    # ======================================================
    # 1. 选择动作（训练 / 测试）
    # ======================================================

    @torch.no_grad()
    def act(self, obs, flow, deterministic=False):
        """
        obs: [B,N,6]
        flow: [B,N,N]
        """
        self.encoder.eval()
        self.actor.eval()

        h = self.encoder(obs, flow)
        action, _ = self.actor(h, deterministic=deterministic)

        return action

    # ======================================================
    # 2. SAC 更新主函数
    # ======================================================

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size, self.device)

        obs = batch["obs"]
        flow = batch["flow"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]
        next_obs = batch["next_obs"]
        next_flow = batch["next_flow"]

        # ========== Encode ==========
        h = self.encoder(obs, flow)
        with torch.no_grad():
            h_next = self.encoder(next_obs, next_flow)

        # ========== Critic Update ==========
        with torch.no_grad():
            next_action, next_logp = self.actor(h_next)
            q1_t, q2_t = self.critic_target(h_next, next_action)
            q_target = torch.min(q1_t, q2_t)
            v_target = q_target - self.alpha * next_logp
            y = reward + self.gamma * (1 - done) * v_target

        q1, q2 = self.critic(h, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ========== Actor Update ==========
        new_action, logp = self.actor(h)
        q1_pi, q2_pi = self.critic(h, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ========== Alpha Update ==========
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ========== Soft Update ==========
        self.soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }

    # ======================================================
    # 3. Utils
    # ======================================================

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def soft_update(self, net, target):
        for p, p_targ in zip(net.parameters(), target.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)
