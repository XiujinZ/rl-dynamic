# train.py
from __future__ import annotations
import sys
import os

# 把项目根目录加入 sys.path
PROJECT_ROOT = "/home/zxj/my_pytorch_project"
sys.path.append(PROJECT_ROOT)

import time
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from utils.utils_data_loader import load_od_matrices, load_residents

from env.env_SEIRHD import SEIRHDEpidemicEnv
from env.env_func_seirhd import SEIRHDParams

from agent.feature_builder import FeatureBuilder
from agent.encoder import MLPFlowGraphEncoder, EncoderConfig
from agent.actor import NodeGaussianActor, ActorConfig
from agent.critic import TwinCritic, CriticConfig
from agent.replay_buffer import ReplayBuffer, ReplayBufferConfig
from agent.sac_agent import SACAgent


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def node_action_to_od_matrix(a_node: torch.Tensor) -> torch.Tensor:
    """
    你的约定：
      A_ij = a_i (j != i)
      A_ii = 1

    输入:
      a_node: [B, N] in [0,1]
    输出:
      A: [B, N, N] in [0,1]
    """
    assert a_node.dim() == 2
    B, N = a_node.shape
    A = a_node.unsqueeze(-1).expand(B, N, N).clone()  # row-wise broadcast
    eye = torch.eye(N, device=a_node.device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
    A.masked_fill_(eye, 1.0)
    return A


def make_initial_infected(
    N: int,
    batch_size: int,
    num_infected_seeds: int = 10,
    device: str | torch.device = "cpu",
    fixed_indices: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """
    生成初始感染者张量 initial_I: [B, N]。

    - 每个感染种子在对应格子放 1 人（与你 baseline 一致）
    - fixed_indices 不为 None 时，使用固定的感染格子（更可复现）
    - fixed_indices 为 None 时，每次随机采样（更鲁棒）

    返回:
        initial_I: [B, N]
        infected_indices: list[int]
    """
    if fixed_indices is None:
        infected_indices = random.sample(range(N), num_infected_seeds)
    else:
        infected_indices = fixed_indices
        assert len(infected_indices) == num_infected_seeds, \
            "fixed_indices 的长度必须等于 num_infected_seeds"

    initial_I = torch.zeros(batch_size, N, dtype=torch.float32, device=device)
    for idx in infected_indices:
        initial_I[:, idx] = 1.0  # 如果你只想给 batch 的第 0 个样本感染，改成 initial_I[0, idx] = 1.0

    return initial_I, infected_indices


@dataclass
class TrainConfig:
    # general
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results/rl_runs/run1"

    # env / reward
    wtp: float = 260_000.0
    episode_steps: int = 240  # 一个 episode 跑多少步（建议 <= baseline 长度）

    # sac
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2
    auto_entropy: bool = True

    # training schedule
    total_env_steps: int = 100_000
    batch_size: int = 256
    start_steps: int = 5_000         # 前多少步随机探索填 buffer
    update_after: int = 5_000
    update_every: int = 1
    updates_per_step: int = 1

    # replay buffer
    replay_capacity: int = 200_000

    # logging / saving
    log_every: int = 500
    save_every: int = 10_000


def main(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # ======================================================
    # 1) 载入数据 OD / residents / epi params
    OD_DIR = "./data/OD_matrices"
    RESIDENT_PATH = "./data/residents.npy"

    od_mats = load_od_matrices(OD_DIR)
    residents = load_residents(RESIDENT_PATH)
    N = residents.shape[0]
    T_total = od_mats.shape[0]
    print(f"[INFO] 数据加载完毕：N={N}, T_total={T_total} 小时")

    epi_params = SEIRHDParams(
        beta=1.8/24,    # 传染率
        sigma=0.833/24,    # 潜伏期 → 感染的速率
        gamma_I=0.179/24,  # 感染者恢复率
        eta_I=0.169/24,   # 感染者住院率
        p_H=0.059,      # 感染者转住院概率
        gamma_H=0.128/24, # 住院者恢复率
        gamma_D=0.091/24, # 住院者死亡率
        p_D=0.139,      # 住院者死亡概率
        dt=1.0,       # 时间步（1小时）
    )
    
    # ======================================================
    # 2) 环境（初始化）
    # ======================================================
    env = SEIRHDEpidemicEnv(
        od_mats=od_mats,
        residents=residents,
        epi_params=epi_params,
        device=str(device),
        max_steps=T_total,
        wtp=cfg.wtp,
    )

    # ======================================================
    # 3) Feature / Encoder / Actor / Critic / Buffer / Agent
    # ======================================================
    feature_builder = FeatureBuilder(
        residents=torch.as_tensor(residents, dtype=torch.float32, device=device)
    )

    # 你需要能知道 feature_dim（F）
    # 约定：FeatureBuilder 有属性 feature_dim；如果你没写这个属性，
    # 就在 FeatureBuilder 里加一个 self.feature_dim = ...
    F = getattr(feature_builder, "feature_dim", None)
    if F is None:
        raise AttributeError("FeatureBuilder 需要提供 feature_dim 属性（节点特征维度 F）")

    enc_cfg = EncoderConfig(hidden_dim=64, num_layers=2, activation="relu")
    encoder = MLPFlowGraphEncoder(
        node_feat_dim=F,
        residents=torch.as_tensor(residents, dtype=torch.float32, device=device),
        cfg=enc_cfg,
    )

    actor_cfg = ActorConfig(hidden_dim=64)
    actor = NodeGaussianActor(node_dim=encoder.output_dim, cfg=actor_cfg)
    # 给 SACAgent 用（如果你 sac_agent 里用到 actor.action_dim）
    actor.action_dim = N

    critic_cfg = CriticConfig(hidden_dim=128, num_layers=2, aggregator="sum")
    critic = TwinCritic(node_dim=encoder.output_dim, cfg=critic_cfg)
    critic_target = TwinCritic(node_dim=encoder.output_dim, cfg=critic_cfg)

    rb_cfg = ReplayBufferConfig(capacity=cfg.replay_capacity, obs_dim=F)
    replay_buffer = ReplayBuffer(N=N, cfg=rb_cfg)

    # target_entropy：node action 一维/节点，总动作维度 = N
    target_entropy = -float(N)

    agent = SACAgent(
        encoder=encoder,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        replay_buffer=replay_buffer,
        device=str(device),
        gamma=cfg.gamma,
        tau=cfg.tau,
        lr=cfg.lr,
        alpha=cfg.alpha,
        auto_entropy=cfg.auto_entropy,
        target_entropy=target_entropy,
    )

    # ======================================================
    # 4) 训练主循环
    # ======================================================
    batch_size = 1
    initial_I, infected_indices = make_initial_infected(
        N=N,
        batch_size=batch_size,
        num_infected_seeds=10,
        device=device,
        fixed_indices= None,  
    )
    print(f"[INFO] 初始感染网格: {infected_indices}")

    obs_raw = env.reset(batch_size=batch_size, initial_infected=initial_I)
    
    B = obs_raw.shape[0]

    prev_obs_raw: Optional[torch.Tensor] = None
    prev_action_node: Optional[torch.Tensor] = None

    ep_step = 0
    episode_return = 0.0
    episode_idx = 0

    t0 = time.time()

    # 设置策略更改周期（1天）
    ACTION_INTERVAL = 24
    
    last_action = None
    
    for global_step in range(1, cfg.total_env_steps + 1):
        # ------- 取 flow（选动作前用 uncontrolled）-------
        flow = env.get_current_flow(controlled=False)  # [B,N,N]

        # ------- build features X_t -------
        X = feature_builder.build(
            obs_t=obs_raw,
            obs_tm1=prev_obs_raw,
            prev_action=prev_action_node,
        )  # [B,N,F]

        # ------- 选动作（action 只在固定间隔更新） -------
        if global_step % ACTION_INTERVAL == 1 or last_action is None:
            if global_step <= cfg.start_steps:
                action_node = torch.rand((B, N), device=device)
            else:
                with torch.no_grad():
                    H = agent.encoder(X, flow)
                    action_node, _ = agent.actor(H, deterministic=False)
            last_action = action_node
        else:
            action_node = last_action

        # ------- node action -> OD matrix action -------
        action_mat = node_action_to_od_matrix(action_node)  # [B,N,N]
        # env.step 支持 [N,N] 或 [B,N,N]，我们传 [B,N,N]
        next_obs_raw, reward, done, info = env.step(action=action_mat)

        # ------- next flow（对应 next_obs）-------
        next_flow = env.get_current_flow(controlled=False)

        # ------- build next features X_{t+1} (用于存 buffer) -------
        next_X = feature_builder.build(
            obs_t=next_obs_raw,
            obs_tm1=obs_raw,
            prev_action=action_node,
        )

        # ------- 存 replay：存 X / flow / node_action / reward / done / next_X / next_flow -------
        replay_buffer.add_batch(
            obs=X,
            flow=flow,
            action=action_node,
            reward=reward,
            done=done,
            next_obs=next_X,
            next_flow=next_flow,
        )

        # ------- 训练更新 -------
        if (global_step >= cfg.update_after) and (len(replay_buffer) >= cfg.batch_size):
            if global_step % cfg.update_every == 0:
                for _ in range(cfg.updates_per_step):
                    metrics = agent.update(cfg.batch_size)

        # ------- episode bookkeeping -------
        episode_return += float(reward.mean().item())
        ep_step += 1

        # ------- reset if done -------
        if bool(done.any().item()):
            episode_idx += 1
            if global_step % cfg.log_every != 0:
                print(f"[EP {episode_idx}] return={episode_return:.4f} steps={ep_step}")

            batch_size = 1
            initial_I, infected_indices = make_initial_infected(
                N=N,
                batch_size=batch_size,
                num_infected_seeds=10,
                device=device,
                fixed_indices= None, 
            )

            obs_raw = env.reset(batch_size=batch_size, initial_infected=initial_I)
            
            prev_obs_raw = None
            prev_action_node = None
            ep_step = 0
            episode_return = 0.0
            last_action = None
            continue

        # ------- advance prev pointers -------
        prev_obs_raw = obs_raw
        prev_action_node = action_node
        obs_raw = next_obs_raw

        # ------- logging -------
        if global_step % cfg.log_every == 0:
            elapsed = time.time() - t0
            msg = f"[step {global_step}] elapsed={elapsed:.1f}s buffer={len(replay_buffer)}"
            if "metrics" in locals():
                msg += f" critic={metrics['critic_loss']:.3f} actor={metrics['actor_loss']:.3f} alpha={metrics['alpha']:.4f}"
            print(msg)

        # ------- save -------
        if global_step % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"ckpt_{global_step}.pt")
            torch.save(
                {
                    "global_step": global_step,
                    "encoder": agent.encoder.state_dict(),
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "critic_target": agent.critic_target.state_dict(),
                    "log_alpha": agent.log_alpha.detach().cpu(),
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
