"""DreamerV3-lite training loop: collect → learn world model → imagine → learn actor-critic."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dreamer.env import DreamerVecEnv
from dreamer.models import Dreamer, DreamerConfig
from dreamer.replay import VecSequenceReplay

logger = logging.getLogger(__name__)


def _random_actions(envs: DreamerVecEnv) -> np.ndarray:
    """Uniform random policy-space actions, shape (N, action_dim)."""
    n, a = envs.num_envs, envs.action_dim
    if envs.is_discrete:
        actions = np.zeros((n, a), dtype=np.float32)
        actions[np.arange(n), np.random.randint(a, size=n)] = 1.0
        return actions
    return np.random.uniform(-1.0, 1.0, size=(n, a)).astype(np.float32)


def train_dreamer(
    env_name: str = "CarRacing-v3",
    device: torch.device = torch.device("cpu"),
    total_steps: int = 100_000,
    prefill: int = 2000,
    train_every: int = 5,
    seq_len: int = 50,
    batch_size: int = 16,
    action_repeat: int = 2,
    horizon: int = 15,
    deter_dim: int = 256,
    cnn_depth: int = 32,
    entropy_scale: float = 1e-3,
    replay_capacity: int = 200_000,
    domain_randomize: bool = False,
    num_envs: int = 1,
    seed: int = 0,
    save_path: Path = Path("./saved_models/"),
    save_every: int = 10_000,
    log_every: int = 1000,
    eval_every: int = 10_000,
    use_tensorboard: bool = False,
    load_path: str = "",
) -> Dreamer:
    """Train a Dreamer agent on ``env_name``. Returns the trained agent.

    Data collection runs on ``num_envs`` parallel environments (subprocesses
    when > 1) with one batched policy call per tick. A world-model +
    actor-critic update runs every ``train_every`` collected env steps.
    ``prefill`` random steps seed the replay buffer before learning starts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = {"domain_randomize": True} if domain_randomize else None
    envs = DreamerVecEnv(
        env_name, num_envs, action_repeat=action_repeat, seed=seed, env_kwargs=env_kwargs
    )
    logger.info(
        f"Dreamer on {env_name} x{num_envs} | image={envs.is_image} discrete={envs.is_discrete} "
        f"obs={envs.obs_shape} action_dim={envs.action_dim} action_repeat={action_repeat} "
        f"domain_randomize={domain_randomize}"
    )

    cfg = DreamerConfig(
        obs_shape=tuple(envs.obs_shape),
        action_dim=envs.action_dim,
        is_image=envs.is_image,
        is_discrete=envs.is_discrete,
        deter_dim=deter_dim,
        cnn_depth=cnn_depth,
        horizon=horizon,
        entropy_scale=entropy_scale,
    )

    if load_path:
        agent = Dreamer.load(Path(load_path), device)
        logger.info(f"Resumed Dreamer from {load_path}")
    else:
        agent = Dreamer(cfg).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"Dreamer parameters: {n_params / 1e6:.2f}M")

    model_opt = torch.optim.Adam(agent.wm.parameters(), lr=cfg.model_lr, eps=cfg.eps)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=cfg.actor_lr, eps=cfg.eps)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg.critic_lr, eps=cfg.eps)

    replay = VecSequenceReplay(
        max(replay_capacity // num_envs, seq_len * 4),
        num_envs,
        tuple(envs.obs_shape),
        envs.action_dim,
        envs.is_image,
    )

    writer = None
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(comment=f"_dreamer_{env_name}")

    save_path.mkdir(parents=True, exist_ok=True)

    n = num_envs
    zero_actions = np.zeros((n, envs.action_dim), dtype=np.float32)
    obs = envs.reset()
    replay.add_batch(obs, zero_actions, np.zeros(n, np.float32), np.ones(n, bool), np.zeros(n, bool))
    state: dict | None = None
    prev_action = torch.zeros(n, envs.action_dim, device=device)
    # is_first flags of the observations currently held in ``obs``.
    act_first = np.ones(n, dtype=np.float32)
    # Envs whose NEXT step() returns an autoreset observation (NextStep mode).
    pending_reset = np.zeros(n, dtype=bool)

    episode_reward = np.zeros(n, dtype=np.float64)
    episode_len = np.zeros(n, dtype=np.int64)
    reward_window: deque[float] = deque(maxlen=20)
    best_mean = float("-inf")
    train_steps = 0
    start_time = time.time()

    def train_step() -> dict:
        batch = replay.sample(batch_size, seq_len, device)
        # --- world model ---
        wm_loss, wm_metrics, post = agent.wm.loss(batch)
        model_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        nn.utils.clip_grad_norm_(agent.wm.parameters(), cfg.grad_clip)
        model_opt.step()
        # --- actor & critic in imagination ---
        actor_loss, critic_loss, ac_metrics = agent.actor_critic_loss(post)
        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.grad_clip)
        nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.grad_clip)
        actor_opt.step()
        critic_opt.step()
        agent.update_slow_critic()
        return {**wm_metrics, **ac_metrics}

    last_metrics: dict = {}
    step = 0  # collected env steps (excludes autoreset rows)
    since_train = 0
    next_log, next_save = log_every, save_every
    while step < total_steps:
        # ----- act (one batched policy call for all envs) -----
        if step <= prefill:
            actions_np = _random_actions(envs)
            action_t = torch.as_tensor(actions_np, device=device)
        else:
            obs_t = torch.as_tensor(obs, device=device)
            act_first_t = torch.as_tensor(act_first, device=device)
            action_t, state = agent.act(obs_t, state, prev_action, act_first_t, training=True)
            actions_np = action_t.detach().cpu().numpy()

        next_obs, rewards, terminated, truncated = envs.step(actions_np)

        # Rows where this step returned an autoreset observation: the env
        # ignored our action and gave reward 0 — store them as episode starts.
        row_first = pending_reset
        live = ~row_first
        replay.add_batch(
            next_obs,
            np.where(live[:, None], actions_np, 0.0).astype(np.float32),
            np.where(live, rewards, 0.0).astype(np.float32),
            row_first,
            terminated & live,
        )

        episode_reward += np.where(live, rewards, 0.0)
        episode_len += live.astype(np.int64)
        dones = (terminated | truncated) & live
        step_now = step + int(live.sum())
        for i in np.flatnonzero(dones):
            reward_window.append(float(episode_reward[i]))
            if writer:
                writer.add_scalar("rollout/episode_reward", episode_reward[i], step_now)
                writer.add_scalar("rollout/episode_len", episode_len[i], step_now)
            episode_reward[i] = 0.0
            episode_len[i] = 0

        since_train += int(live.sum())
        step = step_now
        obs = next_obs
        prev_action = action_t.detach()
        act_first = row_first.astype(np.float32)
        pending_reset = dones

        # ----- learn -----
        if step > prefill and replay.can_sample(seq_len):
            while since_train >= train_every:
                last_metrics = train_step()
                train_steps += 1
                since_train -= train_every
            if writer and train_steps:
                for k, v in last_metrics.items():
                    writer.add_scalar(k, v, step)

        # ----- log -----
        if step >= next_log:
            next_log += log_every
            mean_r = float(np.mean(reward_window)) if reward_window else float("nan")
            fps = step / (time.time() - start_time)
            logger.info(
                f"step {step:>7d} | episodes_mean_R {mean_r:8.1f} | "
                f"train_steps {train_steps:>6d} | fps {fps:5.1f} | "
                f"recon {last_metrics.get('wm/recon', float('nan')):7.3f} | "
                f"kl_dyn {last_metrics.get('wm/kl_dyn', float('nan')):6.3f} | "
                f"imag_R {last_metrics.get('ac/return', float('nan')):7.2f} | "
                f"entropy {last_metrics.get('ac/entropy', float('nan')):6.3f}"
            )
            if reward_window and mean_r > best_mean:
                best_mean = mean_r
                agent.save(save_path / f"dreamer_{env_name}_best.pt")

        # ----- checkpoint -----
        if step >= next_save:
            next_save += save_every
            agent.save(save_path / f"dreamer_{env_name}_step{step}.pt")

    final_path = save_path / f"dreamer_{env_name}_final.pt"
    agent.save(final_path)
    logger.info(f"Training complete. Saved to {final_path}. Best mean reward: {best_mean:.1f}")
    envs.close()
    if writer:
        writer.close()
    return agent


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    parser = argparse.ArgumentParser(description="Train DreamerV3-lite")
    parser.add_argument("--env", type=str, default="CarRacing-v3")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--prefill", type=int, default=2000)
    parser.add_argument("--train-every", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--deter-dim", type=int, default=256)
    parser.add_argument("--cnn-depth", type=int, default=32)
    parser.add_argument("--entropy-scale", type=float, default=1e-3)
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Parallel data-collection environments (subprocesses when > 1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="./saved_models/")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="CarRacing-v3: randomize track/grass/background colors each reset (robustness training).",
    )
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    train_dreamer(
        env_name=args.env,
        device=device,
        total_steps=args.total_steps,
        prefill=args.prefill,
        train_every=args.train_every,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        action_repeat=args.action_repeat,
        horizon=args.horizon,
        deter_dim=args.deter_dim,
        cnn_depth=args.cnn_depth,
        entropy_scale=args.entropy_scale,
        domain_randomize=args.domain_randomize,
        num_envs=args.num_envs,
        seed=args.seed,
        save_path=Path(args.save_path),
        load_path=args.load_path,
        use_tensorboard=args.tensorboard,
    )
