"""DreamerV3-lite training loop: collect → learn world model → imagine → learn actor-critic."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dreamer.env import DreamerEnv
from dreamer.models import Dreamer, DreamerConfig
from dreamer.replay import SequenceReplay

logger = logging.getLogger(__name__)


def _random_action(env: DreamerEnv, device: torch.device) -> torch.Tensor:
    """Sample a uniform random action in policy space, shape (1, action_dim)."""
    if env.is_discrete:
        idx = np.random.randint(env.action_dim)
        a = np.zeros(env.action_dim, dtype=np.float32)
        a[idx] = 1.0
    else:
        a = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
    return torch.as_tensor(a, device=device).unsqueeze(0)


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, device=device).unsqueeze(0)


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
    seed: int = 0,
    save_path: Path = Path("./saved_models/"),
    save_every: int = 10_000,
    log_every: int = 1000,
    eval_every: int = 10_000,
    use_tensorboard: bool = False,
    load_path: str = "",
) -> Dreamer:
    """Train a Dreamer agent on ``env_name``. Returns the trained agent.

    The loop interleaves one environment step with a world-model + actor-critic
    update every ``train_every`` steps. ``prefill`` random steps seed the replay
    buffer before learning starts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = {"domain_randomize": True} if domain_randomize else None
    env = DreamerEnv(env_name, action_repeat=action_repeat, seed=seed, env_kwargs=env_kwargs)
    logger.info(
        f"Dreamer on {env_name} | image={env.is_image} discrete={env.is_discrete} "
        f"obs={env.obs_shape} action_dim={env.action_dim} action_repeat={action_repeat} "
        f"domain_randomize={domain_randomize}"
    )

    cfg = DreamerConfig(
        obs_shape=tuple(env.obs_shape),
        action_dim=env.action_dim,
        is_image=env.is_image,
        is_discrete=env.is_discrete,
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

    replay = SequenceReplay(replay_capacity, tuple(env.obs_shape), env.action_dim, env.is_image)

    writer = None
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(comment=f"_dreamer_{env_name}")

    save_path.mkdir(parents=True, exist_ok=True)

    # ----- episode bookkeeping -----
    def reset_episode() -> tuple[np.ndarray, None, torch.Tensor]:
        obs = env.reset()
        replay.add(obs, np.zeros(env.action_dim, dtype=np.float32), 0.0, True, False)
        prev_action = torch.zeros(1, env.action_dim, device=device)
        return obs, None, prev_action

    obs, state, prev_action = reset_episode()
    is_first = True
    episode_reward = 0.0
    episode_len = 0
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
    for step in range(1, total_steps + 1):
        # ----- act -----
        if step <= prefill:
            action_t = _random_action(env, device)
        else:
            obs_t = _obs_to_tensor(obs, device)
            is_first_t = torch.tensor([1.0 if is_first else 0.0], device=device)
            action_t, state = agent.act(obs_t, state, prev_action, is_first_t, training=True)

        action_np = action_t[0].detach().cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        replay.add(next_obs, action_np, reward, False, terminated)

        episode_reward += reward
        episode_len += 1
        prev_action = action_t.detach()
        obs = next_obs
        is_first = False

        if terminated or truncated:
            reward_window.append(episode_reward)
            if writer:
                writer.add_scalar("rollout/episode_reward", episode_reward, step)
                writer.add_scalar("rollout/episode_len", episode_len, step)
            episode_reward = 0.0
            episode_len = 0
            obs, state, prev_action = reset_episode()
            is_first = True

        # ----- learn -----
        if step > prefill and step % train_every == 0 and replay.can_sample(seq_len):
            last_metrics = train_step()
            train_steps += 1
            if writer:
                for k, v in last_metrics.items():
                    writer.add_scalar(k, v, step)

        # ----- log -----
        if step % log_every == 0:
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
        if step % save_every == 0:
            agent.save(save_path / f"dreamer_{env_name}_step{step}.pt")

    final_path = save_path / f"dreamer_{env_name}_final.pt"
    agent.save(final_path)
    logger.info(f"Training complete. Saved to {final_path}. Best mean reward: {best_mean:.1f}")
    env.close()
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
        seed=args.seed,
        save_path=Path(args.save_path),
        load_path=args.load_path,
        use_tensorboard=args.tensorboard,
    )
