"""Record a showcase video/GIF of a trained Dreamer agent.

For each agent step the panel shows, side by side:

    - the real environment frame the agent is acting on;
    - (image envs) the world model's decoded reconstruction of that frame — proof the
      latent state captures the scene;
    - a live cumulative-reward sparkline and step/return overlay.

Usage:
    PYTHONPATH=src uv run python src/dreamer/record_demo.py \
        --checkpoint saved_models/dreamer_CarRacing-v3_final.pt \
        --env CarRacing-v3 --episodes 3 --out demo/carracing
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from dreamer.env import DreamerEnv
from dreamer.models import Dreamer

logger = logging.getLogger(__name__)

PANEL = 256  # per-panel side length in the composed frame


def _label(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def _to_panel(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _sparkline(history: list[float], width: int, height: int = 90) -> np.ndarray:
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    if len(history) >= 2:
        lo, hi = min(history), max(history)
        span = max(hi - lo, 1e-6)
        pts = []
        for i, v in enumerate(history):
            x = int(i / (len(history) - 1) * (width - 1))
            y = int((1 - (v - lo) / span) * (height - 12)) + 6
            pts.append((x, y))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, (80, 220, 120), 2, cv2.LINE_AA)
    _label(canvas, "cumulative reward", (8, 18))
    return canvas


def compose(
    real: np.ndarray, recon: np.ndarray | None, step: int, ep: int, ep_reward: float, history: list[float]
) -> np.ndarray:
    panels = [_to_panel(real)]
    _label(panels[0], "environment", (8, 22))
    if recon is not None:
        recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        rp = _to_panel(recon_u8)
        _label(rp, "world-model dream", (8, 22))
        panels.append(rp)
    top = np.concatenate(panels, axis=1)
    spark = _sparkline(history, top.shape[1])
    frame = np.concatenate([top, spark], axis=0)
    _label(frame, f"ep {ep}  step {step}  return {ep_reward:7.1f}", (8, frame.shape[0] - 12))
    return frame


def record(
    checkpoint: str,
    env_name: str,
    episodes: int,
    out: str,
    device: torch.device,
    action_repeat: int = 2,
    fps: int = 30,
    max_steps: int = 1000,
    stochastic: bool = False,
) -> None:
    agent = Dreamer.load(Path(checkpoint), device)
    agent.eval()
    env = DreamerEnv(env_name, action_repeat=action_repeat, render_mode="rgb_array")
    logger.info(f"Recording {episodes} episode(s) of {env_name} from {checkpoint}")

    frames: list[np.ndarray] = []
    returns: list[float] = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        state: dict | None = None
        prev_action = torch.zeros(1, env.action_dim, device=device)
        is_first = True
        ep_reward = 0.0
        history: list[float] = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            is_first_t = torch.tensor([1.0 if is_first else 0.0], device=device)
            action, state = agent.act(obs_t, state, prev_action, is_first_t, training=stochastic)

            recon_img = None
            if env.is_image:
                with torch.no_grad():
                    feat = agent.wm.rssm.get_feat(state)
                    recon = agent.wm.decode_obs(agent.wm.decoder(feat))
                recon_img = recon[0].detach().cpu().numpy()

            real = env.render()
            frames.append(compose(real, recon_img, steps, ep, ep_reward, history))

            action_np = action[0].detach().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward
            history.append(ep_reward)
            prev_action = action
            is_first = False
            done = terminated or truncated
            steps += 1

        returns.append(ep_reward)
        logger.info(f"Episode {ep}: return {ep_reward:.1f} over {steps} steps")

    env.close()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gif_path = out_path.with_suffix(".gif")
    imageio.mimsave(gif_path, frames[::2], duration=1.0 / (fps / 2))
    logger.info(f"Saved {gif_path}")
    try:
        mp4_path = out_path.with_suffix(".mp4")
        imageio.mimsave(mp4_path, frames, fps=fps, macro_block_size=None)
        logger.info(f"Saved {mp4_path}")
    except Exception as e:  # ffmpeg may be unavailable
        logger.warning(f"Could not write MP4 ({e}); GIF saved instead.")

    logger.info(f"Mean return over {episodes} episodes: {np.mean(returns):.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    parser = argparse.ArgumentParser(description="Record a Dreamer demo video")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out", type=str, default="demo/dreamer")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of using the greedy mode "
        "(more representative for a mid-training agent whose policy is still exploration-driven).",
    )
    args = parser.parse_args()

    record(
        checkpoint=args.checkpoint,
        env_name=args.env,
        episodes=args.episodes,
        out=args.out,
        device=device,
        action_repeat=args.action_repeat,
        fps=args.fps,
        stochastic=args.stochastic,
    )
