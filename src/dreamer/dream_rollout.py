"""Record the agent *playing inside its own world-model dream* (open-loop imagination).

Procedure:
    1. Warm-up (``context`` steps): the agent acts in the real environment and updates
       its posterior latent from real frames — this grounds the dream in a real moment.
    2. Dream (``horizon`` steps): the environment is cut off. The agent chooses actions
       from its *imagined* latent, the RSSM rolls the prior forward with **no observations**,
       and each imagined latent is decoded back to pixels. The same actions are also applied
       to the real environment so we can watch dream vs. reality diverge.

The result is a side-by-side video: left = reality given the dreamed actions,
right = the agent's hallucinated rollout (what it *thinks* is happening).

Usage:
    PYTHONPATH=src uv run python src/dreamer/dream_rollout.py \
        --checkpoint saved_models/dreamer_CarRacing-v3_best.pt \
        --env CarRacing-v3 --context 25 --horizon 60 --out demo/carracing_dream
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

PANEL = 256


def _label(img: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _panel(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _compose(
    real: np.ndarray,
    dream: np.ndarray,
    phase: str,
    step: int,
    dream_reward: float,
) -> np.ndarray:
    left = _panel(real)
    _label(left, "reality (same actions)", (8, 22))
    dream_u8 = (np.clip(dream, 0, 1) * 255).astype(np.uint8)
    right = _panel(dream_u8)
    border = (90, 220, 120) if phase == "dream" else (200, 200, 200)
    right = cv2.copyMakeBorder(right, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border)
    right = cv2.resize(right, (PANEL, PANEL))
    rlabel = "DREAM — no real input" if phase == "dream" else "world-model (grounded)"
    _label(right, rlabel, (8, 22), color=(120, 255, 160) if phase == "dream" else (255, 255, 255))
    frame = np.concatenate([left, right], axis=1)
    bar = np.full((34, frame.shape[1], 3), 20, dtype=np.uint8)
    frame = np.concatenate([frame, bar], axis=0)
    _label(
        frame,
        f"{phase}  step {step:3d}   imagined reward {dream_reward:+5.2f}",
        (8, frame.shape[0] - 11),
    )
    return frame


@torch.no_grad()
def record_dream(
    checkpoint: str,
    env_name: str,
    context: int,
    horizon: int,
    out: str,
    device: torch.device,
    action_repeat: int = 2,
    stochastic: bool = False,
    fps: int = 20,
) -> None:
    agent = Dreamer.load(Path(checkpoint), device)
    agent.eval()
    wm, rssm = agent.wm, agent.wm.rssm
    env = DreamerEnv(env_name, action_repeat=action_repeat, render_mode="rgb_array")

    frames: list[np.ndarray] = []

    # ---- 1. Warm-up: ground the latent in real frames ----
    obs = env.reset()
    state = rssm.initial(1, device)
    prev_action = torch.zeros(1, env.action_dim, device=device)
    is_first = torch.tensor([1.0], device=device)
    for t in range(context):
        embed = wm.encode(wm.preprocess(torch.as_tensor(obs, device=device).unsqueeze(0)))
        state, _ = rssm.obs_step(state, prev_action, embed, is_first)
        feat = rssm.get_feat(state)
        dist = agent.actor(feat)
        action = dist.sample() if stochastic else dist.mode()
        recon = wm.decode_obs(wm.decoder(feat))[0].cpu().numpy()
        frames.append(_compose(env.render(), recon, "context", t, 0.0))
        action_np = action[0].cpu().numpy()
        obs, _, term, trunc, _ = env.step(action_np)
        prev_action = action
        is_first = torch.tensor([0.0], device=device)
        if term or trunc:
            break

    # ---- 2. Dream: cut the environment off, roll the prior in pure imagination ----
    logger.info(f"Dreaming {horizon} steps with no real input...")
    for t in range(horizon):
        feat = rssm.get_feat(state)
        dist = agent.actor(feat)
        action = dist.sample() if stochastic else dist.mode()
        action_np = action[0].cpu().numpy()

        # Reality is stepped with the SAME action, for comparison only.
        real_obs, _, term, trunc, _ = env.step(action_np)

        # The agent's world advances by imagination alone (prior, no observation).
        state = rssm.img_step(state, action)
        feat_next = rssm.get_feat(state)
        dream_img = wm.decode_obs(wm.decoder(feat_next))[0].cpu().numpy()
        dream_reward = float(wm.reward_head(feat_next).mean().item())
        frames.append(_compose(env.render(), dream_img, "dream", t, dream_reward))
        if term or trunc:
            break

    env.close()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path.with_suffix(".gif"), frames, duration=1.0 / fps)
    try:
        imageio.mimsave(out_path.with_suffix(".mp4"), frames, fps=fps, macro_block_size=None)
        logger.info(f"Saved {out_path.with_suffix('.mp4')}")
    except Exception as e:
        logger.warning(f"MP4 failed ({e}); GIF saved.")
    logger.info(f"Saved {out_path.with_suffix('.gif')} ({len(frames)} frames)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    parser = argparse.ArgumentParser(description="Record the agent playing inside its dream")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--context", type=int, default=25, help="Real warm-up steps.")
    parser.add_argument("--horizon", type=int, default=60, help="Pure-imagination steps.")
    parser.add_argument("--out", type=str, default="demo/dreamer_dream")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    record_dream(
        checkpoint=args.checkpoint,
        env_name=args.env,
        context=args.context,
        horizon=args.horizon,
        out=args.out,
        device=device,
        action_repeat=args.action_repeat,
        stochastic=args.stochastic,
        fps=args.fps,
    )
