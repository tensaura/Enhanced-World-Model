"""Run a trained Dreamer agent on several *random* CarRacing tracks and record one video.

CarRacing procedurally generates a brand-new track on every reset, so each test below is
an unseen map. The video shows, per step: the real environment, the world-model's decoded
reconstruction, and a live reward sparkline — plus a title card and a final results table
(per-test return, mean ± std, best).

Usage:
    PYTHONPATH=src uv run python src/dreamer/eval_tests.py \
        --checkpoint saved_models/dreamer_CarRacing-v3_best.pt \
        --env CarRacing-v3 --tests 6 --out demo/carracing_tests --stochastic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from typing import Any

from dreamer.env import DreamerEnv
from dreamer.models import Dreamer

logger = logging.getLogger(__name__)

PANEL = 256
W = 2 * PANEL  # composed frame width
SPARK_H = 90
H = PANEL + SPARK_H  # composed frame height


def _label(
    img: cv2.typing.MatLike,
    text: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    thick: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel(rgb: cv2.typing.MatLike) -> np.ndarray:
    return cv2.resize(rgb, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _sparkline(history: list[float], width: int) -> np.ndarray:
    canvas = np.full((SPARK_H, width, 3), 20, dtype=np.uint8)
    if len(history) >= 2:
        lo, hi = min(history), max(history)
        span = max(hi - lo, 1e-6)
        pts = [
            (
                int(i / (len(history) - 1) * (width - 1)),
                int((1 - (v - lo) / span) * (SPARK_H - 12)) + 6,
            )
            for i, v in enumerate(history)
        ]
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, (80, 220, 120), 2, cv2.LINE_AA)
    _label(canvas, "cumulative reward", (8, 18))
    return canvas


def _compose(
    real: cv2.typing.MatLike,
    recon: np.typing.NDArray[np.floating[Any]],
    test_idx: int,
    n_tests: int,
    seed: int,
    step: int,
    ep_reward: float,
    history: list[float],
) -> cv2.typing.MatLike:
    left = _panel(real)
    _label(left, "environment", (8, 22))
    recon_u8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
    right = _panel(recon_u8)
    _label(right, "world-model dream", (8, 22))
    top = np.concatenate([left, right], axis=1)
    frame = np.concatenate([top, _sparkline(history, W)], axis=0)
    _label(frame, f"TEST {test_idx}/{n_tests}   random track (seed {seed})", (8, H - 40), 0.55)
    _label(frame, f"step {step:3d}   return {ep_reward:7.1f}", (8, H - 14), 0.6)
    return frame


def _card(
    lines: list[tuple[str, float, tuple[int, int, int]]],
    _highlight_idx: int | None = None,
    hold: int = 40,
) -> list[np.ndarray]:
    """A full-frame text card, repeated ``hold`` times so it lingers in the video."""
    canvas = np.full((H, W, 3), 18, dtype=np.uint8)
    y = 46
    for text, scale, color in lines:
        _label(canvas, text, (28, y), scale, color, thick=2 if scale >= 0.9 else 1)
        y += int(38 * scale) + 18
    return [canvas.copy() for _ in range(hold)]


PanelData = tuple[cv2.typing.MatLike, np.typing.NDArray[np.floating[Any]], int, float, list[float]]


@torch.no_grad()
def run_test(
    agent: Dreamer,
    env_name: str,
    seed: int,
    device: torch.device,
    action_repeat: int,
    stochastic: bool,
    domain_randomize: bool = False,
    max_steps: int = 1000,
) -> tuple[list[PanelData], float]:
    env_kwargs = {"domain_randomize": True} if domain_randomize else None
    env = DreamerEnv(
        env_name,
        action_repeat=action_repeat,
        seed=seed,
        render_mode="rgb_array",
        env_kwargs=env_kwargs,
    )
    obs = env.reset()
    state = agent.wm.rssm.initial(1, device)
    prev_action = torch.zeros(1, env.action_dim, device=device)
    is_first = torch.tensor([1.0], device=device)
    ep_reward, steps, done = 0.0, 0, False
    panels: list[np.ndarray] = []
    history: list[float] = []
    while not done and steps < max_steps:
        embed = agent.wm.encode(
            agent.wm.preprocess(torch.as_tensor(obs, device=device).unsqueeze(0))
        )
        state, _ = agent.wm.rssm.obs_step(state, prev_action, embed, is_first)
        feat = agent.wm.rssm.get_feat(state)
        dist = agent.actor(feat)
        action = dist.sample() if stochastic else dist.mode()
        recon = agent.wm.decode_obs(agent.wm.decoder(feat))[0].cpu().numpy()
        yield_frame = (env.render(), recon, steps, ep_reward, list(history))
        panels.append(yield_frame)
        action_np = action[0].cpu().numpy()
        obs, reward, term, trunc, _ = env.step(action_np)
        ep_reward += reward
        history.append(ep_reward)
        prev_action = action
        is_first = torch.tensor([0.0], device=device)
        done = term or trunc
        steps += 1
    env.close()
    return panels, ep_reward


@torch.no_grad()
def record_tests(
    checkpoint: str | Path,
    env_name: str,
    tests: int,
    out: str | Path,
    device: torch.device,
    action_repeat: int = 2,
    stochastic: bool = False,
    fps: int = 30,
    seed0: int = 0,
    domain_randomize: bool = False,
) -> None:
    agent = Dreamer.load(Path(checkpoint), device)
    agent.eval()
    rng = np.random.default_rng(seed0)
    seeds: list[int] = [int(rng.integers(0, 100_000)) for _ in range(tests)]
    mode = "random colors (domain randomize)" if domain_randomize else "random tracks"
    logger.info(f"Running {tests} tests | {mode} | seeds={seeds}")

    frames: list[cv2.typing.MatLike] = []
    frames += _card(
        [
            ("DreamerV3-lite  -  CarRacing-v3", 0.9, (120, 255, 160)),
            (f"{tests} tests  -  {mode}", 0.6, (255, 255, 255)),
            ("left: environment   right: world-model dream", 0.55, (200, 200, 200)),
        ],
        hold=int(fps * 1.5),
    )

    returns: list[float] = []
    for k, sd in enumerate(seeds, 1):
        panels, ep_reward = run_test(
            agent, env_name, sd, device, action_repeat, stochastic, domain_randomize
        )
        returns.append(ep_reward)
        logger.info(
            f"Test {k}/{tests} (seed {sd}): return {ep_reward:.1f} over {len(panels)} steps"
        )
        for real, recon, step, r, hist in panels:
            frames.append(_compose(real, recon, k, tests, sd, step, r, hist))

    mean, std, best = float(np.mean(returns)), float(np.std(returns)), float(np.max(returns))
    summary = [(f"RESULTS  -  {mode}", 0.8, (120, 255, 160))]
    summary += [
        (f"Test {i + 1} (seed {s}):  {r:7.1f}", 0.55, (255, 255, 255))
        for i, (s, r) in enumerate(zip(seeds, returns, strict=False))
    ]
    summary += [
        (f"mean {mean:.0f} +/- {std:.0f}    best {best:.0f}", 0.7, (120, 255, 160)),
        ("(random policy ~ -30   |   solved ~ 900)", 0.5, (200, 200, 200)),
    ]
    frames += _card(summary, hold=int(fps * 3.5))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path.with_suffix(".gif"), frames[::2], duration=2.0 / fps)
    try:
        imageio.mimsave(out_path.with_suffix(".mp4"), frames, fps=fps, macro_block_size=None)
        logger.info(f"Saved {out_path.with_suffix('.mp4')}")
    except Exception as e:
        logger.warning(f"MP4 failed ({e}); GIF saved.")
    logger.info(
        f"Saved {out_path.with_suffix('.gif')} ({len(frames)} frames). "
        f"Mean {mean:.1f} +/- {std:.1f} over {tests} random tracks."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    parser = argparse.ArgumentParser(description="Evaluate a Dreamer agent on random tracks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--tests", type=int, default=6)
    parser.add_argument("--out", type=str, default="demo/dreamer_tests")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed0", type=int, default=0, help="Seed for choosing the track seeds.")
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="CarRacing-v3: randomize track/grass/background colors each reset (visual robustness test).",
    )
    args = parser.parse_args()

    record_tests(
        checkpoint=args.checkpoint,
        env_name=args.env,
        tests=args.tests,
        out=args.out,
        device=device,
        action_repeat=args.action_repeat,
        stochastic=args.stochastic,
        fps=args.fps,
        seed0=args.seed0,
        domain_randomize=args.domain_randomize,
    )
