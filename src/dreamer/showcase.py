"""Render a LinkedIn-ready cinematic showcase of a trained Dreamer agent.

One self-contained MP4 with a narrative arc:

    1. Title card.
    2. Drive — the agent racing from pixels, side by side with the world model's
       live reconstruction of the scene (proof the latent state "sees").
       The episode is the best of ``--seeds`` scouted tracks.
    3. Transition card — "we switch the environment OFF".
    4. Dream — open-loop imagination: no observations, the RSSM prior rolls
       forward and is decoded to pixels while the same actions play out in the
       real env for comparison. Rendered at half speed for effect.
    5. End card — parameters, mean return over the scouted tracks, and the
       on-device inference latency measured live on this machine.

Also writes a short teaser GIF of the dream segment and a JSON sidecar with the
measured stats (handy for writing the post itself).

Usage:
    PYTHONPATH=src uv run python src/dreamer/showcase.py \
        --checkpoint saved_models/dreamer_CarRacing-v3_best.pt \
        --env CarRacing-v3 --seeds 6 --out demo/showcase
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from dreamer.env import DreamerEnv
from dreamer.models import Dreamer

logger = logging.getLogger(__name__)

W, H, FPS = 1280, 720, 30
PANEL = 500
PANEL_Y = 84
PANEL_X1, PANEL_X2 = 100, 680
BG = (15, 17, 21)
FG = (235, 238, 242)
DIM = (150, 158, 170)
ACCENT = (110, 231, 150)  # dream green
WARN = (255, 120, 110)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX


def _chip_name() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _text(
    img: np.ndarray,
    s: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = FG,
    thickness: int = 1,
    font: int = FONT_S,
    shadow: bool = True,
) -> None:
    # Hershey fonts are ASCII-only; anything else renders as '?'.
    s = s.replace("·", "|").replace("—", "-").replace("–", "-")
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, color, thickness, cv2.LINE_AA)


def _text_centered(img: np.ndarray, s: str, y: int, scale: float, color=FG, thickness=1, font=FONT) -> None:
    (tw, _), _ = cv2.getTextSize(s, font, scale, thickness)
    _text(img, s, ((img.shape[1] - tw) // 2, y), scale, color, thickness, font)


def _canvas() -> np.ndarray:
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[:] = BG
    return frame


def _square(rgb: np.ndarray) -> np.ndarray:
    """Center-crop a render to a square before resizing (keeps aspect ratio)."""
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return rgb[y0 : y0 + side, x0 : x0 + side]


def _panel_img(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(_square(rgb), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _draw_panel(
    frame: np.ndarray,
    img: np.ndarray,
    x: int,
    label: str,
    sublabel: str = "",
    border: tuple[int, int, int] | None = None,
) -> None:
    p = _panel_img(img)
    frame[PANEL_Y : PANEL_Y + PANEL, x : x + PANEL] = p
    color = border or (60, 65, 75)
    cv2.rectangle(frame, (x - 2, PANEL_Y - 2), (x + PANEL + 1, PANEL_Y + PANEL + 1), color, 2)
    _text(frame, label, (x + 12, PANEL_Y + 28), 0.62, FG, 1, FONT)
    if sublabel:
        _text(frame, sublabel, (x + 12, PANEL_Y + 54), 0.5, border or DIM, 1, FONT_S)


def _header(frame: np.ndarray, left: str, right: str) -> None:
    _text(frame, left, (PANEL_X1, 48), 0.7, FG, 1, FONT)
    (tw, _), _ = cv2.getTextSize(right, FONT_S, 0.5, 1)
    _text(frame, right, (PANEL_X2 + PANEL - tw, 48), 0.5, DIM, 1, FONT_S)


def _sparkline(frame: np.ndarray, history: list[float], y0: int, h: int = 70) -> None:
    x0, x1 = PANEL_X1, PANEL_X2 + PANEL
    if len(history) >= 2:
        lo, hi = min(history), max(history)
        span = max(hi - lo, 1e-6)
        pts = []
        for i, v in enumerate(history):
            x = int(x0 + i / (len(history) - 1) * (x1 - x0 - 1))
            y = int(y0 + (1 - (v - lo) / span) * (h - 8)) + 4
            pts.append((x, y))
        cv2.polylines(frame, [np.array(pts, np.int32)], False, ACCENT, 2, cv2.LINE_AA)


def _fade(frame: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return frame
    bg = np.empty_like(frame)
    bg[:] = BG
    return cv2.addWeighted(frame, alpha, bg, 1 - alpha, 0)


def _card(lines: list[tuple[str, float, tuple[int, int, int]]], seconds: float) -> list[np.ndarray]:
    """A fading title card. ``lines`` are (text, scale, color) centered vertically."""
    base = _canvas()
    gap = 64
    total = gap * (len(lines) - 1)
    y = (H - total) // 2
    for s, scale, color in lines:
        _text_centered(base, s, y, scale, color, 1 if scale < 1.0 else 2)
        y += gap
    n = int(seconds * FPS)
    fade_n = int(0.5 * FPS)
    frames = []
    for i in range(n):
        a = min(1.0, (i + 1) / fade_n, (n - i) / fade_n)
        frames.append(_fade(base, a))
    return frames


# --------------------------------------------------------------------- agent I/O


@torch.no_grad()
def _run_episode(
    agent: Dreamer,
    env_name: str,
    seed: int,
    device: torch.device,
    action_repeat: int,
    max_steps: int,
    record: bool = False,
) -> tuple[float, list[dict]]:
    """Greedy episode on a fixed track seed. With ``record``, returns per-step frames."""
    env = DreamerEnv(env_name, action_repeat=action_repeat, seed=seed, render_mode="rgb_array" if record else None)
    obs = env.reset()
    state: dict | None = None
    prev_action = torch.zeros(1, env.action_dim, device=device)
    is_first = True
    ep_reward, steps, done = 0.0, 0, False
    trace: list[dict] = []
    while not done and steps < max_steps:
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        is_first_t = torch.tensor([1.0 if is_first else 0.0], device=device)
        action, state = agent.act(obs_t, state, prev_action, is_first_t, training=False)
        if record:
            feat = agent.wm.rssm.get_feat(state)
            recon = agent.wm.decode_obs(agent.wm.decoder(feat))[0].cpu().numpy()
            trace.append({"real": env.render(), "recon": recon, "ret": ep_reward})
        obs, reward, term, trunc, _ = env.step(action[0].cpu().numpy())
        ep_reward += reward
        prev_action, is_first = action, False
        done = term or trunc
        steps += 1
    env.close()
    return ep_reward, trace


@torch.no_grad()
def _run_dream(
    agent: Dreamer, env_name: str, seed: int, device: torch.device, action_repeat: int, context: int, horizon: int
) -> list[dict]:
    """Warm up on real frames, then roll the prior open-loop (no observations)."""
    wm, rssm = agent.wm, agent.wm.rssm
    env = DreamerEnv(env_name, action_repeat=action_repeat, seed=seed, render_mode="rgb_array")
    obs = env.reset()
    state = rssm.initial(1, device)
    prev_action = torch.zeros(1, env.action_dim, device=device)
    is_first = torch.tensor([1.0], device=device)
    trace: list[dict] = []

    for _t in range(context):
        embed = wm.encode(wm.preprocess(torch.as_tensor(obs, device=device).unsqueeze(0)))
        state, _ = rssm.obs_step(state, prev_action, embed, is_first)
        feat = rssm.get_feat(state)
        action = agent.actor(feat).mode()
        recon = wm.decode_obs(wm.decoder(feat))[0].cpu().numpy()
        trace.append({"real": env.render(), "dream": recon, "phase": "context", "r": 0.0})
        obs, _, term, trunc, _ = env.step(action[0].cpu().numpy())
        prev_action, is_first = action, torch.tensor([0.0], device=device)
        if term or trunc:
            break

    for _t in range(horizon):
        feat = rssm.get_feat(state)
        action = agent.actor(feat).mode()
        real_done = False
        try:
            _, _, term, trunc, _ = env.step(action[0].cpu().numpy())
            real_done = term or trunc
        except Exception:
            real_done = True
        state = rssm.img_step(state, action)
        feat = rssm.get_feat(state)
        dream = wm.decode_obs(wm.decoder(feat))[0].cpu().numpy()
        r = float(wm.reward_head(feat).mean().item())
        trace.append({"real": env.render(), "dream": dream, "phase": "dream", "r": r})
        if real_done:
            break
    env.close()
    return trace


@torch.no_grad()
def _bench_latency(agent: Dreamer, env: DreamerEnv, device: torch.device, iters: int = 100) -> float:
    """Median latency (ms) of one control step: encode + posterior update + actor."""
    obs = torch.zeros(1, *env.obs_shape, device=device)
    prev_action = torch.zeros(1, env.action_dim, device=device)
    is_first = torch.tensor([0.0], device=device)
    state = None
    for _ in range(10):  # warmup
        _, state = agent.act(obs, state, prev_action, is_first, training=False)
    times = []
    for _ in range(iters):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _, state = agent.act(obs, state, prev_action, is_first, training=False)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# ------------------------------------------------------------------ composition


def _compose_drive(step: dict, i: int, n: int, history: list[float], badge: str) -> np.ndarray:
    frame = _canvas()
    _header(frame, "DreamerV3 | from scratch on a MacBook Air", badge)
    _draw_panel(frame, step["real"], PANEL_X1, "REAL ENVIRONMENT", "raw pixels, 64x64 to the agent")
    recon_u8 = (np.clip(step["recon"], 0, 1) * 255).astype(np.uint8)
    _draw_panel(frame, recon_u8, PANEL_X2, "WORLD MODEL", "what the agent believes it sees")
    _sparkline(frame, history, 616)
    _text(frame, f"return {history[-1] if history else 0.0:7.1f}", (PANEL_X1, 706), 0.6, ACCENT, 1, FONT)
    _text(frame, f"step {i}/{n}", (PANEL_X2 + PANEL - 130, 706), 0.55, DIM)
    return frame


def _compose_dream(step: dict, i: int, badge: str) -> np.ndarray:
    frame = _canvas()
    dreaming = step["phase"] == "dream"
    _header(frame, "DreamerV3 | from scratch on a MacBook Air", badge)
    _draw_panel(frame, step["real"], PANEL_X1, "REALITY", "same actions, for comparison")
    dream_u8 = (np.clip(step["dream"], 0, 1) * 255).astype(np.uint8)
    if dreaming:
        _draw_panel(frame, dream_u8, PANEL_X2, "PURE IMAGINATION", "no camera input — env is OFF", border=ACCENT)
        _text(frame, "ENVIRONMENT: OFF", (PANEL_X1, 640), 0.7, WARN, 1, FONT)
        _text(frame, "the agent is driving inside its own world model", (PANEL_X1, 672), 0.55, DIM)
        _text(frame, f"imagined reward {step['r']:+5.2f}", (PANEL_X2 + PANEL - 260, 640), 0.55, ACCENT)
    else:
        _draw_panel(frame, dream_u8, PANEL_X2, "WORLD MODEL", "grounding the dream in real frames")
        _text(frame, "ENVIRONMENT: ON  — warming up the latent state", (PANEL_X1, 640), 0.6, DIM, 1, FONT)
    _text(frame, f"step {i}", (PANEL_X2 + PANEL - 130, 706), 0.55, DIM)
    return frame


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the LinkedIn showcase video")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", type=str, default="CarRacing-v3")
    parser.add_argument("--seeds", type=int, default=6, help="Tracks to scout; best one is recorded.")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--out", type=str, default="demo/showcase")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=90)
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    agent = Dreamer.load(Path(args.checkpoint), device)
    agent.eval()
    n_params = sum(p.numel() for p in agent.parameters())
    n_wm = sum(p.numel() for p in agent.wm.parameters())
    chip = _chip_name()

    bench_env = DreamerEnv(args.env, action_repeat=args.action_repeat, seed=0)
    latency_ms = _bench_latency(agent, bench_env, device)
    bench_env.close()
    control_hz = 1000.0 / latency_ms
    logger.info(f"On-device inference: {latency_ms:.1f} ms/action ({control_hz:.0f} Hz) on {device}")

    # Evaluate on unseen tracks, recording every episode and keeping the best one.
    # (MPS inference is nondeterministic, so a scouted return cannot be reproduced
    # by re-running the seed — the shown drive must BE one of the scored runs.)
    returns: dict[int, float] = {}
    best_seed, best_ret = -1, -np.inf
    drive: list[dict] = []
    for k in range(args.seeds):
        seed = args.seed_base + k
        ret, trace = _run_episode(agent, args.env, seed, device, args.action_repeat, args.max_steps, record=True)
        returns[seed] = ret
        logger.info(f"Eval seed {seed}: return {ret:.1f}")
        if ret > best_ret:
            best_seed, best_ret, drive = seed, ret, trace
    mean_ret = float(np.mean(list(returns.values())))
    logger.info(
        f"Best seed {best_seed} ({best_ret:.1f}); mean over {args.seeds}: {mean_ret:.1f}; "
        f"recorded drive has {len(drive)} steps"
    )

    logger.info("Recording the dream rollout...")
    dream = _run_dream(agent, args.env, best_seed + 7, device, args.action_repeat, args.context, args.horizon)

    # ---- write the video, streaming frames straight to ffmpeg ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4_path = out.with_suffix(".mp4")
    writer = imageio.get_writer(
        mp4_path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None
    )
    badge = f"{chip} | MPS | {latency_ms:.1f} ms/action"
    total_frames = 0

    for f in _card(
        [
            ("Can a neural network dream a racetrack?", 1.15, FG),
            ("A DreamerV3 world model — built and trained from scratch", 0.7, DIM),
            ("on a MacBook Air.", 0.7, DIM),
        ],
        3.2,
    ):
        writer.append_data(f)
        total_frames += 1

    history: list[float] = []
    for i, step in enumerate(drive):
        history.append(step["ret"])
        writer.append_data(_compose_drive(step, i, len(drive), history, badge))
        total_frames += 1

    for f in _card(
        [
            ("Now we switch the environment OFF.", 1.1, FG),
            ("No more camera input. The agent keeps driving", 0.7, DIM),
            ("inside a world it imagines, frame by frame.", 0.7, DIM),
        ],
        3.2,
    ):
        writer.append_data(f)
        total_frames += 1

    teaser: list[np.ndarray] = []
    for i, step in enumerate(dream):
        frame = _compose_dream(step, i, badge)
        repeats = 2 if step["phase"] == "dream" else 1  # slow-mo the imagination
        for _ in range(repeats):
            writer.append_data(frame)
            total_frames += 1
        if step["phase"] == "dream":
            teaser.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))

    stats_lines = [
        ("DreamerV3-lite  ·  from scratch", 1.0, FG),
        (f"{n_params / 1e6:.1f}M parameters  ({n_wm / 1e6:.1f}M world model)", 0.75, ACCENT),
        (
            f"return: best {max(returns.values()):.0f} | "
            f"mean {mean_ret:.0f} over {args.seeds} unseen tracks (random = -30)",
            0.75,
            ACCENT,
        ),
        (f"{latency_ms:.1f} ms/action on-device  ·  {control_hz:.0f} Hz control loop", 0.75, ACCENT),
        (f"{chip} MacBook Air  ·  PyTorch MPS  ·  no CUDA, no cloud", 0.65, DIM),
        ("github.com/Larwive/Enhanced-World-Model", 0.65, DIM),
    ]
    for f in _card(stats_lines, 5.0):
        writer.append_data(f)
        total_frames += 1

    writer.close()
    logger.info(f"Saved {mp4_path} ({total_frames} frames, {total_frames / FPS:.1f}s)")

    gif_path = out.with_name(out.name + "_dream_teaser.gif")
    imageio.mimsave(gif_path, teaser[::2], duration=2 / FPS * 2)
    logger.info(f"Saved {gif_path} ({len(teaser[::2])} frames)")

    stats = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "chip": chip,
        "params_total": n_params,
        "params_world_model": n_wm,
        "latency_ms_per_action": round(latency_ms, 2),
        "control_hz": round(control_hz, 1),
        "scout_returns": returns,
        "mean_return": round(mean_ret, 1),
        "best_seed": best_seed,
        "recorded_return": round(best_ret, 1),
        "video_seconds": round(total_frames / FPS, 1),
    }
    stats_path = out.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Saved {stats_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
