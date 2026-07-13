"""DreamDrive: Real World Edition — drive inside a world model of real streets.

The world model was trained on real dashcam video (comma.ai) with steering and
speed from the CAN bus. A few seconds of real footage ground the latent state —
then the footage stops and every frame is the model's imagination, conditioned
on YOUR steering and speed. No simulator exists for this world: it is a dream
of California, running on a laptop.

Controls:
    left/right  steer
    up/down     raise / lower the speed setpoint
    A           ghost mode: replay the real driver's recorded actions in the dream
    TAB         next level (a different real moment as the starting point)
    R           restart current level
    ESC / Q     quit

Usage:
    PYTHONPATH=src uv run python src/dreamer/dreamdrive_rw.py \
        --checkpoint saved_models/comma_wm/video_wm_final.pt --data data/comma \
        --record demo/dreamdrive_rw_session.mp4

    # headless scripted demo (ghost, then steer left / right / straight):
    ... dreamdrive_rw.py --checkpoint ... --data ... --selftest demo/dreamdrive_rw_selftest.mp4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from dreamer.dreamdrive import (
    ACCENT,
    BG,
    DIM,
    DREAM,
    FG,
    FONT,
    FONT_S,
    H,
    PANEL_X,
    PANEL_Y,
    SIDE_X,
    W,
    WARN,
    _draw_arrow_pad,
    _text,
)
from dreamer.models import Dreamer
from dreamer.video_world import VideoSequences

logger = logging.getLogger(__name__)

FPS = 20  # dataset is 20 Hz, so 20 fps play is exactly real time
SPEED_MAX_MS = 35.0


class RealDreamSession:
    """Latent-state game grounded on real dashcam moments — no environment."""

    def __init__(
        self,
        checkpoint: str,
        data_dir: str,
        device: torch.device,
        n_levels: int = 6,
        context: int = 20,
        ghost_len: int = 400,
        seed: int = 0,
    ) -> None:
        self.device = device
        self.agent = Dreamer.load(Path(checkpoint), device)
        self.agent.eval()
        self.wm, self.rssm = self.agent.wm, self.agent.wm.rssm
        self.n_params = sum(p.numel() for p in self.agent.parameters())
        self.context = context

        data = VideoSequences(Path(data_dir))
        rng = np.random.default_rng(seed)
        candidates = rng.permutation(len(data.frames) - context - ghost_len)
        self.levels: list[dict] = []
        for s in candidates:
            window_first = data.is_first[s : s + context + ghost_len]
            if window_first[1:].any():
                continue  # crosses an episode boundary
            if data.actions[s : s + context, 1].mean() < -0.45:
                continue  # car mostly standing still (< ~10 m/s)
            self.levels.append(
                {
                    "frames": data.frames[s : s + context],
                    "actions": data.actions[s : s + context + ghost_len],
                }
            )
            if len(self.levels) >= n_levels:
                break
        if not self.levels:
            raise RuntimeError("No moving, boundary-free context windows found in the data")
        logger.info(f"Prepared {len(self.levels)} real-moment levels "
                    f"({self.n_params / 1e6:.1f}M params on {device})")
        self.state: dict[str, torch.Tensor] = {}
        self.steps = 0
        self._level = 0
        self._ghost_ptr = context

    @torch.no_grad()
    def reset(self, level: int) -> float:
        """Ground the latent on the level's real context. Returns the recorded
        speed setpoint at hand-over, so the player starts at the real pace."""
        self._level = level % len(self.levels)
        lv = self.levels[self._level]
        state = self.rssm.initial(1, self.device)
        prev = torch.zeros(1, 2, device=self.device)
        is_first = torch.tensor([1.0], device=self.device)
        for t in range(self.context):
            obs_t = torch.as_tensor(lv["frames"][t], device=self.device).unsqueeze(0)
            embed = self.wm.encode(self.wm.preprocess(obs_t))
            state, _ = self.rssm.obs_step(state, prev, embed, is_first)
            prev = torch.as_tensor(lv["actions"][t + 1], device=self.device).unsqueeze(0)
            is_first = torch.tensor([0.0], device=self.device)
        self.state = state
        self.steps = 0
        self._ghost_ptr = self.context
        return float(lv["actions"][self.context, 1])

    @torch.no_grad()
    def step(self, steer: float, speed: float) -> np.ndarray:
        action = torch.tensor([[steer, speed]], dtype=torch.float32, device=self.device)
        self.state = self.rssm.img_step(self.state, action)
        img = self.wm.decode_obs(self.wm.decoder(self.rssm.get_feat(self.state)))[0].cpu().numpy()
        self.steps += 1
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def ghost_action(self) -> tuple[float, float]:
        """The real driver's next recorded action (replayed inside the dream)."""
        lv = self.levels[self._level]
        a = lv["actions"][min(self._ghost_ptr, len(lv["actions"]) - 1)]
        self._ghost_ptr += 1
        return float(a[0]), float(a[1])


def compose_rw(dream64: np.ndarray, hud: dict) -> np.ndarray:
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[:] = BG
    panel = cv2.resize(dream64, (DREAM, DREAM), interpolation=cv2.INTER_NEAREST)
    frame[PANEL_Y : PANEL_Y + DREAM, PANEL_X : PANEL_X + DREAM] = panel
    cv2.rectangle(frame, (PANEL_X - 2, PANEL_Y - 2),
                  (PANEL_X + DREAM + 1, PANEL_Y + DREAM + 1), ACCENT, 2)

    _text(frame, "DREAMDRIVE | REAL WORLD", (SIDE_X, 92), 1.0, FG, 2, FONT)
    _text(frame, "a dream of real California streets", (SIDE_X, 128), 0.6, DIM)
    _text(frame, f"world model: {hud['params'] / 1e6:.0f}M params, trained on dashcam", (SIDE_X, 166), 0.5, DIM)
    _text(frame, "video + CAN bus | no simulator | MacBook Air", (SIDE_X, 190), 0.5, DIM)

    mode = "GHOST - the real driver's actions" if hud["ghost"] else "MANUAL - you are driving the dream"
    _text(frame, mode, (SIDE_X, 240), 0.6, ACCENT if hud["ghost"] else FG, 1, FONT)
    _text(frame, f"level {hud['level'] + 1}/{hud['n_levels']}   dream step {hud['steps']}", (SIDE_X, 272), 0.55, DIM)
    _text(frame, f"{hud['fps']:4.0f} fps | {hud['ms']:4.1f} ms/frame | {hud['device']}", (SIDE_X, 298), 0.55, DIM)

    _draw_arrow_pad(frame, SIDE_X + 40, 370, hud["steer"], hud["accel"], hud["decel"])

    # Speed setpoint bar (what we tell the world model the car is doing).
    x0, y0, w, h = SIDE_X, 540, W - SIDE_X - 60, 16
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (60, 65, 75), 1)
    fill = int((hud["speed"] + 1) / 2 * (w - 2))
    frame[y0 + 1 : y0 + h - 1, x0 + 1 : x0 + 1 + max(fill, 0)] = ACCENT
    kmh = (hud["speed"] + 1) / 2 * SPEED_MAX_MS * 3.6
    _text(frame, f"speed setpoint  {kmh:5.0f} km/h", (x0, y0 + 44), 0.55, ACCENT)

    _text(frame, "arrows steer/speed | A ghost | TAB level | R restart | ESC quit",
          (SIDE_X, H - 46), 0.48, DIM)
    if hud["steps"] > 0 and hud["steps"] < 3 * FPS:
        _text(frame, "FOOTAGE ENDED - dreaming now", (PANEL_X + 12, PANEL_Y + DREAM - 16), 0.62, WARN, 1, FONT)
    return frame


def play(session: RealDreamSession, record: str | None) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DreamDrive: Real World Edition")
    clock = pygame.time.Clock()
    writer = None
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(record, fps=FPS, codec="libx264",
                                    quality=8, pixelformat="yuv420p", macro_block_size=None)

    level, ghost = 0, False
    steer = 0.0
    speed = session.reset(level)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_a:
                    ghost = not ghost
                elif event.key == pygame.K_TAB:
                    level = (level + 1) % len(session.levels)
                    speed, steer, ghost = session.reset(level), 0.0, False
                elif event.key == pygame.K_r:
                    speed, steer, ghost = session.reset(level), 0.0, False

        keys = pygame.key.get_pressed()
        accel = float(keys[pygame.K_UP])
        decel = float(keys[pygame.K_DOWN])
        if ghost:
            g_steer, g_speed = session.ghost_action()
            steer += (g_steer - steer) * 0.5
            speed = g_speed
        else:
            t_steer = 0.5 * (float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT]))
            steer += (t_steer - steer) * 0.25
            speed = float(np.clip(speed + 0.02 * (accel - decel), -1.0, 1.0))

        t0 = time.perf_counter()
        dream64 = session.step(steer, speed)
        ms = (time.perf_counter() - t0) * 1e3

        frame = compose_rw(dream64, {
            "params": session.n_params, "ghost": ghost, "level": level,
            "n_levels": len(session.levels), "steps": session.steps,
            "fps": clock.get_fps(), "ms": ms, "device": str(session.device),
            "steer": steer, "accel": accel, "decel": decel, "speed": speed,
        })
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        if writer is not None:
            writer.append_data(frame)
        clock.tick(FPS)

    if writer is not None:
        writer.close()
        logger.info(f"Session recording saved to {record}")
    pygame.quit()


def selftest(session: RealDreamSession, out: str) -> None:
    """Headless scripted session per level: ghost, steer left, steer right, straight."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=None)
    times = []
    for level in range(len(session.levels)):
        speed = session.reset(level)
        steer = 0.0
        phases = [("ghost", 3 * FPS), ("left", 2 * FPS), ("right", 2 * FPS), ("straight", 3 * FPS)]
        for phase, n in phases:
            for _ in range(n):
                ghost = phase == "ghost"
                if ghost:
                    g_steer, g_speed = session.ghost_action()
                    steer += (g_steer - steer) * 0.5
                    speed = g_speed
                else:
                    target = -0.5 if phase == "left" else (0.5 if phase == "right" else 0.0)
                    steer += (target - steer) * 0.25
                t0 = time.perf_counter()
                dream64 = session.step(steer, speed)
                times.append((time.perf_counter() - t0) * 1e3)
                writer.append_data(compose_rw(dream64, {
                    "params": session.n_params, "ghost": ghost, "level": level,
                    "n_levels": len(session.levels), "steps": session.steps,
                    "fps": FPS, "ms": times[-1], "device": str(session.device),
                    "steer": steer, "accel": 0.0, "decel": 0.0, "speed": speed,
                }))
        logger.info(f"Level {level}: {sum(n for _, n in phases)} dream steps")
    writer.close()
    logger.info(f"Selftest: median {np.median(times):.1f} ms/frame "
                f"(max ~{1000 / np.median(times):.0f} fps). Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive a world model of real streets")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/comma")
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--selftest", type=str, default=None)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    session = RealDreamSession(args.checkpoint, args.data, device,
                               n_levels=args.levels, context=args.context, seed=args.seed)
    if args.selftest:
        selftest(session, args.selftest)
    else:
        play(session, args.record)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
