"""DreamDrive — play a game that doesn't exist.

An interactive window where a human drives *inside the trained world model*:
after a short warm-up on real frames to ground the latent state, the
environment is discarded entirely. Every frame you see is the RSSM prior
rolled forward one step with your keyboard action and decoded to pixels —
no game engine, no environment, pure imagination.

Controls:
    arrows      steer / gas / brake
    A           toggle autopilot (the trained actor takes the wheel)
    TAB         next dream level (a start latent grounded on a different track)
    R           restart current level
    ESC / Q     quit

Usage:
    PYTHONPATH=src uv run python src/dreamer/dreamdrive.py \
        --checkpoint saved_models/dreamer_CarRacing-v3_best.pt

    # record the session to an MP4 alongside playing:
    ... dreamdrive.py --checkpoint ... --record demo/dreamdrive_session.mp4

    # headless smoke test (no window, scripted inputs, writes a video):
    ... dreamdrive.py --checkpoint ... --selftest demo/dreamdrive_selftest.mp4
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

from dreamer.env import DreamerEnv
from dreamer.models import Dreamer

logger = logging.getLogger(__name__)

# CarRacing runs natively at 50 fps and the dream timestep is 2 frames
# (action_repeat), so a 25 fps play loop makes dream time exactly real time.
W, H, FPS = 1280, 720, 25
DREAM = 620  # dream panel side length
PAD = 50
PANEL_X, PANEL_Y = PAD, (H - DREAM) // 2
SIDE_X = PANEL_X + DREAM + 48

BG = (15, 17, 21)
FG = (235, 238, 242)
DIM = (150, 158, 170)
ACCENT = (110, 231, 150)
WARN = (255, 120, 110)
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.6, color=FG, thickness=1, font=FONT_S):
    s = s.replace("·", "|").replace("—", "-")
    cv2.putText(img, s, (org[0] + 1, org[1] + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, color, thickness, cv2.LINE_AA)


class DreamSession:
    """The world-model-only game state: latent, decode, reward — no environment."""

    def __init__(self, checkpoint: str, env_name: str, device: torch.device,
                 level_seeds: list[int], context: int = 20, action_repeat: int = 2) -> None:
        self.device = device
        self.agent = Dreamer.load(Path(checkpoint), device)
        self.agent.eval()
        self.wm = self.agent.wm
        self.rssm = self.agent.wm.rssm
        self.n_params = sum(p.numel() for p in self.agent.parameters())
        self.env_name = env_name
        self.level_seeds = level_seeds
        self.context = context
        self.action_repeat = action_repeat
        self.action_dim = -1
        self.levels: list[dict[str, torch.Tensor]] = []
        self._build_levels()
        self.state: dict[str, torch.Tensor] = {}
        self.steps = 0

    @torch.no_grad()
    def _ground(self, seed: int, drive_steps: int, coast_steps: int = 8) -> dict[str, torch.Tensor]:
        """Ground a start latent: the trained actor drives ``drive_steps`` into
        the real track (following real curves), then lifts off the gas for a few
        steps so the player spawns in a slow car — then the env is discarded."""
        env = DreamerEnv(self.env_name, action_repeat=self.action_repeat,
                         seed=seed, render_mode=None)
        if self.action_dim < 0:
            self.action_dim = env.action_dim
        obs = env.reset()
        state = self.rssm.initial(1, self.device)
        prev_action = torch.zeros(1, env.action_dim, device=self.device)
        is_first = torch.tensor([1.0], device=self.device)
        for t in range(drive_steps + coast_steps):
            embed = self.wm.encode(
                self.wm.preprocess(torch.as_tensor(obs, device=self.device).unsqueeze(0))
            )
            state, _ = self.rssm.obs_step(state, prev_action, embed, is_first)
            if t < drive_steps:
                action = self.agent.actor(self.rssm.get_feat(state)).mode()
            else:
                # No brake: braking to a crawl mid-track is off-distribution
                # and destabilises the dream.
                coast = -np.ones(env.action_dim, dtype=np.float32)
                coast[0] = 0.0
                action = torch.as_tensor(coast, device=self.device).unsqueeze(0)
            obs, _, term, trunc, _ = env.step(action[0].cpu().numpy())
            prev_action = action
            is_first = torch.tensor([0.0], device=self.device)
            if term or trunc:
                break
        env.close()
        return {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def _dream_holds(self, state: dict[str, torch.Tensor], steps: int = 75) -> bool:
        """Can the autopilot keep the dream on-road from this start latent?"""
        s = {k: v.clone() for k, v in state.items()}
        rewards = []
        for _ in range(steps):
            feat = self.rssm.get_feat(s)
            action = self.agent.actor(feat).mode()
            s = self.rssm.img_step(s, action)
            rewards.append(float(self.wm.reward_head(self.rssm.get_feat(s)).mean().item()))
        return max(rewards[-50:]) > 0.0

    @torch.no_grad()
    def _build_levels(self) -> None:
        """One validated start latent per seed, at varying depths into the track.
        A candidate latent is kept only if the autopilot can hold the dream
        together from it; otherwise we re-ground a little deeper and retry."""
        for i, seed in enumerate(self.level_seeds):
            depth = 2 * self.context + 25 * i
            state = None
            for attempt in range(3):
                state = self._ground(seed, depth + 20 * attempt)
                if self._dream_holds(state):
                    logger.info(f"Level {i}: seed {seed}, {depth + 20 * attempt} steps in — validated")
                    break
                logger.info(f"Level {i}: seed {seed}, {depth + 20 * attempt} steps in — unstable, retrying deeper")
            else:
                logger.warning(f"Level {i}: no stable grounding found for seed {seed}; keeping last")
            assert state is not None
            self.levels.append(state)

    def reset(self, level: int) -> None:
        self.state = {k: v.clone() for k, v in self.levels[level % len(self.levels)].items()}
        self.steps = 0

    @torch.no_grad()
    def step(self, action_vec: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Advance the dream one step. Returns (frame64 rgb u8, imagined reward, action)."""
        action = torch.as_tensor(action_vec, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        self.state = self.rssm.img_step(self.state, action)
        feat = self.rssm.get_feat(self.state)
        img = self.wm.decode_obs(self.wm.decoder(feat))[0].cpu().numpy()
        reward = float(self.wm.reward_head(feat).mean().item())
        self.steps += 1
        return (np.clip(img, 0, 1) * 255).astype(np.uint8), reward, action_vec

    @torch.no_grad()
    def autopilot_action(self) -> np.ndarray:
        feat = self.rssm.get_feat(self.state)
        return self.agent.actor(feat).mode()[0].cpu().numpy()


# ------------------------------------------------------------------- rendering


def _draw_arrow_pad(frame: np.ndarray, x: int, y: int, steer: float, gas: float,
                    brake: float) -> None:
    """Visual keyboard: arrows light up with the applied (smoothed) controls."""
    s = 34  # arrow cell size

    def cell(cx, cy, active, shape):
        color = ACCENT if active > 0.15 else (55, 60, 70)
        if shape == "up":
            pts = [(cx, cy - s // 2), (cx - s // 2, cy + s // 2), (cx + s // 2, cy + s // 2)]
        elif shape == "down":
            pts = [(cx, cy + s // 2), (cx - s // 2, cy - s // 2), (cx + s // 2, cy - s // 2)]
        elif shape == "left":
            pts = [(cx - s // 2, cy), (cx + s // 2, cy - s // 2), (cx + s // 2, cy + s // 2)]
        else:
            pts = [(cx + s // 2, cy), (cx - s // 2, cy - s // 2), (cx - s // 2, cy + s // 2)]
        cv2.fillPoly(frame, [np.array(pts, np.int32)], color, cv2.LINE_AA)

    cell(x + 60, y, gas, "up")
    cell(x + 60, y + 90, brake, "down")
    cell(x, y + 45, max(-steer, 0.0), "left")
    cell(x + 120, y + 45, max(steer, 0.0), "right")


def compose_frame(dream64: np.ndarray, hud: dict, flash: str = "") -> np.ndarray:
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[:] = BG

    # Dream panel: nearest-neighbour upscale — honest, crunchy neural pixels.
    panel = cv2.resize(dream64, (DREAM, DREAM), interpolation=cv2.INTER_NEAREST)
    frame[PANEL_Y : PANEL_Y + DREAM, PANEL_X : PANEL_X + DREAM] = panel
    cv2.rectangle(frame, (PANEL_X - 2, PANEL_Y - 2),
                  (PANEL_X + DREAM + 1, PANEL_Y + DREAM + 1), ACCENT, 2)
    if flash:
        overlay = frame[PANEL_Y : PANEL_Y + DREAM, PANEL_X : PANEL_X + DREAM]
        overlay[:] = (overlay * 0.35).astype(np.uint8)
        _text(frame, "YOU LEFT THE KNOWN WORLD", (PANEL_X + 60, PANEL_Y + DREAM // 2 - 20),
              0.95, WARN, 2, FONT)
        _text(frame, flash, (PANEL_X + 60, PANEL_Y + DREAM // 2 + 24), 0.55, FG)

    _text(frame, "DREAMDRIVE", (SIDE_X, 92), 1.15, FG, 2, FONT)
    _text(frame, "a game that does not exist", (SIDE_X, 128), 0.62, DIM)
    _text(frame, f"every pixel is generated by a {hud['params'] / 1e6:.0f}M-param", (SIDE_X, 168), 0.52, DIM)
    _text(frame, "world model | no game engine | no environment", (SIDE_X, 192), 0.52, DIM)

    mode = "AUTOPILOT (the agent drives its own dream)" if hud["autopilot"] else "MANUAL - you are driving"
    _text(frame, mode, (SIDE_X, 244), 0.62, ACCENT if hud["autopilot"] else FG, 1, FONT)
    _text(frame, f"level {hud['level'] + 1}/{hud['n_levels']}   dream step {hud['steps']}", (SIDE_X, 278), 0.55, DIM)
    _text(frame, f"{hud['fps']:4.0f} fps | {hud['ms']:4.1f} ms/frame | {hud['device']}", (SIDE_X, 306), 0.55, DIM)

    _draw_arrow_pad(frame, SIDE_X + 40, 380, hud["steer"], hud["gas"], hud["brake"])

    # Imagined-reward sparkline: the model's own belief about how well you drive.
    hist = hud["reward_hist"][-240:]
    x0, y0, w, h = SIDE_X, 530, W - SIDE_X - PAD, 70
    if len(hist) >= 2:
        lo, hi = min(hist), max(hist)
        span = max(hi - lo, 1e-6)
        pts = [(int(x0 + i / (len(hist) - 1) * (w - 1)),
                int(y0 + (1 - (v - lo) / span) * (h - 8)) + 4) for i, v in enumerate(hist)]
        cv2.polylines(frame, [np.array(pts, np.int32)], False, ACCENT, 2, cv2.LINE_AA)
    _text(frame, f"imagined reward {hud['reward']:+5.2f}", (x0, y0 + h + 26), 0.55, ACCENT)
    _text(frame, "(predicted by the world model itself)", (x0, y0 + h + 50), 0.45, DIM)

    _text(frame, "arrows drive | A autopilot | TAB level | R restart | ESC quit",
          (SIDE_X, H - 46), 0.48, DIM)
    return frame


# ------------------------------------------------------------------ interactive


def play(session: DreamSession, record: str | None) -> None:
    import pygame  # imported here so --selftest never needs a display

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("DreamDrive - a game that does not exist")
    clock = pygame.time.Clock()

    writer = None
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(record, fps=FPS, codec="libx264",
                                    quality=8, pixelformat="yuv420p", macro_block_size=None)

    level, autopilot = 0, False
    steer = gas = brake = 0.0
    reward_hist: list[float] = []
    session.reset(level)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_a:
                    autopilot = not autopilot
                elif event.key == pygame.K_TAB:
                    level = (level + 1) % len(session.levels)
                    session.reset(level)
                    reward_hist.clear()
                elif event.key == pygame.K_r:
                    session.reset(level)
                    reward_hist.clear()

        keys = pygame.key.get_pressed()
        # Caps and smoothing tuned for keyboard play at the dream timestep
        # (2 env frames/step): full-lock steering or full throttle put the car
        # beyond controllable speed in under a second.
        t_steer = 0.6 * (float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT]))
        t_gas = 0.55 * float(keys[pygame.K_UP])
        t_brake = float(keys[pygame.K_DOWN])
        steer += (t_steer - steer) * 0.22
        gas += (t_gas - gas) * (0.12 if t_gas > gas else 0.45)  # slow ramp, quick release
        brake += (t_brake - brake) * 0.35

        if autopilot:
            action = session.autopilot_action()
            steer, gas, brake = float(action[0]), (action[1] + 1) / 2, (action[2] + 1) / 2
        else:
            # Policy action space is [-1, 1]^3 (affine-mapped to env bounds).
            action = np.array([steer, gas * 2 - 1, brake * 2 - 1], dtype=np.float32)

        t0 = time.perf_counter()
        dream64, reward, _ = session.step(action)
        ms = (time.perf_counter() - t0) * 1e3
        reward_hist.append(reward)

        # Off-road detection: the world model's own reward head flatlines
        # negative once the car is lost in imagined grass. Crash -> next level.
        # Only armed once the player has actually been on the road this level
        # (positive imagined reward), so idling at spawn is safe.
        flash = ""
        window = reward_hist[-int(3.0 * FPS):]
        armed = any(r > 0.0 for r in reward_hist)
        if armed and len(window) >= int(3.0 * FPS) and max(window) < 0.0 and not autopilot:
            flash = "the model forgets roads it cannot see | dream restarting..."
            frame = compose_frame(dream64, {
                "params": session.n_params, "autopilot": autopilot, "level": level,
                "n_levels": len(session.levels), "steps": session.steps,
                "fps": clock.get_fps(), "ms": ms, "device": str(session.device),
                "steer": steer, "gas": gas, "brake": brake,
                "reward": reward, "reward_hist": reward_hist,
            }, flash=flash)
            for _ in range(int(1.2 * FPS)):
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                if writer is not None:
                    writer.append_data(frame)
                clock.tick(FPS)
            level = (level + 1) % len(session.levels)
            session.reset(level)
            reward_hist.clear()
            steer = gas = brake = 0.0
            continue

        frame = compose_frame(dream64, {
            "params": session.n_params, "autopilot": autopilot, "level": level,
            "n_levels": len(session.levels), "steps": session.steps,
            "fps": clock.get_fps(), "ms": ms, "device": str(session.device),
            "steer": steer, "gas": gas, "brake": brake,
            "reward": reward, "reward_hist": reward_hist,
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


# -------------------------------------------------------------------- selftest


def selftest(session: DreamSession, out: str, seconds: int = 12) -> None:
    """Headless coherence test: the trained actor drives its own dream on every
    level. Reports how long the dream stays 'on road' (positive imagined reward
    within a rolling window) — the practical limit of a play session."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=None)
    n = seconds * FPS
    times = []
    for level in range(len(session.levels)):
        session.reset(level)
        reward_hist: list[float] = []
        lost_at = n
        for i in range(n):
            action = session.autopilot_action()
            steer, gas, brake = float(action[0]), (action[1] + 1) / 2, (action[2] + 1) / 2
            t0 = time.perf_counter()
            dream64, reward, _ = session.step(action)
            times.append((time.perf_counter() - t0) * 1e3)
            reward_hist.append(reward)
            window = reward_hist[-int(2.0 * FPS):]
            if lost_at == n and len(window) == int(2.0 * FPS) and max(window) < 0.0:
                lost_at = i
            writer.append_data(compose_frame(dream64, {
                "params": session.n_params, "autopilot": True, "level": level,
                "n_levels": len(session.levels), "steps": session.steps,
                "fps": FPS, "ms": times[-1], "device": str(session.device),
                "steer": steer, "gas": gas, "brake": brake,
                "reward": reward, "reward_hist": reward_hist,
            }))
        status = "coherent throughout" if lost_at == n else f"lost road at step {lost_at} (~{lost_at / FPS:.1f}s)"
        logger.info(f"Level {level}: {status}")
    writer.close()
    logger.info(
        f"Selftest: median {np.median(times):.1f} ms/frame "
        f"(max sustainable ~{1000 / np.median(times):.0f} fps). Saved {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive inside the world model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", type=str, default="CarRacing-v3")
    parser.add_argument("--levels", type=int, default=5, help="Number of dream start states.")
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--record", type=str, default=None, help="Record the session to this MP4.")
    parser.add_argument("--selftest", type=str, default=None,
                        help="Run a headless scripted drive and save it to this MP4.")
    args = parser.parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    session = DreamSession(
        args.checkpoint, args.env, device,
        level_seeds=[args.seed_base + i for i in range(args.levels)],
        context=args.context,
    )
    if args.selftest:
        selftest(session, args.selftest)
    else:
        play(session, args.record)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
