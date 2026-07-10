"""Convert a comma.ai (2016 'research' dataset) drive into video_world .npz episodes.

Input: one camera .h5 (dataset ``X``: (N, 3, 160, 320) uint8, 20 Hz) and the
matching log .h5 (100 Hz signals with ``cam1_ptr`` giving, for every log
sample, the index of the camera frame it belongs to).

Output: .npz episodes in the format video_world.py trains on —
    frames  uint8  (T, 64, 64, 3)   center-cropped square, resized
    actions float32 (T, 2)           [steering, speed], action[t] led INTO frame[t]

Steering (logged in degrees x 10) is normalised so 180 degrees of wheel angle
maps to |1| — urban turns saturate, highway corrections stay resolvable.
Speed is m/s mapped to [-1, 1] over 0..35 m/s. Chunks with mean speed
< 5 m/s (parking lots, long stops) are dropped.

Usage:
    PYTHONPATH=src uv run python src/dreamer/comma_convert.py \
        --camera data/comma_raw/camera_2016-06-08.h5 \
        --log data/comma_raw/log_2016-06-08.h5 \
        --out data/comma
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np

logger = logging.getLogger(__name__)

SIZE = 64
EPISODE_LEN = 1200  # 60 s at 20 Hz
MIN_MEAN_SPEED = 5.0  # m/s; drop chunks that are mostly standing still
STEER_SCALE = 1800.0  # raw log units are deg*10; 180 deg of wheel angle -> |1|
SPEED_SCALE = 35.0  # m/s mapped to action range


def frame_to_64(frame_chw: np.ndarray) -> np.ndarray:
    """(3, 160, 320) uint8 -> (64, 64, 3): center square crop, then resize."""
    img = np.transpose(frame_chw, (1, 2, 0))  # HWC, 160 x 320
    h, w = img.shape[:2]
    x0 = (w - h) // 2
    img = img[:, x0 : x0 + h]
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def per_frame_signals(log: h5py.File, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Average the 100 Hz steering/speed signals over each camera frame."""
    ptr = np.asarray(log["cam1_ptr"], dtype=np.int64)
    steer = np.asarray(log["steering_angle"], dtype=np.float64)
    speed = np.asarray(log["speed"], dtype=np.float64)
    valid = (ptr >= 0) & (ptr < n_frames)
    ptr, steer, speed = ptr[valid], steer[valid], speed[valid]

    counts = np.bincount(ptr, minlength=n_frames).astype(np.float64)
    steer_sum = np.bincount(ptr, weights=steer, minlength=n_frames)
    speed_sum = np.bincount(ptr, weights=speed, minlength=n_frames)
    counts[counts == 0] = 1.0
    return steer_sum / counts, speed_sum / counts


def convert(camera_path: Path, log_path: Path, out_dir: Path, max_frames: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(camera_path, "r") as cam, h5py.File(log_path, "r") as log:
        X = cam["X"]
        n = X.shape[0] if not max_frames else min(X.shape[0], max_frames)
        logger.info(f"{camera_path.name}: {X.shape[0]} frames at 20 Hz "
                    f"(~{X.shape[0] / 20 / 60:.0f} min), converting {n}")
        steer_deg, speed_ms = per_frame_signals(log, X.shape[0])

        ep = 0
        for start in range(0, n - EPISODE_LEN + 1, EPISODE_LEN):
            speeds = speed_ms[start : start + EPISODE_LEN]
            if speeds.mean() < MIN_MEAN_SPEED:
                logger.info(f"  chunk @{start}: mean speed {speeds.mean():.1f} m/s — skipped")
                continue
            frames = np.empty((EPISODE_LEN, SIZE, SIZE, 3), dtype=np.uint8)
            block = np.asarray(X[start : start + EPISODE_LEN])
            for t in range(EPISODE_LEN):
                frames[t] = frame_to_64(block[t])
            steer_n = np.clip(steer_deg[start : start + EPISODE_LEN] / STEER_SCALE, -1, 1)
            speed_n = np.clip(speeds / SPEED_SCALE, 0, 1) * 2 - 1
            actions = np.zeros((EPISODE_LEN, 2), dtype=np.float32)
            # action[t] led INTO frame[t]: controls measured at the previous frame.
            actions[1:, 0] = steer_n[:-1]
            actions[1:, 1] = speed_n[:-1]
            np.savez_compressed(out_dir / f"comma_{camera_path.stem}_{ep:03d}.npz",
                                frames=frames, actions=actions)
            ep += 1
            logger.info(f"  chunk @{start}: saved episode {ep} "
                        f"(mean speed {speeds.mean():.1f} m/s, "
                        f"|steer| p95 {np.percentile(np.abs(steer_n), 95):.2f})")
    logger.info(f"Done: {ep} episodes in {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="comma.ai h5 -> video_world npz")
    parser.add_argument("--camera", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/comma")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit for quick tests.")
    args = parser.parse_args()
    convert(Path(args.camera), Path(args.log), Path(args.out), args.max_frames)
