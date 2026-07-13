"""Action-conditioned video world model — the RSSM on offline driving video.

No environment and no policy: the world model is trained directly on recorded
(frames, actions) sequences, then rolled forward in imagination to *dream video
continuations conditioned on steering*. This is the GAIA/Waymo world-model idea
at laptop scale — the pipeline is data-source agnostic, so real dashcam data
(steering/speed from a CAN log) drops into the same .npz format.

Episode format (one .npz per episode):
    frames  uint8  (T, 64, 64, 3)
    actions float32 (T, A)   action[t] led INTO frame[t]; actions[0] is zeros

Usage:
    # 1. generate a procedural toy driving dataset (pipeline validation)
    PYTHONPATH=src uv run python src/dreamer/video_world.py --make-toy data/toy_drive

    # 2. train the world model on it
    PYTHONPATH=src uv run python src/dreamer/video_world.py --train --data data/toy_drive \
        --out saved_models/video_wm --steps 2000 --device cpu

    # 3. dream counterfactual continuations (recorded vs steer-left vs steer-right)
    PYTHONPATH=src uv run python src/dreamer/video_world.py --dream \
        --checkpoint saved_models/video_wm/video_wm_final.pt --data data/toy_drive \
        --out demo/video_dream
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
import torch.nn as nn

from dreamer.models import Dreamer, DreamerConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- toy data


def make_toy_dataset(out_dir: Path, episodes: int = 20, length: int = 300, size: int = 64) -> None:
    """Procedural 'driving' video: a curving road viewed from a steerable car.

    Causal structure matches real dashcam data: the road curves on its own
    (world dynamics), steering shifts the car laterally (action-conditioned),
    so a correct world model must dream different futures for different actions.
    """
    rng = np.random.default_rng(7)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(episodes):
        frames = np.zeros((length, size, size, 3), dtype=np.uint8)
        actions = np.zeros((length, 2), dtype=np.float32)  # [steer, speed]
        curve_phase = rng.uniform(0, 2 * np.pi)
        curve_freq = rng.uniform(0.02, 0.06)
        car_x = 0.0  # lateral offset relative to road center, in [-1, 1]
        prev_steer = 0.0
        for t in range(length):
            road_center = 0.55 * np.sin(curve_phase + curve_freq * t)
            # A noisy 'driver' that mostly tracks the road:
            steer = np.clip(0.9 * (road_center - car_x) + rng.normal(0, 0.15), -1, 1)
            steer = 0.7 * prev_steer + 0.3 * steer
            prev_steer = steer
            car_x = np.clip(car_x + 0.10 * steer, -1.2, 1.2)

            img = np.zeros((size, size, 3), np.uint8)
            img[:] = (60, 160, 60)  # grass
            # Road: a band whose horizontal position interpolates from the
            # car-relative position (bottom) toward upcoming curvature (top).
            for row in range(size):
                depth = 1 - row / size  # 1 = horizon, 0 = car
                center_here = road_center + depth * 0.4 * np.sin(
                    curve_phase + curve_freq * (t + 40 * depth)
                )
                cx = int((center_here - car_x) * size * 0.4 + size / 2)
                half = int(size * (0.10 + 0.10 * (1 - depth)))
                img[row, max(cx - half, 0) : min(cx + half, size)] = (110, 110, 110)
            # Car marker at bottom center.
            img[size - 10 : size - 3, size // 2 - 3 : size // 2 + 3] = (200, 40, 40)

            frames[t] = img
            if t + 1 < length:
                actions[t + 1] = (steer, 1.0)
        np.savez_compressed(out_dir / f"episode_{ep:03d}.npz", frames=frames, actions=actions)
    logger.info(f"Wrote {episodes} toy episodes of {length} frames to {out_dir}")


# --------------------------------------------------------------------- loader


class VideoSequences:
    """Flat concatenation of episodes with is_first flags; samples (B, L) windows."""

    def __init__(self, data_dir: Path) -> None:
        frames, actions, is_first = [], [], []
        files = sorted(data_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz episodes in {data_dir}")
        for f in files:
            ep = np.load(f)
            frames.append(ep["frames"])
            actions.append(ep["actions"].astype(np.float32))
            flags = np.zeros(len(ep["frames"]), dtype=bool)
            flags[0] = True
            is_first.append(flags)
        self.frames = np.concatenate(frames)
        self.actions = np.concatenate(actions)
        self.is_first = np.concatenate(is_first)
        self.action_dim = self.actions.shape[1]
        self.obs_shape = self.frames.shape[1:]
        logger.info(
            f"Loaded {len(files)} episodes, {len(self.frames)} frames total, "
            f"action_dim={self.action_dim}"
        )

    def _start_probs(self, seq_len: int) -> np.ndarray:
        """Window-start probabilities oversampling high-|steer| moments.

        The recorded driver goes mostly straight, so uniform sampling starves
        steering dynamics (the dream's steering feels mushy). Weight each
        window by baseline + mean |steer|, giving turns a few times the mass."""
        if getattr(self, "_probs_len", None) != seq_len:
            cs = np.concatenate([[0.0], np.cumsum(np.abs(self.actions[:, 0]))])
            n = len(self.frames) - seq_len
            w = 0.15 + (cs[seq_len:seq_len + n] - cs[:n]) / seq_len
            self._probs = w / w.sum()
            self._probs_len = seq_len
        return self._probs

    def sample(self, batch_size: int, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
        n = len(self.frames) - seq_len
        starts = np.random.choice(n, size=batch_size, p=self._start_probs(seq_len))
        idx = starts[:, None] + np.arange(seq_len)[None, :]
        zeros = np.zeros(idx.shape, dtype=np.float32)
        return {
            "obs": torch.as_tensor(self.frames[idx], device=device),
            "action": torch.as_tensor(self.actions[idx], device=device),
            "reward": torch.as_tensor(zeros, device=device),
            "is_first": torch.as_tensor(self.is_first[idx], device=device).float(),
            "is_terminal": torch.as_tensor(zeros, device=device),
        }


# -------------------------------------------------------------------- trainer


class VGGPerceptual(nn.Module):
    """VGG16-feature perceptual loss on decoded frames.

    MSE decoders average over uncertain futures, which reads as blur; matching
    VGG features pushes the decoder toward sharp, texture-consistent frames.
    Applied to a random subset of the batch's frames to bound the cost."""

    def __init__(self, device: torch.device, weight: float, frac: float = 0.25) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.weight = weight
        self.frac = frac

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # recon/target are decoder-space (image - 0.5); map to [0, 1].
        n = recon.shape[0]
        k = max(1, int(n * self.frac))
        idx = torch.randperm(n, device=recon.device)[:k]
        pred = ((recon[idx] + 0.5).clamp(0, 1) - self.mean) / self.std
        tgt = ((target[idx] + 0.5).clamp(0, 1) - self.mean) / self.std
        return self.weight * torch.nn.functional.mse_loss(self.vgg(pred), self.vgg(tgt.detach()))


def train_video_wm(
    data_dir: Path,
    out_dir: Path,
    steps: int,
    device: torch.device,
    batch_size: int = 16,
    seq_len: int = 32,
    deter_dim: int = 256,
    cnn_depth: int = 32,
    perceptual: float = 0.0,
    log_every: int = 100,
    save_every: int = 1000,
) -> Path:
    data = VideoSequences(data_dir)
    is_128 = data.obs_shape[0] == 128
    cfg = DreamerConfig(
        obs_shape=tuple(data.obs_shape),
        action_dim=data.action_dim,
        is_image=True,
        is_discrete=False,
        deter_dim=deter_dim,
        cnn_depth=cnn_depth,
        encoder="ConvEncoder128" if is_128 else "",
        decoder="ConvDecoder128" if is_128 else "",
    )
    # Full Dreamer container so existing save/load & demo tooling work;
    # the actor/critic simply stay untrained.
    agent = Dreamer(cfg).to(device)
    n_wm = sum(p.numel() for p in agent.wm.parameters())
    logger.info(f"Video world model: {n_wm / 1e6:.2f}M parameters on {device}")
    opt = torch.optim.Adam(agent.wm.parameters(), lr=cfg.model_lr, eps=cfg.eps)
    recon_hook = VGGPerceptual(device, weight=perceptual) if perceptual > 0 else None
    if recon_hook is not None:
        logger.info(f"Perceptual sharpness loss enabled (weight {perceptual})")

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for step in range(1, steps + 1):
        batch = data.sample(batch_size, seq_len, device)
        loss, metrics, _ = agent.wm.loss(batch, recon_hook=recon_hook)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.wm.parameters(), cfg.grad_clip)
        opt.step()
        if step % log_every == 0:
            sps = step / (time.time() - start)
            logger.info(
                f"step {step:>6d} | recon {metrics.get('wm/recon', float('nan')):8.3f} | "
                f"kl_dyn {metrics.get('wm/kl_dyn', float('nan')):6.3f} | "
                f"percep {metrics.get('wm/recon_hook', float('nan')):7.3f} | {sps:4.1f} steps/s"
            )
        if step % save_every == 0:
            agent.save(out_dir / f"video_wm_step{step}.pt")
    final = out_dir / "video_wm_final.pt"
    agent.save(final)
    logger.info(f"Saved {final}")
    return final


# ------------------------------------------------------------------- dreaming


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


@torch.no_grad()
def dream_counterfactuals(
    checkpoint: Path,
    data_dir: Path,
    out: Path,
    device: torch.device,
    context: int = 15,
    horizon: int = 40,
    steer_bias: float = 0.6,
    panel: int = 192,
    fps: int = 12,
    seed: int = 0,
) -> None:
    """Ground on real frames, then dream three futures: recorded actions,
    steer-left, steer-right. If the model learned action conditioning, the
    dreamed roads must diverge."""
    agent = Dreamer.load(checkpoint, device)
    agent.eval()
    wm, rssm = agent.wm, agent.wm.rssm
    data = VideoSequences(data_dir)

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(data.frames) - (context + horizon)))
    frames = data.frames[start : start + context + horizon]
    actions = data.actions[start : start + context + horizon]

    # Ground the posterior on the context.
    state = rssm.initial(1, device)
    prev_action = torch.zeros(1, data.action_dim, device=device)
    is_first = torch.tensor([1.0], device=device)
    for t in range(context):
        obs_t = torch.as_tensor(frames[t], device=device).unsqueeze(0)
        embed = wm.encode(wm.preprocess(obs_t))
        state, _ = rssm.obs_step(state, prev_action, embed, is_first)
        prev_action = torch.as_tensor(actions[t + 1], device=device).unsqueeze(0)
        is_first = torch.tensor([0.0], device=device)

    variants: dict[str, dict] = {
        "recorded actions": {"bias": 0.0},
        f"steer left {-steer_bias:+.1f}": {"bias": -steer_bias},
        f"steer right {+steer_bias:+.1f}": {"bias": +steer_bias},
    }
    for v in variants.values():
        v["state"] = {k: t.clone() for k, t in state.items()}
        v["frames"] = []

    for t in range(horizon):
        base = actions[min(context + t, len(actions) - 1)].copy()
        for v in variants.values():
            a = base.copy()
            a[0] = np.clip(a[0] + v["bias"], -1, 1)
            action = torch.as_tensor(a, device=device).unsqueeze(0)
            v["state"] = rssm.img_step(v["state"], action)
            feat = rssm.get_feat(v["state"])
            img = wm.decode_obs(wm.decoder(feat))[0].cpu().numpy()
            v["frames"].append((np.clip(img, 0, 1) * 255).astype(np.uint8))

    # Compose: real future | dream(recorded) | dream(left) | dream(right)
    out.parent.mkdir(parents=True, exist_ok=True)
    video = []
    names = ["real future"] + list(variants)
    for t in range(horizon):
        cols = [frames[context + t]] + [v["frames"][t] for v in variants.values()]
        row = []
        for name, img in zip(names, cols):
            p = cv2.resize(img, (panel, panel), interpolation=cv2.INTER_NEAREST)
            _label(p, name)
            row.append(p)
        video.append(np.concatenate(row, axis=1))
    imageio.mimsave(out.with_suffix(".mp4"), video, fps=fps, codec="libx264",
                    quality=8, pixelformat="yuv420p", macro_block_size=None)
    imageio.mimsave(out.with_suffix(".gif"), video[::2], duration=2 / fps)
    logger.info(f"Saved {out.with_suffix('.mp4')} and .gif ({horizon} dream steps)")


# ------------------------------------------------------------------------ cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Video world model on offline driving data")
    parser.add_argument("--make-toy", type=str, default=None, metavar="DIR")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dream", action="store_true")
    parser.add_argument("--data", type=str, default="data/toy_drive")
    parser.add_argument("--out", type=str, default="saved_models/video_wm")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--deter-dim", type=int, default=256)
    parser.add_argument("--cnn-depth", type=int, default=32)
    parser.add_argument("--perceptual", type=float, default=0.0,
                        help="Weight of the VGG perceptual sharpness loss (0 = off; try 20).")
    parser.add_argument("--context", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="",
                        help="cpu | mps | cuda (default: auto). Use cpu while another run owns the GPU.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

    if args.make_toy:
        make_toy_dataset(Path(args.make_toy))
    if args.train:
        train_video_wm(
            Path(args.data), Path(args.out), args.steps, device,
            batch_size=args.batch_size, seq_len=args.seq_len,
            deter_dim=args.deter_dim, cnn_depth=args.cnn_depth,
            perceptual=args.perceptual,
        )
    if args.dream:
        ckpt = Path(args.checkpoint or (Path(args.out) / "video_wm_final.pt"))
        dream_counterfactuals(
            ckpt, Path(args.data), Path("demo/video_dream"), device,
            context=args.context, horizon=args.horizon, seed=args.seed,
        )
    if not (args.make_toy or args.train or args.dream):
        parser.error("Pick at least one of --make-toy / --train / --dream")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
