# 3080 Ti Training Runbook — Real-World Video World Model

Goal: train the comma.ai video world model at scale on a 3080 Ti (12 GB), then
**bring the checkpoint back to the MacBook** for the interactive demo
(`dreamdrive_rw.py`) and renders. Train on a gaming GPU, deploy on a laptop —
no cloud anywhere in the pipeline.

Everything below Phase A runs on **already-committed code** — it can be executed
tonight without touching a line. Phase B lists the upgrades to ask Claude for
next session.

---

## 0. Machine setup (once)

Ubuntu 22.04/24.04 (or Windows + WSL2). Then:

```bash
nvidia-smi                        # driver present, shows the 3080 Ti
git clone git@github.com:Larwive/Enhanced-World-Model.git
cd Enhanced-World-Model && git checkout feature/dreamer-v3
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv missing
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# must print: True NVIDIA GeForce RTX 3080 Ti
```

Disk: ~60 GB free for raw drives + converted data + checkpoints.

## 1. Fetch the comma drives (minutes, not hours)

Do **not** use `curl` against `archive.org/download/...zip/<inner file>` — that
endpoint streams the whole inner file with no Range support, silently truncates
around 1.7–3.2 GB, and curl still exits 0. (Two days were lost to this.)

Use ranged reads against the zip itself via `remotezip` (already a dependency):

```bash
mkdir -p data/comma_raw
uv run python - <<'EOF'
import shutil
from remotezip import RemoteZip

URL = "https://archive.org/download/comma-dataset/comma-dataset.zip"
# Daytime drives (name: approx camera size). The three marked * are already on
# the MacBook — re-fetching here is simplest.
DRIVES = [
    "2016-01-30--11-24-51",   # 8.2G day
    "2016-01-30--13-46-00",   # 9.1G day
    "2016-02-02--10-16-58",   # 8.7G day
    "2016-03-29--10-50-20",   # 12.1G day
    "2016-02-08--14-56-28",   # 4.1G day *
    "2016-04-21--14-48-08",   # 4.7G day *
    "2016-06-08--11-46-01",   # 2.8G day *
    # optional night flavor: "2016-02-11--21-32-47" (13.2G), "2016-06-02--21-39-29" (6.9G)
]
with RemoteZip(URL) as z:
    for d in DRIVES:
        for kind in ("camera", "log"):
            out = f"data/comma_raw/{kind}_{d}.h5"
            print("fetching", kind, d)
            with z.open(f"{kind}/{d}.h5") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
EOF

# Verify EVERY file actually opens (file size alone proves nothing):
uv run python - <<'EOF'
import glob, h5py
for f in sorted(glob.glob("data/comma_raw/*.h5")):
    try: h5py.File(f, "r").close(); print("OK ", f)
    except Exception as e: print("BAD", f, e)
EOF
```

## 2. Convert to 128px episodes

```bash
for cam in data/comma_raw/camera_*.h5; do
  log="${cam/camera_/log_}"
  PYTHONPATH=src uv run python src/dreamer/comma_convert.py \
      --camera "$cam" --log "$log" --out data/comma128
done
du -sh data/comma128   # expect roughly 15–20 GB for the 7 daytime drives
```

Notes baked into the converter: steering in the logs is **degrees × 10**
(normalised /1800), CAN is aligned to frames via `cam1_ptr`, slow/stopped
chunks are KEPT (a model that never saw a stationary world dissolves when the
player brakes), only fully-parked chunks are dropped.

RAM budget: the loader holds all frames in RAM — ~49 KB/frame at 128px.
The 7 daytime drives ≈ 340k frames ≈ **17 GB RAM**. Fine on a 32 GB box; if
tight, drop a drive or two.

## 3. Train (overnight)

```bash
mkdir -p saved_models/comma_wm128_gpu
PYTHONPATH=src nohup uv run python src/dreamer/video_world.py --train \
    --data data/comma128 --out saved_models/comma_wm128_gpu \
    --steps 60000 --batch-size 32 --seq-len 32 \
    --deter-dim 512 --cnn-depth 48 --perceptual 20 \
    > train_gpu.log 2>&1 &
echo $! > train_gpu.pid
tail -f train_gpu.log
```

- Device auto-selects CUDA. Expect ~2–3 steps/s at batch 32 (vs 0.6 on the M5
  at batch 8 — ~10–15× frame throughput). 60k steps ≈ 6–8 h.
- `--perceptual 20` is the VGG sharpness loss (first run downloads VGG16 weights).
- Checkpoints land every 1000 steps: `video_wm_step<N>.pt`, final `video_wm_final.pt`.
- Watch `recon` and `percep` fall; `kl_dyn` settling under ~6 is when
  steering-conditioning gets crisp (observed on the M5 runs).
- If VRAM overflows (unlikely at these sizes): `--batch-size 16`.

## 4. Preview while it trains (optional)

```bash
PYTHONPATH=src uv run python src/dreamer/video_world.py --dream \
    --checkpoint saved_models/comma_wm128_gpu/video_wm_step10000.pt \
    --data data/comma128 --seed 4 --horizon 30
# writes demo/video_dream.mp4: real future | dream | steer-left | steer-right
```

## 5. Bring the checkpoint home (MacBook)

```bash
# from the Mac:
scp <gpu-box>:~/Enhanced-World-Model/saved_models/comma_wm128_gpu/video_wm_final.pt \
    saved_models/comma_wm128_gpu/

# renders + the interactive demo (all resolution-agnostic, load fine on MPS):
PYTHONPATH=src uv run python src/dreamer/video_world.py --dream \
    --checkpoint saved_models/comma_wm128_gpu/video_wm_final.pt --data data/comma128 --seed 4
PYTHONPATH=src uv run python src/dreamer/dreamdrive_rw.py \
    --checkpoint saved_models/comma_wm128_gpu/video_wm_final.pt --data data/comma128 \
    --record demo/dreamdrive_rw_128.mp4
```

(The Mac needs `data/comma128` for grounding contexts — the 3 drives already
converted there are enough; no need to copy the GPU box's full dataset.)

## Phase B — upgrades for the next iteration (ask Claude)

1. `--size` flag on `comma_convert.py` + `ConvEncoder256`/`ConvDecoder256`
   (same 5→6-stage pattern as the 128 pair) → 256px training.
2. Mixed precision (AMP) in `train_video_wm` → ~2× throughput on Ampere.
3. Memory-mapped episode loader → full 11-drive / 256px datasets without
   fitting everything in RAM.
4. Bigger model (`--deter-dim 1024`, `--cnn-depth 64`) once AMP is in.

## Known gotchas

- **curl vs archive.org**: see §1. Always verify h5 files by opening them.
- The in-RAM loader will get OOM-killed silently if the dataset exceeds RAM —
  check `dmesg | tail` if training dies without a traceback.
- Keep `saved_models/comma_wm128_gpu` (GPU runs) separate from
  `saved_models/comma_wm128` (M5 runs) so nothing gets clobbered.
- MPS-vs-CUDA nondeterminism is expected: the same checkpoint produces slightly
  different rollouts on the two backends; judgments should be made from renders,
  not bitwise comparisons.
