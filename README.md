# Enhanced World Model

A from-scratch, modular implementation of **model-based reinforcement learning with latent
imagination** — a compact **DreamerV3** ([Hafner et al., 2023](https://arxiv.org/abs/2301.04104))
built in the spirit of Ha & Schmidhuber's *World Models* (2018). Agents learn a **world
model** of their environment and then train their policy by *dreaming* inside it.

Everything here was trained on an **Apple-Silicon MacBook Air (MPS — no CUDA)**.

<p align="center"><img src="assets/carracing_dream.gif" width="78%"></p>

> **The agent driving inside a world it imagines.** After a few real frames the environment
> is switched off; the right panel is generated entirely from the model's latent state — it
> hallucinates the track, the car and the grass, and the policy keeps driving inside that dream.

---

## Highlights

- **DreamerV3-lite from scratch** — a categorical RSSM world model (KL balancing + free
  bits, symlog two-hot reward/value heads) with an actor-critic trained **purely on imagined
  latent rollouts**. ~11 M params, MPS-friendly (no `multinomial` / `quantile` / `bucketize`).
- **CarRacing-v3 from pixels:** mean return **≈ 632** over 5 episodes (random ≈ −30).
- **Custom MuJoCo triple inverted pendulum** environment + a balancing agent.
- **Pluggable components:** swap the encoder / dynamics / actor / critic by name via a
  registry — add a new world-model piece without touching the agent.
- **Honest experiments:** a domain-randomization robustness study and the limits of the
  triple-pendulum agent, written up as findings.

## Results

| Task | Metric | Result | Reference |
|------|--------|--------|-----------|
| CarRacing-v3 (pixels) | mean return, 5 ep | **632** | random ≈ −30 · "solved" ≈ 900 |
| CarRacing-v3, unseen random tracks | mean return, 6 ep | **377 ± 246** | generalizes to new tracks |
| CartPole-v1 | return | solved | — |
| Triple inverted pendulum (custom) | balance time | **~2 s** | random ≈ instant fall |

## Demos

| CarRacing — driving | World-model "dream" | Triple pendulum |
|:---:|:---:|:---:|
| ![drive](assets/carracing_drive.gif) | ![dream](assets/carracing_dream.gif) | ![pendulum](assets/triple_pendulum.gif) |

*Left → right: the trained agent driving · the agent acting inside its imagined world ·
the custom 3-link inverted pendulum balanced on a cart.*

## How it works

```
observation ─► Encoder ─► embed ─┐
                                 ▼
        ┌──────────── RSSM  (GRU deterministic h + categorical stochastic z) ───────────┐
        │      posterior q(z | h, embed)        prior p(z | h)   ← used for imagination   │
        └───────────────────────────────┬───────────────────────────────────────────────┘
                       feat = [h, z] ────┼──► Decoder        (reconstruct observation)
                                         ├──► Reward head    (two-hot symlog)
                                         ├──► Continue head  (discount γ)
                                         ├──► Actor   ┐  trained on imagined rollouts
                                         └──► Critic  ┘  with λ-returns (real env never
                                                         touched during the AC update)
```

The world model compresses observations into a latent state and learns to predict its own
future. The **actor-critic is trained entirely on short trajectories imagined by rolling the
latent prior forward** — the real environment is only used to collect data for the world
model. DreamerV3 ingredients implemented here: symlog + two-hot regression, KL balancing with
free bits, percentile return normalization, straight-through categorical latents, and an EMA
target critic.

## Quickstart

```bash
uv sync                                   # Python 3.11–3.12

# Train (standalone Dreamer entry point; auto-detects image vs vector envs)
PYTHONPATH=src uv run python src/dreamer/train.py --env CartPole-v1 --total-steps 20000
PYTHONPATH=src uv run python src/dreamer/train.py --env CarRacing-v3 --total-steps 300000 \
    --seq-len 50 --batch-size 16 --deter-dim 256 --cnn-depth 32 --action-repeat 2
PYTHONPATH=src uv run python src/dreamer/train.py --env InvertedTriplePendulum-v0 \
    --total-steps 300000 --action-repeat 1 --deter-dim 256 --entropy-scale 2e-3

# Showcase tooling
PYTHONPATH=src uv run python src/dreamer/record_demo.py   --checkpoint <ckpt> --env <env> --out demo/run
PYTHONPATH=src uv run python src/dreamer/dream_rollout.py --checkpoint <ckpt> --env CarRacing-v3 --out demo/dream
PYTHONPATH=src uv run python src/dreamer/eval_tests.py    --checkpoint <ckpt> --env CarRacing-v3 --tests 6 --stochastic
PYTHONPATH=src uv run python src/dreamer/showcase.py      --checkpoint <ckpt> --seeds 16 --out demo/showcase
```

### DreamDrive — play inside the world model

```bash
PYTHONPATH=src uv run python src/dreamer/dreamdrive.py --checkpoint <ckpt> --record demo/session.mp4
```

An interactive window where **you** drive inside the trained world model. After a short
warm-up grounds the latent state on real frames, the environment is discarded — every
frame is the RSSM prior stepped forward with your keyboard action and decoded to pixels
(~3.5 ms/frame on MPS). Arrows drive, `A` hands the wheel to the trained actor, `TAB`
switches to a dream grounded on a different track, and staying in the imagined grass for
3 seconds ends the run: the model genuinely forgets roads it cannot see.

### Vectorized collection

Data collection runs on parallel subprocess environments with one batched policy call
per tick — on CarRacing this triples wall-clock throughput:

```bash
PYTHONPATH=src uv run python src/dreamer/train.py --env CarRacing-v3 --total-steps 800000 \
    --num-envs 6 --train-every 10 --deter-dim 512 --cnn-depth 48
```

Checkpoints (`dreamer_<env>_best.pt`, `_step<N>.pt`, `_final.pt`) are written to `--save-path`.
Resume with `--load-path <ckpt>`.

### Long / overnight runs (macOS)

```bash
# caffeinate stops idle/system sleep (which suspends training); nohup detaches the run.
PYTHONPATH=src nohup uv run python src/dreamer/train.py \
    --env CarRacing-v3 --total-steps 300000 --tensorboard \
    > train.log 2>&1 &
echo $! > train.pid
nohup caffeinate -i -m -s -w "$(cat train.pid)" >/dev/null 2>&1 &
```

> **MacBook note:** `caffeinate` cannot override *clamshell* sleep — keep the lid open (or
> external display + AC). At `deter 256 / cnn 32` throughput is ≈ 8–9 env-steps/s on MPS.

## Pluggable components

Dreamer's world-model pieces are selected from registries, mirroring the project's modular
philosophy. Add a new component and use it with one line — no agent changes:

```python
from dreamer.networks import Dynamics

class MyTransformerDynamics(Dynamics):          # auto-registers as "MyTransformerDynamics"
    @classmethod
    def from_config(cls, cfg, embed_dim): ...
    # implement initial / obs_step / img_step / get_feat / feat_dim

# select it:  DreamerConfig(..., dynamics="MyTransformerDynamics")
```

Registries: `ENCODER_REGISTRY`, `DECODER_REGISTRY`, `DYNAMICS_REGISTRY`, `ACTOR_REGISTRY`,
`CRITIC_REGISTRY`. `DreamerConfig` picks each by name (empty = modality-appropriate default).

## Experiments & honest findings

- **Domain-randomization robustness.** The agent generalizes to unseen track *shapes* but
  collapses under `domain_randomize` (random colors) — and the world-model reconstruction
  shows *why*. The study also surfaced that CarRacing's `domain_randomize` is partly
  **ill-posed**: it can sample road ≈ grass colour → genuinely unwinnable maps. Fine-tuning
  taught the vision to *see* random palettes but didn't yield a clean robustness win on a
  broken benchmark.
- **Triple inverted pendulum.** A from-scratch MuJoCo env. The agent stands all three links
  upright for ~2 s but doesn't fully stabilize; the bottleneck is reaction speed on the
  chaotic dynamics, not (as first hypothesized) cart-rail length.
- **Train vs. eval gap.** The training rolling-mean reward repeatedly disagreed with held-out
  greedy evaluation — a reminder to score on a fixed protocol, not the training metric.

## Repo layout

```
src/dreamer/            DreamerV3-lite — networks, models (agent), replay, train, env,
                        triple_pendulum (custom MuJoCo env), record_demo, dream_rollout,
                        eval_tests, showcase, dreamdrive (playable dream)
src/{vision,memory,controller}/   original modular PPO world-model framework (registry-based)
tests/test_dreamer.py   unit tests for the Dreamer agent (math, world model, actor-critic)
```

## Limitations & next steps

- CarRacing 632 is strong but not "solved" (~900) — a larger-scale vectorized run is in
  progress to close the gap.
- No automated eval harness yet (evaluation runs through the demo/eval scripts).
- Next: an action-conditioned world model trained on real dashcam video — same RSSM,
  real roads.

## References

- DreamerV3 — *Mastering Diverse Domains through World Models* — https://arxiv.org/abs/2301.04104
- World Models (Ha & Schmidhuber, 2018) — https://worldmodels.github.io
