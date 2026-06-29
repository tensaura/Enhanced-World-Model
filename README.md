# Enhanced-World-Model
An extensible and modular reimplementation of Ha & Schmidhuber’s
*Recurrent World Models Facilitate Policy Evolution (2018)*.

This project provides a flexible framework for experimenting with
modern world-model architectures in reinforcement learning. It enables
researchers to easily swap vision, memory, and controller components
while preserving a unified training pipeline.

## Architecture Overview

The enhanced world model is composed of three interchangeable modules:

1. **Vision Model** – Encodes observations into latent representations
2. **Memory Model** – Models temporal dynamics in latent space
3. **Controller** – Produces actions based on latent states

Each component can be independently selected and trained, enabling
rapid experimentation with different architectures.

## Features

- **Modular design** allowing independent vision, memory and controller components
- **Flexible training pipeline** supporting multiple environments and
  model architectures
- **Reproducible experiments** through explicit configuration and
  seeding

This modular design enables rapid experimentation with different world-model configurations.

## Environments

The framework targets compatibility with environments provided by
**Gymnasium**.
https://gymnasium.farama.org

Currently tested environments:
- CartPole-v1
- CarRacing-v3

## Requirements

- Python **3.11 – 3.12.11**
- NVIDIA GPU recommended for vision-based environments

## Installation
This project uses **uv** for dependency management.

- Clone the repository
  ```$ git clone https://github.com/Larwive/Enhanced-World-Model/```
- Install dependencies
  ```$ uv sync```

## Training

To train or pretrain models, run `src/main.py`.

The accepted arguments are:
### Interface
- `--ui` to use the Gradio web interface. Not compatible with `--cli`.
- `--cli` to use the CLI. Not compatible with `--ui`.

### Environment & Training
- `--env` to set the environment to use.
- `--vision` to set the vision model to use. Loading an existing model will overwrite this argument.
- `--memory` to set the memory model to use. Loading an existing model will overwrite this argument.
- `--controller` to set the controller model to use. Loading an existing model will overwrite this argument.

- `--epochs` to set the number of epochs to run.
- `--patience` to set the number of iterations without noticeable improvement before early stopping.
- `--batch-size` to set the number of environments to run in parallel. Can be set automatically with `auto`.
- `--lr` the learning rate for the model, except the controller model.
- `--dropout` to control the dropout rate.
- `--render-mode` to set render mode between `human` and `rgb_array` (no render).

- `--seed` to set the seed.
- `--save-path` for the path to save the model.
- `--load-path` to load an existing model.
- `--patch-load-path` to modularly load a model on top of the existing model.
- `--patch` to specify which parts (vision, memory, controller) to load with `--patch-load-path`.
- `--save-freq` the number of epochs between each save.
- `--log-freq` the number of epochs between each log.
- `--tensorboard` whether to log gradients and losses into TensorBoard.

## Pretraining

Pretraining allows the vision and memory models to be trained
independently before full reinforcement learning.

During pretraining:
- The controller is bypassed
- Actions are either random or manually provided
- The resulting model can later be loaded for full training

Pretraining-specific arguments:
- `--pretrain-vision`
- `--pretrain-memory`
- `--pretrain-mode` (`random` or `manual`)
- `--manual-mode-delay` to set the delay between each step during manual pretraining.

## PPO Configuration

The controller is trained using Proximal Policy Optimization (PPO).
The following arguments control the PPO training process:
- `--rollout-steps` the number of rollout steps.
- `--ppo-epochs` the number of PPO epochs.
- `--ppo-lr` for the PPO learning rate.
- `--ppo-batch-size` the batch size for PPO updates by batch.
- `--ppo-clip-range` the interval for PPO gradient clipping.
- `--ppo-range-vf` the value function for PPO gradient clipping.
- `--gamma` the discount factor.
- `--gae-lambda` the GAE lambda parameter.
- `--value-coef` the coefficient for the value loss.
- `--entropy-coef` the coefficient for the entropy loss.
- `--max-grad-norm` the value to clip gradients at.
- `--train-world-model` whether to train vision and memory too.
- `--world-model-epochs` the number of epochs per rollout to train vision and memory.

For example, if you want to train a model on CartPole-v3, run:
```$ python src/main.py --env CartPole-v3```

To use specific models, run (as an example):
```$ python src/main.py --env CartPole-v3 --vision VQ_VAE --memory LSTMMemory --controller DeepContinuousController```

## Inference

To infer on pretrained models, run `src/inference.py`.

The accepted arguments are:
- `--load-path` the path to the pretrained model.
- `--episodes` the number of episodes to infer on.
- `--render-mode` to set render mode between `human` and `rgb_array` (no render).

## What you can do
You can train your own models, using the implemented vision, memory and controller sub models.

Alternatively, you can use the pretrained models provided and run them
on their respective environments.

## Quick Start

```bash
uv sync
python src/main.py --env CartPole-v1 --render-mode human
```

## DreamerV3-lite (latent imagination)

The strongest agent in this repo is a compact, self-contained
[DreamerV3](https://arxiv.org/pdf/2301.04104v1) implementation that trains the policy
**inside an imagined latent world model** rather than with on-policy PPO. It lives in
`src/dreamer/` and is selected with `--algo dreamer`.

Why it succeeds on image environments like **CarRacing-v3** where the PPO path struggled:

- **No spatial information is thrown away.** The PPO path mean-pooled the conv feature
  map to a single vector; Dreamer keeps a full recurrent latent state.
- **The recurrent state captures motion** (velocity, heading), so no frame-stacking
  hacks are needed — a single resized 64×64 frame per step is enough.
- **The policy learns by imagination.** The actor-critic is trained on thousands of
  latent rollouts *dreamed* by the world model, decoupling policy learning from the
  (initially noisy) representation.

Architecture: a CNN/MLP **encoder**, a categorical **RSSM** (deterministic GRU state +
discrete stochastic latents), a **decoder** + **reward** + **continue** heads, and an
**actor-critic** trained on λ-returns over imagined trajectories. Robustness tricks from
the paper are included: symlog two-hot reward/value heads, KL balancing with free bits,
percentile return normalisation, an EMA target critic, and unimix categoricals.

### Train

Quick proof that the pipeline learns (vector env, a few minutes on CPU):

```bash
python src/main.py --algo dreamer --env CartPole-v1 \
    --total-steps 20000 --action-repeat 1 --deter-dim 128 \
    --seq-len 32 --dreamer-batch 16 --entropy-scale 1e-2
```

CarRacing proof run (~20 min on Apple-Silicon/MPS — world-model reconstructions become
recognisable and reward starts trending up):

```bash
python src/main.py --algo dreamer --env CarRacing-v3 \
    --total-steps 25000 --seq-len 32 --dreamer-batch 8 \
    --deter-dim 256 --cnn-depth 32 --action-repeat 2
```

Full CarRacing run for a strong agent (~10 h overnight on Apple-Silicon/MPS). Measured
throughput is ≈8.6 env-steps/s at `deter 256 / cnn 32 / batch 16 / seq 50`, so 300k steps
fits a night; the `512`/`48` model scores higher but needs a multi-day budget on a Mac.

```bash
# `caffeinate -i -m -s` keeps macOS from idle/system-sleeping mid-run (which suspends
# training); `-w <pid>` releases automatically when training exits. nohup detaches it so
# it survives closing the terminal/app. Run via the standalone entry for the cleanest path.
PYTHONPATH=src nohup uv run python src/dreamer/train.py \
    --env CarRacing-v3 --total-steps 300000 --prefill 5000 \
    --seq-len 50 --batch-size 16 --deter-dim 256 --cnn-depth 32 \
    --train-every 5 --action-repeat 2 --entropy-scale 1.5e-3 --tensorboard \
    > overnight_carracing.log 2>&1 &
echo $! > overnight_carracing.pid
nohup caffeinate -i -m -s -w "$(cat overnight_carracing.pid)" >/dev/null 2>&1 &
```

> **MacBook note:** `caffeinate` cannot override *clamshell* sleep — **keep the lid open**
> (an external display + AC also works). For closed-lid training, run
> `sudo pmset -c disablesleep 1` first (AC only) and `sudo pmset -c disablesleep 0` after.
> Monitor with `tail -f overnight_carracing.log | grep "step "`; stop with
> `kill $(cat overnight_carracing.pid)`.

Resume from a checkpoint with `--load-path saved_models/dreamer_CarRacing-v3_step50000.pt`.
Checkpoints (`dreamer_<env>_best.pt`, `_step<N>.pt`, `_final.pt`) are written to
`--save-path` (default `./saved_models/`).

### Hyperparameter guidance

- `--entropy-scale`: continuous envs (CarRacing) want a small value (`1e-3`–`3e-4`);
  discrete envs (CartPole) want more exploration (`1e-2`–`3e-2`).
- `--deter-dim` / `--cnn-depth`: the main capacity knobs. `256`/`32` trains fast on a
  Mac; `512`/`48` is closer to the paper's "small" model and scores higher given time.
- `--seq-len` / `--dreamer-batch`: bigger is better for the world model but costs memory
  and step time (≈0.2 s/step at `8`×`32`, ≈0.55 s/step at `16`×`50` on MPS).
- `--train-every`: lower = higher replay ratio = more sample-efficient but slower wall-clock.

### Record a demo video

```bash
PYTHONPATH=src uv run python src/dreamer/record_demo.py \
    --checkpoint saved_models/dreamer_CarRacing-v3_final.pt \
    --env CarRacing-v3 --episodes 3 --out demo/carracing
```

This writes `demo/carracing.mp4` (+ `.gif`) showing the agent driving next to the world
model's decoded "dream" of each frame, with a live reward sparkline.
