"""DreamerV3-lite: a compact, MPS-friendly implementation of latent-imagination RL.

This package implements a self-contained DreamerV3-style agent (Hafner et al., 2023,
https://arxiv.org/pdf/2301.04104v1):

    - A recurrent state-space world model (RSSM) with categorical latents that learns
      to reconstruct observations, predict rewards and episode continuation, and
      imagine future latent trajectories.
    - An actor-critic trained purely on imagined rollouts inside the world model.

It is wired into the project CLI via ``python src/main.py --algo dreamer ...`` and is
deliberately kept independent from the PPO ``WorldModel`` plumbing so the latent
imagination loop stays clean and faithful to the paper.
"""

# Importing this module registers the custom ``InvertedTriplePendulum-v0`` env with
# Gymnasium so ``gym.make`` can find it from any entry point.
from dreamer import triple_pendulum as triple_pendulum  # noqa: E402,F401
