# TODO

## Done — DreamerV3-lite (latent imagination)

A self-contained DreamerV3 agent now lives in `src/dreamer/` and is the recommended
path for image environments like CarRacing-v3 (run with `--algo dreamer`). It learns the
policy by imagination inside a categorical RSSM world model and fixes the failure modes
of the PPO path (spatial info loss from mean-pooling, drifting representation). See the
README "DreamerV3-lite" section. Validated: world model reconstructs CarRacing frames;
imagination actor-critic improves reward on CartPole. Next: the long CarRacing run
(`--total-steps 500000 --deter-dim 512 --cnn-depth 48`) and record the demo.

## Original modular-world-model goals

The first goal is to have a working full model:
- VQ-VAE as the vision model (https://arxiv.org/pdf/1711.00937)
- Temporal transformer + Contrastive Predictive Coding (CPC) for the memory model (see TWISTER, https://arxiv.org/abs/2503.04416)
- The same controller model as the original article (https://worldmodels.github.io)


Then, we will test by swapping out different parts of the full model to test stability, performance, and training requirements.

Also see [Vision, Memory, Controller]:
- PlaNet (https://arxiv.org/pdf/1811.04551) [CNN + Variational Autoencoder (VAE), Recurrent State-Space Model, Model Predictive Control]
- STORM (https://arxiv.org/pdf/2310.09615) [VAE or CNN, Transformer variationnel, RL on latent]
- TWISTER, close from our first attempt (https://arxiv.org/abs/2503.04416) [VAE or CNN, Causal transformer, Model Predictive Control]
- DreamerV3, totally different in architecture (https://arxiv.org/pdf/2301.04104v1)
