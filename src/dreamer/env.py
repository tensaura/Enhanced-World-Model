"""Environment wrapper for DreamerV3-lite.

A thin single-environment adapter that:

    - resizes image observations to ``image_size`` and keeps them as ``uint8`` HWC
      (cheap to store in replay; converted to float on the GPU at train time);
    - applies action-repeat (frame-skip), summing reward over the repeated steps;
    - exposes a uniform *policy action space*: a one-hot vector for discrete envs and
      a ``[-1, 1]`` vector for continuous envs, and maps it to the native env action.

DreamerV3 needs no frame-stacking — the RSSM's recurrent state already captures
motion — so a single (resized) frame per step is all we store.
"""

from __future__ import annotations

from typing import Any

import cv2
import gymnasium as gym
import numpy as np


class DreamerEnv:
    """Single-environment wrapper producing Dreamer-ready observations and actions."""

    def __init__(
        self,
        env_name: str,
        action_repeat: int = 1,
        image_size: int = 64,
        seed: int = 0,
        render_mode: str | None = None,
        env_kwargs: dict | None = None,
    ) -> None:
        self.env_name = env_name
        self.action_repeat = action_repeat
        self.image_size = image_size
        self._seed = seed
        # Extra gym.make kwargs (e.g. {"domain_randomize": True} for CarRacing-v3).
        self.env = gym.make(env_name, render_mode=render_mode, **(env_kwargs or {}))

        obs_space = self.env.observation_space
        assert obs_space.shape is not None
        self.is_image = len(obs_space.shape) == 3

        if self.is_image:
            # Stored/processed observations are (image_size, image_size, C) uint8.
            channels = obs_space.shape[2]
            self.obs_shape: tuple[int, ...] = (image_size, image_size, channels)
        else:
            self.obs_shape = obs_space.shape

        act_space = self.env.action_space
        self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
        if self.is_discrete:
            assert isinstance(act_space, gym.spaces.Discrete)
            self.action_dim = int(act_space.n)
            self.act_low = None
            self.act_high = None
        else:
            assert isinstance(act_space, gym.spaces.Box)
            self.action_dim = int(act_space.shape[0])
            self.act_low = act_space.low.astype(np.float32)
            self.act_high = act_space.high.astype(np.float32)

    # ------------------------------------------------------------------ helpers

    def _process(self, obs: np.ndarray) -> np.ndarray:
        if self.is_image:
            if obs.shape[0] != self.image_size or obs.shape[1] != self.image_size:
                obs = cv2.resize(
                    obs, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
                )
            return obs.astype(np.uint8)
        return obs.astype(np.float32)

    def to_env_action(self, policy_action: np.ndarray) -> Any:
        """Map a policy-space action to the native environment action.

        Discrete: ``policy_action`` is a one-hot vector → integer index.
        Continuous: ``policy_action`` is in ``[-1, 1]`` → affine map to ``[low, high]``.
        """
        if self.is_discrete:
            return int(np.argmax(policy_action))
        assert self.act_low is not None and self.act_high is not None
        scaled = 0.5 * (policy_action + 1.0) * (self.act_high - self.act_low) + self.act_low
        return scaled.astype(np.float32)

    # --------------------------------------------------------------------- api

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset(seed=self._seed)
        self._seed += 1  # vary seed across episodes for diversity
        return self._process(obs)

    def step(self, policy_action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        env_action = self.to_env_action(policy_action)
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        obs = None
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        assert obs is not None
        return self._process(obs), total_reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
