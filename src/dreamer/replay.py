"""Sequence replay buffer for the Dreamer world model.

Stores a flat ring of transitions and samples fixed-length ``(B, L)`` windows. Each
transition records ``obs`` (the observation at this step), ``action`` (the action that
*led into* this observation), ``reward`` (reward for reaching it), ``is_first`` (episode
start) and ``is_terminal`` (true episode end, not a time-limit truncation).

Windows may span an episode boundary — the ``is_first`` flag resets the RSSM mid-window —
but never straddle the ring's write pointer (those windows are rejected when sampling).
"""

from __future__ import annotations

import numpy as np
import torch


class SequenceReplay:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int, is_image: bool) -> None:
        self.capacity = capacity
        self.action_dim = action_dim
        obs_dtype = np.uint8 if is_image else np.float32
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.is_first = np.zeros(capacity, dtype=bool)
        self.is_terminal = np.zeros(capacity, dtype=bool)
        self.ptr = 0
        self.size = 0
        self.wrapped = False

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, is_first: bool, is_terminal: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.is_first[i] = is_first
        self.is_terminal[i] = is_terminal
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.ptr == 0:
            self.wrapped = True

    def can_sample(self, seq_len: int) -> bool:
        return self.size > seq_len + 1

    def sample(self, batch_size: int, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
        high = self.size - seq_len
        starts = np.empty(batch_size, dtype=np.int64)
        count = 0
        while count < batch_size:
            s = np.random.randint(0, high + 1)
            # Reject windows that straddle the ring's old/new boundary.
            if self.wrapped and s <= self.ptr < s + seq_len:
                continue
            starts[count] = s
            count += 1

        idx = starts[:, None] + np.arange(seq_len)[None, :]  # (B, L)
        batch = {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "action": torch.as_tensor(self.action[idx], device=device),
            "reward": torch.as_tensor(self.reward[idx], device=device),
            "is_first": torch.as_tensor(self.is_first[idx], device=device).float(),
            "is_terminal": torch.as_tensor(self.is_terminal[idx], device=device).float(),
        }
        return batch


class VecSequenceReplay:
    """Sequence replay for N lockstep parallel environments.

    All envs advance together, so storage is a 2-D ring: row ``r`` holds the
    r-th transition of every env, and a single write pointer serves all
    columns. Sampling picks an (env column, row window) pair; windows never
    straddle the write pointer. Flag semantics match :class:`SequenceReplay`.
    """

    def __init__(
        self, capacity_per_env: int, num_envs: int, obs_shape: tuple[int, ...], action_dim: int, is_image: bool
    ) -> None:
        self.capacity = capacity_per_env
        self.num_envs = num_envs
        self.action_dim = action_dim
        obs_dtype = np.uint8 if is_image else np.float32
        self.obs = np.zeros((capacity_per_env, num_envs, *obs_shape), dtype=obs_dtype)
        self.action = np.zeros((capacity_per_env, num_envs, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity_per_env, num_envs), dtype=np.float32)
        self.is_first = np.zeros((capacity_per_env, num_envs), dtype=bool)
        self.is_terminal = np.zeros((capacity_per_env, num_envs), dtype=bool)
        self.ptr = 0
        self.size = 0
        self.wrapped = False

    def add_batch(
        self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, is_first: np.ndarray, is_terminal: np.ndarray
    ) -> None:
        """Write one row: the current transition of every env, shapes (N, ...)."""
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.is_first[i] = is_first
        self.is_terminal[i] = is_terminal
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.ptr == 0:
            self.wrapped = True

    def can_sample(self, seq_len: int) -> bool:
        return self.size > seq_len + 1

    def sample(self, batch_size: int, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
        high = self.size - seq_len
        starts = np.empty(batch_size, dtype=np.int64)
        count = 0
        while count < batch_size:
            s = np.random.randint(0, high + 1)
            if self.wrapped and s <= self.ptr < s + seq_len:
                continue
            starts[count] = s
            count += 1
        cols = np.random.randint(0, self.num_envs, size=batch_size)

        idx = starts[:, None] + np.arange(seq_len)[None, :]  # (B, L)
        cols_b = cols[:, None]  # (B, 1) broadcast against (B, L)
        batch = {
            "obs": torch.as_tensor(self.obs[idx, cols_b], device=device),
            "action": torch.as_tensor(self.action[idx, cols_b], device=device),
            "reward": torch.as_tensor(self.reward[idx, cols_b], device=device),
            "is_first": torch.as_tensor(self.is_first[idx, cols_b], device=device).float(),
            "is_terminal": torch.as_tensor(self.is_terminal[idx, cols_b], device=device).float(),
        }
        return batch
