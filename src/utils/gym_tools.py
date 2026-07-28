import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import registry
from gymnasium.spaces import Space


def get_all_gym_envs() -> list:
    """Returns a list of all registered gym environments."""
    all_envs = list(registry.keys())
    return [env for env in all_envs if not env.startswith("_")]


def get_env_info(env_name: str) -> tuple[Space, Space, bool, bool]:
    """Returns information about the environment."""
    env = gym.make(env_name)
    observation_space = env.observation_space
    action_space = env.action_space
    assert observation_space.shape is not None
    is_image_based = len(observation_space.shape) == 3
    is_discrete = isinstance(action_space, gym.spaces.Discrete)
    env.close()
    return observation_space, action_space, is_image_based, is_discrete


def auto_batch_size(env_name: str) -> int:
    """
    Heuristically pick the number of parallel environments when --batch-size auto is used.

    Strategy:
      - Vector-based environments (e.g. CartPole) are cheap: scale up to the CPU count.
      - Image-based environments (e.g. CarRacing) are heavier: use at most half the CPUs
        so the data-loading / GPU-transfer pipeline doesn't become the bottleneck.

    The result is clamped to [2, 16] to stay safe on small machines and avoid
    diminishing returns on very large ones.

    Args:
        env_name: Gymnasium environment ID.

    Returns:
        Recommended number of parallel environments.
    """
    cpu_count = os.cpu_count() or 4
    _, _, is_image_based, _ = get_env_info(env_name)

    if is_image_based:
        # Heavier per-step cost; fewer envs keep the pipeline balanced
        n = max(2, min(cpu_count // 2, 8))
    else:
        # Lightweight vector envs scale well with core count
        n = max(2, min(cpu_count, 16))

    return n


def state_transform(state: np.ndarray, is_image_based: bool, device: torch.device) -> torch.Tensor:
    """
    Convert a numpy observation from a Gymnasium environment into a normalized
    PyTorch tensor ready for the world model.

    Handles both single observations and batches from vectorized environments.

    Args:
        state:          Numpy array with shape:
                        - Images: (H, W, C) or (B, H, W, C)
                        - Vectors: (D,) or (B, D)
        is_image_based: Whether the observation is image-based.
        device:         Target torch device.

    Returns:
        Tensor of shape (B, C, H, W) normalized to [0, 1] for images,
        or (B, D) for vectors.
    """
    if is_image_based:
        if state.ndim == 3:  # single (H, W, C) → add batch dim
            state = state[None]
        # (B, H, W, C) → (B, C, H, W), then normalize to [0, 1]
        return torch.from_numpy(np.transpose(state, (0, 3, 1, 2))).float().to(device) / 255.0
    else:
        if state.ndim == 1:  # single (D,) → add batch dim
            state = state[None]
        return torch.from_numpy(state).float().to(device)
