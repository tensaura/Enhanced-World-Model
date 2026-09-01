"""Rollout buffer for PPO training."""

from collections.abc import Generator
import torch
import numpy as np


class RolloutBuffer:
    """
    Buffer for storing rollout experiences and computing returns/advantages.

    Stores:
        - observations (states)
        - actions
        - rewards
        - values (from critic)
        - log_probs (from policy)
        - dones (episode terminations)

    Computes:
        - GAE (Generalized Advantage Estimation)
        - Returns for value function training
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        is_discrete: bool = True,
    ) -> None:
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Number of steps to store per environment
            num_envs: Number of parallel environments
            obs_shape: Shape of observations
            action_dim: Dimension of actions
            device: Torch device
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            is_discrete: Whether action space is discrete
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.is_discrete = is_discrete

        # Pre-allocate tensors
        self.observations = torch.zeros((buffer_size, num_envs, *obs_shape), dtype=torch.float32, device=device)
        # Actions: discrete -> (buffer, envs), continuous -> (buffer, envs, action_dim)
        if is_discrete:
            self.actions = torch.zeros((buffer_size, num_envs), dtype=torch.long, device=device)
        else:
            self.actions = torch.zeros((buffer_size, num_envs, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=device)

        # Computed after rollout
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)

        # For memory model - store hidden states
        self.hidden_states: torch.Tensor | None = None  # Will be set based on memory model
        self.latent_states: torch.Tensor | None = None  # z_t from vision encoder
        self.next_latent_states: torch.Tensor | None = None  # z_{t+1} for memory training

        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        """Reset the buffer for a new rollout."""
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        z_t: torch.Tensor | None = None,
        h_t: torch.Tensor | None = None,
        z_next: torch.Tensor | None = None,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: Observations (num_envs, *obs_shape)
            action: Actions (num_envs,)
            reward: Rewards (num_envs,)
            value: Value estimates (num_envs,)
            log_prob: Log probabilities (num_envs,)
            done: Done flags (num_envs,)
            z_t: Latent states from vision (num_envs, latent_dim)
            h_t: Hidden states from memory (num_envs, h_dim)
            z_next: Next latent states for memory training (num_envs, latent_dim)
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.squeeze(-1) if value.dim() > 1 else value
        self.log_probs[self.ptr] = log_prob.squeeze(-1) if log_prob.dim() > 1 else log_prob
        self.dones[self.ptr] = done

        # Store latent/hidden states if provided
        if z_t is not None:
            if self.latent_states is None:
                self.latent_states = torch.zeros(
                    (self.buffer_size, self.num_envs, z_t.shape[-1]), dtype=torch.float32, device=self.device
                )
            assert self.latent_states is not None  # for type checker
            self.latent_states[self.ptr] = z_t

        if h_t is not None:
            if self.hidden_states is None:
                self.hidden_states = torch.zeros(
                    (self.buffer_size, self.num_envs, h_t.shape[-1]), dtype=torch.float32, device=self.device
                )
            assert self.hidden_states is not None  # for type checker
            self.hidden_states[self.ptr] = h_t

        if z_next is not None:
            if self.next_latent_states is None:
                self.next_latent_states = torch.zeros(
                    (self.buffer_size, self.num_envs, z_next.shape[-1]), dtype=torch.float32, device=self.device
                )
            assert self.next_latent_states is not None  # for type checker
            self.next_latent_states[self.ptr] = z_next

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor) -> None:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE(γ,λ) = Σ (γλ)^t * δ_t
        where δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)

        Args:
            last_value: Value estimate for the state after the last step
            last_done: Done flag for the last step
        """
        last_value = last_value.squeeze(-1) if last_value.dim() > 1 else last_value

        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = ~last_done
                next_value = last_value
            else:
                next_non_terminal = ~self.dones[step + 1]
                next_value = self.values[step + 1]

            # TD error: δ = r + γ * V(s') * (1-done) - V(s)
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal.float() - self.values[step]

            # GAE: A = δ + γλ * (1-done) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal.float() * gae
            self.advantages[step] = gae

        # Returns = Advantages + Values
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int, shuffle: bool = True) -> Generator[dict[str, torch.Tensor], None, None]:
        """
        Generate mini-batches for PPO update.

        Args:
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle indices

        Yields:
            Dictionary containing batch data
        """
        total_size = self.buffer_size * self.num_envs
        indices = np.arange(total_size)

        if shuffle:
            np.random.shuffle(indices)

        # Flatten all tensors: (buffer_size, num_envs, ...) -> (total_size, ...)
        flat_obs = self.observations.view(total_size, *self.obs_shape)
        # Actions: discrete -> (total_size,), continuous -> (total_size, action_dim)
        if self.is_discrete:
            flat_actions = self.actions.view(total_size)
        else:
            flat_actions = self.actions.view(total_size, self.action_dim)
        flat_values = self.values.view(total_size)
        flat_log_probs = self.log_probs.view(total_size)
        flat_advantages = self.advantages.view(total_size)
        flat_returns = self.returns.view(total_size)

        flat_z = None
        flat_h = None
        flat_z_next = None
        if self.latent_states is not None:
            flat_z = self.latent_states.view(total_size, -1)
        if self.hidden_states is not None:
            flat_h = self.hidden_states.view(total_size, -1)
        if self.next_latent_states is not None:
            flat_z_next = self.next_latent_states.view(total_size, -1)

        # Generate batches
        start_idx = 0
        while start_idx < total_size:
            batch_indices = indices[start_idx : start_idx + batch_size]

            batch = {
                "observations": flat_obs[batch_indices],
                "actions": flat_actions[batch_indices],
                "old_values": flat_values[batch_indices],
                "old_log_probs": flat_log_probs[batch_indices],
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
            }

            if flat_z is not None:
                batch["latent_states"] = flat_z[batch_indices]
            if flat_h is not None:
                batch["hidden_states"] = flat_h[batch_indices]
            if flat_z_next is not None:
                batch["next_latent_states"] = flat_z_next[batch_indices]

            yield batch
            start_idx += batch_size
