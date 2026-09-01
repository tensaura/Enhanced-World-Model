from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from controller import ControllerModel
from utils.tags import ControllerTag


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization (common in RL for stability)."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DeepContinuousController(ControllerModel):
    """
    A deeper continuous controller with:
    - Multi-layer feature extraction
    - Layer normalization for training stability
    - Separate actor/critic heads
    - Orthogonal initialization
    - Gaussian policy for continuous actions
    """

    tags = frozenset({ControllerTag.CONTINUOUS})

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        log_std_min: float = -5.0,
        log_std_max: float = 0.5,
        **_kwargs: Any,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        input_dim = z_dim + h_dim

        # Shared feature extraction layers
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        self.shared_features = nn.Sequential(*layers)

        # Actor head (policy) - outputs mean and log_std
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bound mean to [-1, 1]
        )

        self.actor_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        # Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.shared_features.modules():
            orthogonal_init(module, gain=1.0)

        for module in self.actor_mean.modules():
            orthogonal_init(module, gain=0.01)  # Small gain for policy head

        for module in self.actor_log_std.modules():
            orthogonal_init(module, gain=0.01)

        for module in self.critic.modules():
            orthogonal_init(module, gain=1.0)

    def forward(
        self, z_t: torch.Tensor, h_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Concatenate latent state and hidden state
        x = torch.cat([z_t, h_t], dim=-1)  # (B, z_dim + h_dim)

        # Shared feature extraction
        features = self.shared_features(x)  # (B, hidden_dim)

        # Actor: compute mean and std for Gaussian policy
        mu = self.actor_mean(features)  # (B, action_dim), already in [-1, 1]
        log_std = self.actor_log_std(features)  # (B, action_dim)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # Sample action from Gaussian
        dist = Normal(mu, std)
        raw_action = dist.rsample()  # Reparameterized sample for backprop
        action = torch.tanh(raw_action)  # Squash to [-1, 1]

        # Log probability with tanh correction
        log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)
        # Correction for tanh squashing
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        # Entropy
        entropy = dist.entropy().sum(-1, keepdim=True)

        # Critic: compute value estimate
        value = self.critic(features)  # (B, 1)

        return action, log_prob, value, entropy

    def evaluate_actions(
        self, z_t: torch.Tensor, h_t: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given actions."""
        x = torch.cat([z_t, h_t], dim=-1)
        features = self.shared_features(x)

        mu = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mu, std)

        # Inverse tanh to get raw actions (for log_prob computation)
        # Clamp to avoid numerical issues at boundaries
        actions_clamped = torch.clamp(actions, -0.999, 0.999)
        raw_actions = torch.atanh(actions_clamped)

        log_prob = dist.log_prob(raw_actions).sum(-1)
        # Correction for tanh squashing
        log_prob = log_prob - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)

        value = self.critic(features).squeeze(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, value, entropy

    def export_hyperparams(self) -> dict[str, Any]:
        return {
            "z_dim": self.z_dim,
            "h_dim": self.h_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
        }

    def save_state(self) -> Any:
        return self.state_dict()

    def load(self, state_dict: dict[str, Any]) -> None:
        self.load_state_dict(state_dict)
