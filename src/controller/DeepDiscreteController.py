from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from controller import ControllerModel
from utils.tags import ControllerTag


def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> None:
    """Apply orthogonal initialization (common in RL for stability)."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class DeepDiscreteController(ControllerModel):
    """
    A deeper discrete controller with:
    - Multi-layer feature extraction
    - Layer normalization for training stability
    - Separate actor/critic heads
    - Orthogonal initialization
    """

    tags = frozenset({ControllerTag.DISCRETE, ControllerTag.STOCHASTIC})

    def __init__(
        self, z_dim: int, h_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2, **_kwargs: Any
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = z_dim + h_dim

        # Shared feature extraction layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        self.shared_features = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

        # Critic head (value function)
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        # Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.shared_features.modules():
            orthogonal_init(module, gain=1.0)

        for module in self.actor.modules():
            orthogonal_init(module, gain=0.01)  # Small gain for policy head

        for module in self.critic.modules():
            orthogonal_init(module, gain=1.0)

    def forward(
        self, z_t: torch.Tensor, h_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Concatenate latent state and hidden state
        x = torch.cat([z_t, h_t], dim=-1)  # (B, z_dim + h_dim)

        # Shared feature extraction
        features = self.shared_features(x)  # (B, hidden_dim)

        # Actor: compute action logits and sample
        logits = self.actor(features)  # (B, action_dim)
        dist = Categorical(logits=logits)
        action = dist.sample()  # (B,)
        log_prob = dist.log_prob(action).unsqueeze(-1)  # (B, 1)
        entropy = dist.entropy().unsqueeze(-1)  # (B, 1)

        # Critic: compute value estimate
        value = self.critic(features)  # (B, 1)

        return action, log_prob, value, entropy

    def evaluate_actions(self, z_t: torch.Tensor, h_t: torch.Tensor, actions: torch.Tensor) -> tuple[Any, Any, Any]:
        """Evaluate log probability and entropy for given actions."""
        x = torch.cat([z_t, h_t], dim=-1)
        features = self.shared_features(x)

        logits = self.actor(features)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(actions.long())
        value = self.critic(features).squeeze(-1)
        entropy = dist.entropy()

        return log_prob, value, entropy

    def export_hyperparams(self) -> dict[str, Any]:
        return {
            "z_dim": self.z_dim,
            "h_dim": self.h_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }

    def save_state(self) -> Any:
        return self.state_dict()

    def load(self, state_dict: dict[str, Any]) -> None:
        self.load_state_dict(state_dict)
