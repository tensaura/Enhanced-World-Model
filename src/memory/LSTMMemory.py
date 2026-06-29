from typing import Any

import torch
import torch.nn as nn

from memory import MemoryModel


class LSTMMemory(MemoryModel):
    """
    LSTM-based memory model following the original World Model architecture.

    This is more suitable than Transformers for incremental state updates
    because:
    - Gradients flow naturally through the recurrent connection
    - No sparse attention issues
    - Designed for sequential processing
    """

    tags: frozenset = frozenset()

    def __init__(
        self,
        latent_dim: int = 4,
        action_dim: int = 2,
        d_model: int = 128,
        num_layers: int = 1,
        **_kwargs: Any,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output projection for next state prediction
        self.output_proj = nn.Sequential(
            nn.Linear(d_model + self.input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, latent_dim),
        )

        # Hidden state storage
        self.h_state = None  # (num_layers, B, d_model)
        self.c_state = None  # (num_layers, B, d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stability."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in [self.input_proj, *self.output_proj.modules()]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Initialize hidden states."""
        self.h_state = torch.zeros(self.num_layers, batch_size, self.d_model, device=device)
        self.c_state = torch.zeros(self.num_layers, batch_size, self.d_model, device=device)

    def update_memory(self, z_t: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """
        Update memory with new observation and previous action.

        Args:
            z_t: Current latent state (B, latent_dim)
            a_prev: Previous action (B, action_dim)

        Returns:
            h_t: Updated hidden state (B, d_model)
        """
        B = z_t.size(0)
        device = z_t.device

        # Initialize hidden state if needed
        if self.h_state is None or self.h_state.size(1) != B:
            self._init_hidden(B, device)

        # Project input
        x = torch.cat([z_t, a_prev], dim=-1)  # (B, input_dim)
        x = self.input_proj(x)  # (B, d_model)
        x = self.input_norm(x)
        x = x.unsqueeze(1)  # (B, 1, d_model)

        # LSTM forward
        output, (h_n, c_n) = self.lstm(x, (self.h_state, self.c_state))

        # Store new hidden state (detached to prevent BPTT through entire history)
        self.h_state = h_n.detach()
        self.c_state = c_n.detach()

        # Return the output hidden state
        h_t = output.squeeze(1)  # (B, d_model)
        return h_t

    def predict_next(self, z_t: torch.Tensor, a_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent state.

        Args:
            z_t: Current latent state (B, latent_dim)
            a_t: Current action (B, action_dim)
            h_t: Current hidden state (B, d_model)

        Returns:
            z_next: Predicted next latent state (B, latent_dim)
        """
        x = torch.cat([z_t, a_t], dim=-1)  # (B, input_dim)
        combined = torch.cat([h_t, x], dim=-1)  # (B, d_model + input_dim)
        z_next = self.output_proj(combined)  # (B, latent_dim)
        return z_next

    def forward(
        self, z_t: torch.Tensor, a_prev: torch.Tensor, a_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: update memory and predict next state.

        Args:
            z_t: Current latent state (B, latent_dim)
            a_prev: Previous action (B, action_dim)
            a_t: Current action (B, action_dim)

        Returns:
            z_next: Predicted next latent state (B, latent_dim)
            h_t: Hidden state (B, d_model)
        """
        h_t = self.update_memory(z_t, a_prev)
        z_next = self.predict_next(z_t, a_t, h_t)
        return z_next, h_t

    def reset_env_memory(self, env_idx: int) -> None:
        """Reset memory for a specific environment."""
        env_idx = int(env_idx)
        if self.h_state is not None and self.c_state is not None:
            self.h_state[:, env_idx] = 0
            self.c_state[:, env_idx] = 0

    def export_hyperparams(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }

    def save_state(self) -> Any:
        return self.state_dict()

    def load(self, state_dict: dict[str, Any]) -> None:
        self.load_state_dict(state_dict)
