"""
Recurrent State-Space Model (RSSM).

The RSSM separates world-model dynamics into two parts:
  - A deterministic recurrent state h_t (GRU), which acts as a "highway"
    for gradient flow across timesteps.
  - A stochastic state z_t sampled from either:
      • the posterior q(z_t | h_t, o_t)  during training  (observation available)
      • the prior    p(z_t | h_t)         at inference/imagination (no observation)

The combined state [h_t, z_t] is what the controller observes.
The KL divergence KL(posterior || prior) is an auxiliary loss that encourages
the prior to approximate the posterior, enabling imagination without observations.

References:
- PlaNet:   https://arxiv.org/pdf/1811.04551
- DreamerV3: https://arxiv.org/pdf/2301.04104v1
"""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import MemoryModel


class RSSM(MemoryModel):
    """
    Recurrent State-Space Model memory module.

    Attributes exported to WorldModel:
        d_model = rnn_dim + stoch_dim  (controller's h_dim)
    """

    tags: frozenset = frozenset()

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 2,
        rnn_dim: int = 128,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        kl_weight: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        """
        Args:
            latent_dim:  Observation latent dimension (from vision encoder, o_t).
            action_dim:  Action dimension.
            rnn_dim:     Dimension of the deterministic GRU hidden state.
            stoch_dim:   Dimension of the stochastic latent z_t.
            hidden_dim:  Hidden units in prior / posterior MLPs.
            kl_weight:   Multiplier on the KL loss (β-style weighting).
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_dim = rnn_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        # Total output dimension returned by update_memory; used by WorldModel as h_dim.
        self.d_model = rnn_dim + stoch_dim

        # GRU cell: input = cat(z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(stoch_dim + action_dim, rnn_dim)

        # Prior:     p(z_t | h_t)          → 2*stoch_dim (μ and log-std)
        self.prior_mlp = nn.Sequential(nn.Linear(rnn_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stoch_dim))

        # Posterior: q(z_t | h_t, o_t)     → 2*stoch_dim
        self.posterior_mlp = nn.Sequential(
            nn.Linear(rnn_dim + latent_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 2 * stoch_dim)
        )

        # Project stochastic state back to observation-latent space for predict_next output.
        # This lets WorldModel compute MSE loss against the vision encoder's z_{t+1}.
        self.stoch_to_latent = nn.Linear(stoch_dim, latent_dim)

        # Persistent state buffers (set at first call, reset per episode)
        self.h_state: torch.Tensor | None = None  # (B, rnn_dim)
        self.z_state: torch.Tensor | None = None  # (B, stoch_dim)

        # KL loss from last update_memory call (exposed via get_extra_loss)
        self._last_kl_loss: torch.Tensor | None = None

    # ------------------------------------------------------------------ helpers

    def _init_states(self, batch_size: int, device: torch.device) -> None:
        self.h_state = torch.zeros(batch_size, self.rnn_dim, device=device)
        self.z_state = torch.zeros(batch_size, self.stoch_dim, device=device)

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split 2*stoch_dim tensor into (mean, std); std > 0 via softplus."""
        mu, raw_std = stats.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1  # minimum std of 0.1 for numerical stability
        return mu, std

    @staticmethod
    def _kl_divergence(
        mu_post: torch.Tensor, std_post: torch.Tensor, mu_prior: torch.Tensor, std_prior: torch.Tensor
    ) -> torch.Tensor:
        """
        KL(q || p) where both are diagonal Gaussians.
        Averaged over batch and latent dimensions.
        """
        kl = (
            torch.log(std_prior / std_post)
            + (std_post.pow(2) + (mu_post - mu_prior).pow(2)) / (2.0 * std_prior.pow(2))
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def _sample(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample during training; use mean deterministically at inference."""
        if self.training:
            return mu + std * torch.randn_like(std)
        return mu

    # ------------------------------------------------------------------ public API

    def update_memory(self, o_t: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """
        Advance the recurrent state and sample z_t from the posterior.

        Args:
            o_t:    Current observation latent from vision encoder (B, latent_dim).
            a_prev: Previous action (B, action_dim).

        Returns:
            combined: cat(h_t, z_t) of shape (B, rnn_dim + stoch_dim).
                      This is what the controller and predict_next receive as h_t.
        """
        B, device = o_t.size(0), o_t.device

        if self.h_state is None or self.h_state.size(0) != B:
            self._init_states(B, device)
        assert self.h_state is not None and self.z_state is not None

        # ── Deterministic update ──────────────────────────────────────────────
        gru_input = torch.cat([self.z_state, a_prev], dim=-1)  # (B, stoch+action)
        h_t = self.gru(gru_input, self.h_state)  # (B, rnn_dim)

        # ── Prior p(z_t | h_t) ───────────────────────────────────────────────
        mu_prior, std_prior = self._split_stats(self.prior_mlp(h_t))

        # ── Posterior q(z_t | h_t, o_t) ─────────────────────────────────────
        mu_post, std_post = self._split_stats(self.posterior_mlp(torch.cat([h_t, o_t], dim=-1)))

        # ── KL divergence loss ───────────────────────────────────────────────
        kl = self._kl_divergence(mu_post, std_post, mu_prior, std_prior)
        self._last_kl_loss = self.kl_weight * kl

        # ── Sample stochastic state ──────────────────────────────────────────
        z_t = self._sample(mu_post, std_post)  # (B, stoch_dim)

        # Update persistent state (detach to prevent BPTT through history)
        self.h_state = h_t.detach()
        self.z_state = z_t.detach()

        return torch.cat([h_t, z_t], dim=-1)  # (B, rnn_dim + stoch_dim)

    def predict_next(self, _o_t: torch.Tensor, a_t: torch.Tensor, h_combined: torch.Tensor) -> torch.Tensor:
        """
        Predict the next observation latent using the prior (no observation needed).
        Used for world-model imagination and the memory MSE training loss.

        Args:
            _o_t:       Current observation latent (unused; kept for interface compatibility).
            a_t:        Current action (B, action_dim).
            h_combined: Combined state from update_memory (B, rnn_dim + stoch_dim).

        Returns:
            z_next_latent: Predicted next observation latent (B, latent_dim).
        """
        h_t = h_combined[..., : self.rnn_dim]
        z_stoch = h_combined[..., self.rnn_dim :]

        # Step GRU forward with current action
        gru_input = torch.cat([z_stoch, a_t], dim=-1)
        h_next = self.gru(gru_input, h_t)

        # Sample from prior
        mu_prior, std_prior = self._split_stats(self.prior_mlp(h_next))
        z_next = self._sample(mu_prior, std_prior)  # (B, stoch_dim)

        # Project to observation-latent space for MSE comparison with vision output
        return self.stoch_to_latent(z_next)  # (B, latent_dim)

    def get_extra_loss(self) -> torch.Tensor | None:
        """Return KL divergence from the last update_memory call."""
        return self._last_kl_loss

    def reset_env_memory(self, env_idx: int | torch.Tensor) -> None:
        env_idx = int(env_idx)
        if self.h_state is not None:
            self.h_state[env_idx].zero_()
        if self.z_state is not None:
            self.z_state[env_idx].zero_()

    def forward(self, z_t: torch.Tensor, a_prev: torch.Tensor, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward step: update memory then predict next latent."""
        h_combined = self.update_memory(z_t, a_prev)
        z_next = self.predict_next(z_t, a_t, h_combined)
        return z_next, h_combined

    def export_hyperparams(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "rnn_dim": self.rnn_dim,
            "stoch_dim": self.stoch_dim,
            "hidden_dim": self.hidden_dim,
            "kl_weight": self.kl_weight,
            "d_model": self.d_model,  # exported so that patch_load restores d_model correctly
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
