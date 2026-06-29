"""Neural network components for DreamerV3-lite.

    - ``MLP``            : SiLU + LayerNorm multilayer perceptron used everywhere.
    - ``ConvEncoder`` /
      ``ConvDecoder``    : image observation encoder/decoder (64x64).
    - ``MLPEncoder`` /
      ``MLPDecoder``     : vector observation encoder/decoder (symlog space).
    - ``RSSM``           : recurrent state-space model with categorical latents.
    - ``RewardHead`` /
      ``ContinueHead``   : predict reward (two-hot symlog) and episode continuation.
    - ``Actor`` / ``Critic`` : policy and value functions trained in imagination.

State dictionaries used throughout have keys ``deter`` (B, deter_dim),
``stoch`` (B, stoch_dim) and ``logits`` (B, num_categoricals, num_classes).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributions as D
import torch.nn as nn

from dreamer.utils import OneHotCategoricalST, TwoHotSymlog


def _mlp_layers(in_dim: int, hidden: int, layers: int, norm: bool) -> list[nn.Module]:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        if norm:
            mods.append(nn.LayerNorm(hidden))
        mods.append(nn.SiLU())
        d = hidden
    return mods


class MLP(nn.Module):
    """SiLU + LayerNorm MLP with a linear output head."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2, norm: bool = True
    ) -> None:
        super().__init__()
        mods = _mlp_layers(in_dim, hidden, layers, norm)
        mods.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------- encoders


class ConvEncoder(nn.Module):
    """Maps a 64x64 image to a flat embedding via four stride-2 conv layers."""

    def __init__(self, channels: int, depth: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(depth, 2 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.embed_dim = 8 * depth * 4 * 4  # 4x4 spatial after four /2 downsamples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, 64, 64) → (N, embed_dim)
        h = self.net(x)
        return h.reshape(h.shape[0], -1)


class MLPEncoder(nn.Module):
    """Maps a (symlog) vector observation to an embedding."""

    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 2) -> None:
        super().__init__()
        self.net = MLP(obs_dim, hidden, hidden=hidden, layers=layers)
        self.embed_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------- decoders


class ConvDecoder(nn.Module):
    """Reconstructs a 64x64 image from the latent feature vector."""

    def __init__(self, feat_dim: int, channels: int, depth: int = 32) -> None:
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(feat_dim, 8 * depth * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(8 * depth, 4 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(2 * depth, depth, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(depth, channels, 4, stride=2, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, feat_dim) → (N, C, 64, 64)
        h = self.fc(feat).reshape(-1, 8 * self.depth, 4, 4)
        return self.net(h)


class MLPDecoder(nn.Module):
    """Reconstructs a (symlog) vector observation from the latent feature vector."""

    def __init__(self, feat_dim: int, obs_dim: int, hidden: int = 256, layers: int = 2) -> None:
        super().__init__()
        self.net = MLP(feat_dim, obs_dim, hidden=hidden, layers=layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# -------------------------------------------------------------------------- RSSM


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical stochastic latents (DreamerV3)."""

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int = 256,
        num_categoricals: int = 32,
        num_classes: int = 32,
        hidden: int = 256,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.stoch_dim = num_categoricals * num_classes
        self.unimix = unimix

        # Pre-GRU projection of (stoch, action).
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.gru = nn.GRUCell(hidden, deter_dim)
        # Prior  p(z_t | h_t).
        self.img_out = nn.Sequential(nn.Linear(deter_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.prior_logits = nn.Linear(hidden, self.stoch_dim)
        # Posterior q(z_t | h_t, e_t).
        self.obs_out = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.post_logits = nn.Linear(hidden, self.stoch_dim)

    @property
    def feat_dim(self) -> int:
        return self.deter_dim + self.stoch_dim

    def initial(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
            "logits": torch.zeros(
                batch_size, self.num_categoricals, self.num_classes, device=device
            ),
        }

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def get_dist(self, logits: torch.Tensor) -> OneHotCategoricalST:
        return OneHotCategoricalST(logits, unimix=self.unimix)

    def _sample_stoch(self, logits: torch.Tensor) -> torch.Tensor:
        sample = self.get_dist(logits).sample()  # (B, num_cat, num_classes)
        return sample.reshape(sample.shape[0], -1)

    def img_step(
        self, prev_state: dict[str, torch.Tensor], prev_action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Advance the prior one step (used in imagination, no observation)."""
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.img_in(x)
        deter = self.gru(x, prev_state["deter"])
        logits = self.prior_logits(self.img_out(deter)).reshape(
            -1, self.num_categoricals, self.num_classes
        )
        stoch = self._sample_stoch(logits)
        return {"deter": deter, "stoch": stoch, "logits": logits}

    def obs_step(
        self,
        prev_state: dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Advance one step with an observation; returns (posterior, prior) states."""
        # Reset state and action at episode starts.
        mask = (1.0 - is_first).unsqueeze(-1)  # (B, 1)
        prev_action = prev_action * mask
        prev_state = {k: v * mask if k != "logits" else v for k, v in prev_state.items()}

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], dim=-1)
        logits = self.post_logits(self.obs_out(x)).reshape(
            -1, self.num_categoricals, self.num_classes
        )
        stoch = self._sample_stoch(logits)
        post = {"deter": prior["deter"], "stoch": stoch, "logits": logits}
        return post, prior


# --------------------------------------------------------------------------- heads


class RewardHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 2, num_bins: int = 255):
        super().__init__()
        self.net = MLP(feat_dim, num_bins, hidden=hidden, layers=layers)

    def forward(self, feat: torch.Tensor) -> TwoHotSymlog:
        return TwoHotSymlog(self.net(feat))


class ContinueHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.net = MLP(feat_dim, 1, hidden=hidden, layers=layers)

    def forward(self, feat: torch.Tensor) -> D.Bernoulli:
        return D.Bernoulli(logits=self.net(feat).squeeze(-1))


# ----------------------------------------------------------------- actor / critic


class SquashedNormal:
    """Tanh-squashed diagonal Gaussian for continuous actions, in ``[-1, 1]``."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.base = D.Normal(mean, std)
        self.mean = mean

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.base.rsample())

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(action, -0.999, 0.999)
        raw = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh, MPS-safe
        log_prob = self.base.log_prob(raw).sum(-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return log_prob

    def entropy(self) -> torch.Tensor:
        return self.base.entropy().sum(-1)


class Actor(nn.Module):
    """Policy over the latent feature: one-hot categorical (discrete) or squashed
    Gaussian (continuous)."""

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        discrete: bool,
        hidden: int = 256,
        layers: int = 2,
        unimix: float = 0.01,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        self.unimix = unimix
        self.min_std = min_std
        self.max_std = max_std
        out_dim = action_dim if discrete else 2 * action_dim
        self.net = MLP(feat_dim, out_dim, hidden=hidden, layers=layers)

    def forward(self, feat: torch.Tensor) -> Any:
        out = self.net(feat)
        if self.discrete:
            return OneHotCategoricalST(out, unimix=self.unimix)
        mean, std = out.chunk(2, dim=-1)
        mean = torch.tanh(mean)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(std)
        return SquashedNormal(mean, std)


class Critic(nn.Module):
    """Two-hot symlog value function over the latent feature."""

    def __init__(self, feat_dim: int, hidden: int = 256, layers: int = 2, num_bins: int = 255):
        super().__init__()
        self.net = MLP(feat_dim, num_bins, hidden=hidden, layers=layers)

    def forward(self, feat: torch.Tensor) -> TwoHotSymlog:
        return TwoHotSymlog(self.net(feat))
