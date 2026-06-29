"""Math utilities for DreamerV3-lite.

Implements the tricks that make DreamerV3 robust across reward/return scales and
stable to optimise:

    - ``symlog`` / ``symexp``: squashing transform applied to vector observations,
      rewards and value targets so the networks regress over a compressed range.
    - ``TwoHotSymlog``: a categorical distribution over a fixed set of symlog-spaced
      bins, used as the reward and value head. Two-hot regression turns scalar
      regression into a classification problem that is far less sensitive to scale.
    - ``OneHotCategoricalST``: a straight-through one-hot categorical used for the
      RSSM stochastic latent and the discrete actor. Sampling uses the Gumbel-argmax
      trick so it works on every backend (no ``multinomial`` dependency on MPS).
    - ``lambda_return``: the TD(λ) return used as the critic target in imagination.

References: DreamerV3 (https://arxiv.org/pdf/2301.04104v1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- symlog


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Signed logarithm: sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


# --------------------------------------------------------------- two-hot value head


class TwoHotSymlog:
    """Categorical distribution over symlog-spaced bins for scalar regression.

    The network outputs ``logits`` over ``num_bins`` bins whose centres are evenly
    spaced in symlog space over ``[low, high]``. The distribution mean (in real
    space) is ``symexp(sum(softmax(logits) * bins))``. Targets are encoded with a
    two-hot vector over the same bins and trained with cross-entropy via
    :meth:`log_prob`.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        low: float = -20.0,
        high: float = 20.0,
        num_bins: int = 255,
    ) -> None:
        self.logits = logits
        self.low = low
        self.high = high
        self.num_bins = num_bins
        self.probs = F.softmax(logits, dim=-1)
        self.bins = torch.linspace(low, high, num_bins, device=logits.device, dtype=logits.dtype)

    def mean(self) -> torch.Tensor:
        """Expected value mapped back to real space via symexp."""
        symlog_value = (self.probs * self.bins).sum(dim=-1, keepdim=True)
        return symexp(symlog_value)

    def _two_hot(self, x: torch.Tensor) -> torch.Tensor:
        """Encode targets ``x`` (real space) as two-hot vectors over symlog bins.

        Bins are uniformly spaced, so the surrounding bin index is computed
        analytically (no ``torch.bucketize`` — keeps it MPS-compatible).
        """
        x = symlog(x)
        x = torch.clamp(x, self.low, self.high)
        step = (self.high - self.low) / (self.num_bins - 1)
        pos = (x - self.low) / step  # continuous bin position
        idx = torch.clamp(torch.floor(pos).long(), 0, self.num_bins - 2)
        weight_upper = pos - idx.to(pos.dtype)
        weight_lower = 1.0 - weight_upper
        two_hot = torch.zeros_like(self.logits)
        two_hot.scatter_(-1, idx.unsqueeze(-1), weight_lower.unsqueeze(-1))
        two_hot.scatter_add_(-1, (idx + 1).unsqueeze(-1), weight_upper.unsqueeze(-1))
        return two_hot

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-entropy between the two-hot target and the predicted distribution."""
        target = self._two_hot(x)
        log_probs = F.log_softmax(self.logits, dim=-1)
        return (target * log_probs).sum(dim=-1)


# ------------------------------------------------------------- categorical latents


class OneHotCategoricalST:
    """One-hot categorical with straight-through gradients and Gumbel sampling.

    ``logits`` has shape ``(..., num_classes)``. Sampling returns a one-hot vector
    of the same shape; gradients flow straight through the sample to the logits.
    A small uniform mixture (``unimix``) is blended into the probabilities as in
    DreamerV3 to keep the distribution from collapsing.
    """

    def __init__(self, logits: torch.Tensor, unimix: float = 0.01) -> None:
        if unimix > 0.0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1.0 - unimix) * probs + unimix * uniform
            logits = torch.log(probs + 1e-8)
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)

    def sample(self) -> torch.Tensor:
        """Gumbel-argmax sample with a straight-through estimator."""
        # Gumbel(0, 1) noise — MPS-safe (no multinomial).
        noise = torch.rand_like(self.logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(noise))
        index = torch.argmax(self.logits + gumbel, dim=-1)
        sample = F.one_hot(index, self.logits.shape[-1]).to(self.logits.dtype)
        # Straight-through: forward = hard sample, backward = soft probs.
        return sample + (self.probs - self.probs.detach())

    def mode(self) -> torch.Tensor:
        """Greedy one-hot action (argmax of the logits) for evaluation."""
        index = torch.argmax(self.logits, dim=-1)
        return F.one_hot(index, self.logits.shape[-1]).to(self.logits.dtype)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log-probability of a one-hot ``value`` under the distribution."""
        log_probs = F.log_softmax(self.logits, dim=-1)
        return (value * log_probs).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        log_probs = F.log_softmax(self.logits, dim=-1)
        return -(self.probs * log_probs).sum(dim=-1)


def categorical_kl(logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) for (independent) categorical logits, summed over the last dim."""
    probs_q = F.softmax(logits_q, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    return (probs_q * (log_q - log_p)).sum(dim=-1)


# --------------------------------------------------------------------- λ-returns


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute TD(λ) returns over an imagined trajectory.

    Args:
        rewards:   ``(H, B)`` predicted rewards for steps ``1..H``.
        values:    ``(H + 1, B)`` critic values for steps ``1..H+1`` (bootstrap last).
        continues: ``(H, B)`` discount factors ``gamma * P(continue)`` for steps ``1..H``.
        lambda_:   GAE-style mixing parameter.

    Returns:
        ``(H, B)`` λ-returns for steps ``1..H``.
    """
    horizon = rewards.shape[0]
    outputs: list[torch.Tensor] = []
    last = values[-1]
    for t in reversed(range(horizon)):
        bootstrap = (1.0 - lambda_) * values[t + 1] + lambda_ * last
        last = rewards[t] + continues[t] * bootstrap
        outputs.append(last)
    return torch.stack(list(reversed(outputs)), dim=0)
