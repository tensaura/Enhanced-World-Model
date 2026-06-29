"""
Variational Autoencoder (β-VAE) vision model.

Encodes observations to a continuous Gaussian latent space via reparameterization.
Unlike VQ-VAE (discrete codebook), the latent is a smooth, differentiable distribution,
which couples naturally with RSSM's stochastic state.

References:
- VAE: https://arxiv.org/abs/1312.6114
- β-VAE: https://openreview.net/forum?id=Sy2fchgv (β > 1 encourages disentanglement)
"""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision import VisionModel
from utils.tags import VisionTag


class ResidualBlock(nn.Module):
    """Residual block shared with VQ-VAE architecture."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VAE(VisionModel):
    """
    β-Variational Autoencoder vision model.

    Architecture:
        Encoder: CNN → 2*embed_dim feature map (mean + log-variance)
        Reparameterize: z = μ + ε·σ   (ε ~ N(0,I))
        Decoder: CNN transpose → reconstructed observation

    Loss:
        reconstruction_loss (MSE) + β · KL(q(z|x) || p(z))

    Notes:
        - At inference (eval mode) the mean μ is used deterministically.
        - embed_dim is the per-spatial-location latent dimension; downstream
          modules pool over spatial dimensions to obtain a flat vector.
    """

    tags = frozenset({VisionTag.IMAGE_BASED})

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        output_dim: int = 3,
        embed_dim: int = 64,
        kernel_size: int = 4,
        stride: int = 2,
        beta: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        """
        Args:
            input_shape: (C, H, W) of the input image.
            hidden_dim: Number of channels in encoder/decoder hidden layers.
            output_dim: Number of channels in the reconstruction (usually = input_shape[0]).
            embed_dim: Latent dimension per spatial location (the μ/σ channels).
            kernel_size: Convolution kernel size.
            stride: Convolution stride (controls spatial downsampling).
            beta: Weight on KL term.  β=1 is standard VAE; β>1 is β-VAE.
        """
        super().__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta

        nb_channels = input_shape[0]

        # Encoder: image → (B, 2*embed_dim, H', W')
        # Two channels per latent dim: mean and log-variance.
        self.encoder = nn.Sequential(
            nn.Conv2d(nb_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 2 * embed_dim, kernel_size=1),
        )

        # Decoder: (B, embed_dim, H', W') → reconstructed image
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z via reparameterization trick; use mean at inference."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input: (B, C, H, W)

        Returns:
            recon: Reconstructed image (B, output_dim, H, W)
            z:     Sampled latent (B, embed_dim, H', W')
            loss:  β·KL loss (scalar); reconstruction MSE is added by WorldModel
        """
        stats = self.encoder(input)  # (B, 2*embed_dim, H', W')
        mu, log_var = stats.chunk(2, dim=1)  # each (B, embed_dim, H', W')

        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)

        # KL divergence: -0.5 * Σ(1 + log_var - μ² - exp(log_var))
        kl_loss = self.beta * (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean()

        return recon, z, kl_loss

    def encode(self, input: torch.Tensor, is_image_based: bool) -> torch.Tensor:
        """
        Returns:
            If is_image_based=True:  (B, embed_dim, H', W')
            If is_image_based=False: (B, embed_dim)   [mean-pooled]
        """
        stats = self.encoder(input)
        mu, log_var = stats.chunk(2, dim=1)
        z = self.reparameterize(mu, log_var)
        if not is_image_based:
            z = z.mean(dim=(2, 3))
        return z

    def export_hyperparams(self) -> dict[str, Any]:
        return {
            "input_shape": self.input_shape,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "embed_dim": self.embed_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "beta": self.beta,
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
