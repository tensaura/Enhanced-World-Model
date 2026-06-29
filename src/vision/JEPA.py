"""
JEPA (Joint Embedding Predictive Architecture) Vision Encoder.

Based on I-JEPA (Image-based Joint-Embedding Predictive Architecture) by Assran et al.
https://arxiv.org/abs/2301.08243

Key differences from reconstruction-based models (VQ-VAE):
- NO reconstruction loss - predicts in latent space only
- Learns task-relevant features, ignores visual noise
- Target encoder updated via EMA (exponential moving average)
- More sample efficient for RL tasks
"""

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision import VisionModel
from utils.tags import VisionTag


class ResidualBlock(nn.Module):
    """Residual block for encoder architecture."""

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


class JEPAEncoder(nn.Module):
    """
    CNN encoder for JEPA. Outputs spatial feature maps.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        embed_dim: int,
        kernel_size: int = 4,
        stride: int = 2,
        **_kwargs: Any,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class JEPAPredictor(nn.Module):
    """
    Predictor network for JEPA.
    Predicts target embeddings from context embeddings.
    Uses a lightweight architecture (smaller than encoder).
    """

    def __init__(self, embed_dim: int, predictor_dim: int = 256) -> None:
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Conv2d(embed_dim, predictor_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(predictor_dim, predictor_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(predictor_dim, embed_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class JEPA(VisionModel):
    """
    JEPA (Joint Embedding Predictive Architecture) Vision Model.

    Architecture:
        - Context encoder: Trainable CNN that encodes observations
        - Target encoder: EMA copy of context encoder (provides prediction targets)
        - Predictor: Lightweight network that predicts target from context

    Training:
        - Context encoder encodes input → z_context
        - Target encoder (no grad) encodes input → z_target
        - Predictor predicts z_target from z_context → z_pred
        - Loss = VICReg loss between z_pred and z_target

    Benefits over VQ-VAE:
        - No reconstruction needed → learns only task-relevant features
        - EMA target prevents collapse
        - More sample efficient
    """

    tags = frozenset({VisionTag.IMAGE_BASED, VisionTag.NO_RECONSTRUCTION})
    # JEPA does NOT reconstruct pixels - it predicts in latent space

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        embed_dim: int = 64,
        predictor_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 2,
        ema_decay: float = 0.996,
        vicreg_sim_weight: float = 25.0,
        vicreg_var_weight: float = 25.0,
        vicreg_cov_weight: float = 1.0,
        **_kwargs: Any,  # Ignore extra args like output_dim (used by VQ_VAE)
    ) -> None:
        """
        Initialize JEPA vision model.

        Args:
            input_shape: Shape of input images (C, H, W)
            hidden_dim: Hidden dimension in encoder
            embed_dim: Latent embedding dimension
            predictor_dim: Hidden dimension in predictor
            kernel_size: Convolution kernel size
            stride: Convolution stride
            ema_decay: EMA decay rate for target encoder (higher = slower update)
            vicreg_sim_weight: Weight for similarity term in VICReg loss
            vicreg_var_weight: Weight for variance term in VICReg loss
            vicreg_cov_weight: Weight for covariance term in VICReg loss
        """
        super().__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.ema_decay = ema_decay
        self.vicreg_sim_weight = vicreg_sim_weight
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight

        nb_channels = input_shape[0]
        assert len(input_shape) == 3, "JEPA supports 2D inputs only (images)"

        # Context encoder (trainable)
        self.context_encoder = JEPAEncoder(
            input_channels=nb_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Target encoder (EMA of context encoder)
        self.target_encoder = JEPAEncoder(
            input_channels=nb_channels,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Initialize target encoder with context encoder weights
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Freeze target encoder - only updated via EMA
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor network
        self.predictor = JEPAPredictor(embed_dim=embed_dim, predictor_dim=predictor_dim)

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """Update target encoder with EMA of context encoder."""
        for target_param, context_param in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=False
        ):
            target_param.data.mul_(self.ema_decay).add_(
                context_param.data, alpha=1 - self.ema_decay
            )

    def _vicreg_loss(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute VICReg (Variance-Invariance-Covariance Regularization) loss.

        VICReg prevents representation collapse without requiring:
        - Negative samples (like contrastive learning)
        - Momentum encoders being the only collapse prevention
        - Clustering or online assignments

        Args:
            z_pred: Predicted embeddings (B, D, H', W')
            z_target: Target embeddings (B, D, H', W')

        Returns:
            loss: Combined VICReg loss
            metrics: Dict with individual loss components
        """
        # Flatten spatial dimensions: (B, D, H', W') → (B, D)
        z_pred_flat = z_pred.mean(dim=(2, 3))
        z_target_flat = z_target.mean(dim=(2, 3))

        batch_size, embed_dim = z_pred_flat.shape
        device = z_pred_flat.device

        # === Invariance loss (similarity) ===
        sim_loss = F.mse_loss(z_pred_flat, z_target_flat)

        # Variance and covariance losses require batch_size > 1
        if batch_size > 1:
            # === Variance loss (prevent collapse) ===
            # Ensure variance of each embedding dimension is above threshold
            std_pred = torch.sqrt(z_pred_flat.var(dim=0) + 1e-4)
            std_target = torch.sqrt(z_target_flat.var(dim=0) + 1e-4)
            var_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

            # === Covariance loss (decorrelate dimensions) ===
            z_pred_centered = z_pred_flat - z_pred_flat.mean(dim=0)
            z_target_centered = z_target_flat - z_target_flat.mean(dim=0)

            cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
            cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)

            # Zero out diagonal (we only penalize off-diagonal covariance)
            cov_pred = cov_pred - torch.diag(torch.diag(cov_pred))
            cov_target = cov_target - torch.diag(torch.diag(cov_target))

            cov_loss = (cov_pred.pow(2).sum() + cov_target.pow(2).sum()) / embed_dim
        else:
            # Skip variance/covariance for batch_size=1 (undefined)
            var_loss = torch.tensor(0.0, device=device)
            cov_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = (
            self.vicreg_sim_weight * sim_loss
            + self.vicreg_var_weight * var_loss
            + self.vicreg_cov_weight * cov_loss
        )

        metrics = {
            "sim_loss": sim_loss.item(),
            "var_loss": var_loss.item() if batch_size > 1 else 0.0,
            "cov_loss": cov_loss.item() if batch_size > 1 else 0.0,
        }

        return loss, metrics

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Note: Unlike VQ-VAE, JEPA doesn't reconstruct the input.
        The first return value is the predicted embedding (not a reconstruction).

        Args:
            input: Input images (B, C, H, W)

        Returns:
            z_pred: Predicted target embedding (for compatibility, not a pixel reconstruction)
            z_context: Context encoder output — the latent used downstream (B, embed_dim, H', W')
            loss: VICReg loss for training
        """
        # Encode with context encoder
        z_context = self.context_encoder(input)

        # Get target embeddings (no gradient)
        with torch.no_grad():
            z_target = self.target_encoder(input)

        # Predict target from context
        z_pred = self.predictor(z_context)

        # Compute VICReg loss
        loss, _ = self._vicreg_loss(z_pred, z_target)

        # Update target encoder with EMA (only during training)
        if self.training:
            self._update_target_encoder()

        # Return prediction, context encoding, and loss
        # z_context is the latent used for memory/controller; z_pred is in-latent-space "recon"
        return z_pred, z_context, loss

    def encode(self, input: torch.Tensor, is_image_based: bool) -> torch.Tensor:
        """
        Encode observation to latent representation.

        Args:
            input: Input images (B, C, H, W)
            is_image_based: Whether to preserve spatial dimensions

        Returns:
            latent: Encoded representation
                - If is_image_based=True: (B, embed_dim, H', W')
                - If is_image_based=False: (B, embed_dim)
        """
        z = self.context_encoder(input)
        if not is_image_based:
            z = z.mean(dim=(2, 3))
        return z

    def export_hyperparams(self) -> dict[str, Any]:
        """Export hyperparameters for checkpoint saving."""
        return {
            "input_shape": self.input_shape,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "predictor_dim": self.predictor_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "ema_decay": self.ema_decay,
            "vicreg_sim_weight": self.vicreg_sim_weight,
            "vicreg_var_weight": self.vicreg_var_weight,
            "vicreg_cov_weight": self.vicreg_cov_weight,
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        """Save model state for checkpointing."""
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load model state from checkpoint."""
        self.load_state_dict(state_dict)
