from typing import Any, cast

import torch
import torch.nn.functional as F

from vision import VisionModel
from utils.tags import VisionTag


class ResidualBlock(torch.nn.Module):
    """
    An implementation of a residual block.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VectorQuantizer(torch.nn.Module):
    """
    Implementation of the vector quantizer.
    """

    def __init__(self, num_embeddings: int, embed_dim: int, beta: float = 0.25) -> None:
        """
        Implements vector quantization layer (VQ) as described in the VQ-VAE paper.

        :param num_embeddings: Number of vectors in the codebook (K)
        :param embed_dim: Dimensionality of each embedding vector (D)
        :param beta: Weight for commitment loss (equation (3) in the paper)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.beta = beta

        # Codebook: (K, D)
        self.embedding = torch.nn.Parameter(torch.randn(num_embeddings, embed_dim) * 0.1)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of vector quantizer.

        :param z_e: Encoder output of shape (B, D, ...)
        :return: quantized tensor z_q, commitment loss, and indices
        """
        # Flatten input → (B*..., D)
        z_e_flat = z_e.permute(0, *range(2, z_e.ndim), 1).contiguous()
        flat_shape = z_e_flat.shape
        z_e_flat = z_e_flat.view(-1, self.embed_dim)  # (N, D)

        # Compute distances to embeddings: (N, K)
        distances = (
            torch.sum(z_e_flat**2, dim=1, keepdim=True)
            - 2 * z_e_flat @ self.embedding.t()
            + torch.sum(self.embedding**2, dim=1, keepdim=True).t()
        )
        # Get nearest code indices (N,)
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantized vectors: (N, D)
        z_q_flat = self.embedding[encoding_indices]

        # Reshape back to original input shape
        z_q = z_q_flat.view(*flat_shape)
        z_q = z_q.permute(0, -1, *range(1, z_e.ndim - 1)).contiguous()  # (B, D, ...)

        # Straight-through estimator (copy gradients)
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss
        loss = F.mse_loss(z_e.detach(), z_q) + self.beta * F.mse_loss(z_e, z_q.detach())

        return z_q_st, loss, encoding_indices.view(z_q.shape[0], *z_q.shape[2:])


class VectorQuantizerEMA(torch.nn.Module):
    """
    VQ-VAE v2: Exponential Moving Average codebook updates.
    Much more stable than the original VQ-VAE.
    """

    def __init__(
        self, num_embeddings: int, embed_dim: int, decay: float = 0.99, eps: float = 1e-5
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embed_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

        self.embedding: torch.Tensor
        self.cluster_size: torch.Tensor
        self.embed_avg: torch.Tensor

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e : (B, D, H, W)
        returns : z_q (quantized), loss, indices
        """

        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e_perm.view(-1, self.embed_dim)

        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.T
            + self.embedding.pow(2).sum(1)
        )

        indices = torch.argmin(distances, dim=1)

        z_q = self.embedding[indices].view_as(z_e_perm)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        z_q_st = z_e + (z_q - z_e).detach()

        if self.training:
            # EMA updates don't need gradients - wrap to prevent memory leak
            with torch.no_grad():
                encodings = torch.zeros(indices.size(0), self.num_embeddings, device=z_e.device)
                encodings.scatter_(1, indices.unsqueeze(1), 1)

                cluster_size = encodings.sum(0)
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                embed_sum = flat.t() @ encodings
                self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                )

                self.embedding.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        commitment_loss = torch.mean((z_q_st.detach() - z_e) ** 2)

        return z_q_st, commitment_loss, indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])


class VQ_VAE(VisionModel):
    """
    Implementation of the VQ-VAE model (https://arxiv.org/pdf/1711.00937).
    """

    tags = frozenset({VisionTag.IMAGE_BASED})

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        output_dim: int = 3,
        num_embed: int = 512,
        embed_dim: int = 64,
        kernel_size: int = 4,
        stride: int = 2,
        **_kwargs: Any,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride

        nb_channels = input_shape[0]
        # assert len(input_shape) == 3, "This version supports 2D inputs only (e.g. images)"

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                nb_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),  # → 32x32
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            torch.nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

        self.vq = VectorQuantizerEMA(num_embeddings=num_embed, embed_dim=embed_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            torch.nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),  # → 64x64
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                hidden_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=1
            ),  # → 128x128
        )

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(input)
        z_q, vq_loss, _ = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_q, vq_loss

    def encode(self, input: torch.Tensor, is_image_based: bool) -> torch.Tensor:
        z_e = self.encoder(input)
        z_q, _, _ = self.vq(z_e)
        return z_q

    def export_hyperparams(self) -> dict[str, tuple[int, ...] | int]:
        return {
            "input_shape": self.input_shape,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_embed": self.num_embed,
            "embed_dim": self.embed_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
