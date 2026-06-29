from abc import abstractmethod
from typing import Any
import torch

from Model import Model

REGISTRY = {}


class VisionModel(Model):
    """
    Base class for vision (V) models.

    Responsible for encoding observations into latent representations.

    Attributes:
        input_shape: Shape of input observations (e.g., (3, 96, 96) for images or (4,) for vectors)
        embed_dim: Dimension of the encoded latent representation
        is_reconstruction_based: Whether this model outputs pixel reconstructions (True for VQ_VAE)
            or latent predictions (False for JEPA). Used to determine loss computation.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__] = cls

    @abstractmethod
    def forward(
        self, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode and decode observation.

        Args:
            input: Raw observation, shape depends on input_shape
                   - Images: (B, C, H, W)
                   - Vectors: (B, D)

        Returns:
            reconstruction: Reconstructed observation (or latent prediction for JEPA),
                            same shape as input
            z: Encoded latent representation (spatial for images: (B, embed_dim, H', W'))
            loss: Reconstruction/VQ/VICReg loss for training, scalar
        """
        pass

    @abstractmethod
    def encode(self, input: torch.Tensor, is_image_based: bool) -> torch.Tensor:
        """
        Encode observation to latent representation.

        Args:
            input: Raw observation, shape depends on input_shape
                   - Images: (B, C, H, W)
                   - Vectors: (B, D)
            is_image_based: Whether the input is image-based (affects output shape)

        Returns:
            latent: Encoded latent representation
                   - If is_image_based=True: (B, embed_dim, H', W')
                   - If is_image_based=False: (B, embed_dim)
        """
        pass
