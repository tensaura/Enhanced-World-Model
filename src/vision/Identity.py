from typing import Any, cast

import torch

from vision import VisionModel
from utils.tags import VisionTag


class Identity(VisionModel):
    """
    A dummy vision model that acts as an identity function for 1D vector-based environments.
    It mimics the output of a real vision model to be compatible with the WorldModel class.
    """

    tags = frozenset({VisionTag.VECTOR_BASED})

    def __init__(self, input_shape: tuple[int], **_kwargs: Any) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = input_shape[0]

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass-through: reconstruction = input, z = input, loss = 0
        zero = torch.tensor(0.0, device=input.device)
        return input, input, zero

    def encode(self, input: torch.Tensor, is_image_based: bool) -> torch.Tensor:
        if is_image_based:
            # Add dummy H, W dims for flatten_vision_latents
            return input.unsqueeze(-1).unsqueeze(-1)
        return input

    def export_hyperparams(self) -> dict[str, tuple[int]]:
        return {"input_shape": self.input_shape}

    def save_state(self) -> dict[str, torch.Tensor]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
