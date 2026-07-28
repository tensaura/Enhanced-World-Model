import abc
from pathlib import Path
from typing import Any, Self

import torch
from torch.nn import Module


class Model(Module):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...] | torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass. Must be implemented in subclasses.

        :param input: Input tensor
        :return: Tuple of tensors (e.g., reconstruction, loss)
        """
        pass

    @abc.abstractmethod
    def export_hyperparams(self) -> dict:
        """
        Export model hyperparameters for saving or logging.
        """
        pass

    def save(self, path: Path, *_args: Any, **_kwargs: Any) -> None:
        """
        Save the model's weights and config.

        :param path: Path to save the model checkpoint (.pt file)
        """
        torch.save({"state_dict": self.state_dict(), "hyperparam": self.export_hyperparams()}, path)

    @abc.abstractmethod
    def load(cls, *args: Any, **kwargs: Any) -> None:
        """
        Load model from a checkpoint.
        Behaviour changes according to the type of model.
        """
        pass


def load_model(path: Path, available_classes: dict[str, type[Model]]) -> Model:
    """
    Load a model from a saved state.
    :param path: The path to the saved state.
    :param available_classes: The available classes.
    :return: The loaded model.
    """
    state = torch.load(path, weights_only=True)
    class_name = state.get("class_name", None)
    if class_name is None:
        raise Exception("No `class_name` field in saved model.")
    try:
        model_class = available_classes[class_name]
    except Exception as e:
        print(f"Class {class_name} not found in provided dict.")
        raise e
    model = model_class()
    model.load(path)
    return model
