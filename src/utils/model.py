from argparse import Namespace
import gymnasium as gym
import torch
from gymnasium.spaces import Space

from utils.gym_tools import get_env_info
from WorldModel import WorldModel


def create_world_model(
    args: Namespace,
    vision_registry: dict,
    memory_registry: dict,
    controller_registry: dict,
    device: torch.device,
    messages: dict[str, list[str]] | None = None,
) -> tuple[WorldModel, Space, Space, dict[str, list[str]]]:
    # Useful to allow passing local variables.
    if messages is None:
        log_messages: dict[str, list[str]] = {"info": [], "warning": [], "error": []}
    else:
        log_messages = messages

    obs_space, action_space, is_image_based, is_discrete = get_env_info(args.env)

    obs_shape = obs_space.shape
    assert obs_shape is not None

    vision_model = vision_registry.get(args.vision, None)
    if vision_model is None:
        raise Exception(
            f"Vision model {args.vision} is not available.\nAvailable models: {list(vision_registry.keys())}"
        )

    if is_image_based:
        log_messages["info"].append("Detected image-based environment.")
        # (H, W, C) -> (C, H, W)
        input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        vision_args = {"output_dim": input_shape[0], "embed_dim": 64}
        if "image_based" not in vision_model.tags:
            log_messages["warning"].append(f"Vision model {args.vision} is not image-based.")
    else:
        log_messages["info"].append("Detected vector-based environment.")
        input_shape = obs_shape
        vision_args = {"embed_dim": obs_shape[0]}
        if "vector_based" not in vision_model.tags:
            log_messages["warning"].append(f"Vision model {args.vision} is not vector-based.")

    memory_model = memory_registry.get(args.memory, None)
    if memory_model is None:
        raise Exception(
            f"Memory model {args.memory} is not available.\nAvailable models: {list(memory_registry.keys())}"
        )

    # Configure memory and controller based on environment
    controller_model = controller_registry.get(args.controller, None)
    if controller_model is None:
        raise Exception(
            f"Controller model {args.controller} is not available.\nAvailable models: {list(controller_registry.keys())}"
        )

    if is_discrete:
        assert isinstance(action_space, gym.spaces.Discrete)
        action_dim = action_space.n  # action_space.n is actually the number of possible values
        if "discrete" not in controller_model.tags:
            log_messages["warning"].append(
                f"Controller model {args.controller} is not suitable for discrete action space."
            )
    else:
        assert action_space.shape
        action_dim = action_space.shape[0]
        if "continuous" not in controller_model.tags:
            log_messages["warning"].append(
                f"Controller model {args.controller} is not suitable for continuous action space."
            )

    memory_args = {"d_model": 128, "latent_dim": vision_args["embed_dim"], "action_dim": action_dim, "nhead": 8}
    controller_args = {"action_dim": action_dim}

    world_model = WorldModel(
        vision_model=vision_model,
        memory_model=memory_model,
        controller_model=controller_model,  # StochasticController,  #ModelPredictiveController,
        input_shape=input_shape,
        vision_args=vision_args,
        memory_args=memory_args,
        controller_args=controller_args,
    ).to(device)

    return world_model, obs_space, action_space, log_messages
