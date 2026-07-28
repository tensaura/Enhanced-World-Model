from typing import Any

import gymnasium as gym
import numpy as np
import torch

import vision
import memory
import controller
from utils.registry import discover_modules
from WorldModel import WorldModel

VISION_REGISTRY: dict = discover_modules(vision)
MEMORY_REGISTRY: dict = discover_modules(memory)
CONTROLLER_REGISTRY: dict = discover_modules(controller)

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

torch.set_default_device(device)


def evaluate(
    model: WorldModel, env_name: str, num_episodes: int = 5, device: torch.device = device, render_mode: str = "human"
) -> list[dict[str, Any]]:
    """Evaluate trained model and show sample frames."""
    env = gym.make(env_name, render_mode=render_mode)
    model.eval()

    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    is_img = len(obs_shape) == 3
    results = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        frames = []

        # Reset model memory between episodes — covers all memory types:
        #   LSTMMemory:          h_state, c_state
        #   TemporalTransformer: seq_buffer, seq_lengths
        #   RSSM:                h_state, z_state
        model.a_prev = None
        for attr in ("h_state", "c_state", "z_state", "seq_buffer", "seq_lengths"):
            if hasattr(model.memory, attr):
                setattr(model.memory, attr, None)

        while not done:
            # Prepare observation
            if is_img:
                obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            else:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            obs_t = obs_t.to(device)

            # Get action
            with torch.no_grad():
                action = model(obs_t, env.action_space, is_img, return_losses=False)
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

            if steps % 10 == 0:
                frames.append(env.render())

        results.append({"reward": total_reward, "steps": steps, "frames": frames})
        print(f"Episode {ep + 1}: Reward = {total_reward:.1f}, Steps = {steps}")

    env.close()

    rewards = [r["reward"] for r in results]
    print(f"\nMean Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained WorldModel")
    parser.add_argument("--load-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., CartPole-v1)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render-mode", type=str, default="human", help="Render mode")
    args = parser.parse_args()

    # Create environment to get observation/action spaces
    env = gym.make(args.env)
    obs_space = env.observation_space
    action_space = env.action_space
    obs_shape = obs_space.shape
    assert obs_shape is not None
    is_img = len(obs_shape) == 3
    input_shape = obs_shape if is_img else (obs_shape[0],)
    env.close()

    # Load the model
    saved_dict = torch.load(args.load_path, weights_only=False, map_location=device)

    # Reconstruct WorldModel from checkpoint
    vision_model = VISION_REGISTRY[saved_dict["vision_model"]]
    memory_model = MEMORY_REGISTRY[saved_dict["memory_model"]]
    controller_model = CONTROLLER_REGISTRY[saved_dict["controller_model"]]

    print(vision_model, "\n", memory_model, "\n", controller_model)
    saved_dict["vision_args"] = {}
    saved_dict["controller_args"] = {"action_dim": saved_dict["controller_args"]["action_dim"]}

    world_model = WorldModel(
        vision_model=vision_model,
        memory_model=memory_model,
        controller_model=controller_model,
        input_shape=input_shape,
        vision_args=saved_dict["vision_args"],
        memory_args=saved_dict["memory_args"],
        controller_args=saved_dict["controller_args"],
    ).to(device)

    world_model.load(args.load_path, obs_space=obs_space, action_space=action_space, device=device)

    results = evaluate(world_model, args.env, num_episodes=args.episodes, render_mode=args.render_mode)
