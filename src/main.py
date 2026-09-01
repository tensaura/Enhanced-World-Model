import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

import controller
import memory
import vision
from inference import evaluate
from train import train
from utils.cli import CLI
from utils.gym_tools import auto_batch_size
from utils.model import create_world_model
from utils.registry import discover_modules

VISION_REGISTRY: dict = discover_modules(vision)
MEMORY_REGISTRY: dict = discover_modules(memory)
CONTROLLER_REGISTRY: dict = discover_modules(controller)

# Anomaly detection is a debugging aid that makes every backward pass 2-3x slower.
# Enable it only when EWM_DETECT_ANOMALY=1 is set, not by default.
torch.autograd.set_detect_anomaly(os.environ.get("EWM_DETECT_ANOMALY") == "1")

device: torch.device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("train.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument(
        "--ui", action="store_true", help="Launch the Gradio interface instead of training directly."
    )
    interface_group.add_argument("--cli", action="store_true", help="Runs the command line interface.")
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",  # "CarRacing-v3",
        help="The Gym environment to use.",
    )  # CartPole-v1
    parser.add_argument("--vision", type=str, default="Identity")
    parser.add_argument("--memory", type=str, default="TemporalTransformer")
    parser.add_argument("--controller", type=str, default="DeepDiscreteController")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Epochs without mean-reward improvement before early stopping. 0 = disabled.",
    )
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--render-mode", type=str, default="rgb_array")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="./saved_models/")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--patch-load-path", type=str, default="", help="Path to model to load on top.")
    parser.add_argument(
        "--patch",
        type=str,
        default="vmc",
        choices=["v", "m", "c", "vm", "vc", "mc", "vmc"],
        help="The sub models to patch.",
    )
    parser.add_argument("--save-freq", type=int, default=10, help="Frequency of saving model checkpoints.")
    parser.add_argument("--log-freq", type=int, default=10, help="Frequency of logging training progress.")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging.")

    # Other args
    parser.add_argument(
        "--manual-mode-delay", type=float, default=0.05, help="Delay between each step during manual training."
    )

    # PPO arguments
    parser.add_argument("--rollout-steps", type=int, default=128, help="Number of rollout steps.")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Number of epochs for PPO training.")
    parser.add_argument("--ppo-lr", type=float, default=3e-4, help="Learning rate for PPO training.")
    parser.add_argument("--ppo-batch-size", type=int, default=64, help="Batch size for PPO training.")
    parser.add_argument("--ppo-clip-range", type=float, default=0.2, help="Clipping parameter for PPO training.")
    parser.add_argument("--ppo-range-vf", type=float, default=None, help="Value function for PPO training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma parameter for GAE in PPO training.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda parameter for GAE in PPO training.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient in PPO training.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient in PPO training.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm in PPO training.")
    parser.add_argument("--no-train-world-model", action="store_true", help="Train the world model.")
    parser.add_argument("--world-model-epochs", type=int, default=1, help="Number of epochs for world model training.")

    # Inference arguments
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--infer", action="store_true", help="Enable inference mode.")

    # Algorithm selection + DreamerV3-lite arguments
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dreamer"],
        help="Training algorithm: 'ppo' (modular world model) or 'dreamer' (latent imagination).",
    )
    parser.add_argument("--total-steps", type=int, default=100_000, help="[dreamer] Total environment steps.")
    parser.add_argument("--prefill", type=int, default=2000, help="[dreamer] Random steps before learning starts.")
    parser.add_argument("--train-every", type=int, default=5, help="[dreamer] Env steps between train steps.")
    parser.add_argument("--seq-len", type=int, default=50, help="[dreamer] Replay sequence length.")
    parser.add_argument("--dreamer-batch", type=int, default=16, help="[dreamer] World-model batch size.")
    parser.add_argument("--action-repeat", type=int, default=2, help="[dreamer] Frame-skip / action repeat.")
    parser.add_argument("--horizon", type=int, default=15, help="[dreamer] Imagination horizon.")
    parser.add_argument("--deter-dim", type=int, default=256, help="[dreamer] RSSM GRU state size.")
    parser.add_argument("--cnn-depth", type=int, default=32, help="[dreamer] Base conv channels.")
    parser.add_argument("--entropy-scale", type=float, default=1e-3, help="[dreamer] Actor entropy bonus scale.")

    args = parser.parse_args()
    if args.cli:
        CLI(args, VISION_REGISTRY, MEMORY_REGISTRY, CONTROLLER_REGISTRY)
    if args.batch_size.isdigit():
        env_batch_size = int(args.batch_size)
    else:
        env_batch_size = auto_batch_size(args.env)
        logger.info(f"Auto batch size selected: {env_batch_size} parallel environments.")

    logger.info(f"Running with {env_batch_size} parallel environments.")

    if args.ui:
        import subprocess
        import sys

        # Launch the Gradio app and exit
        subprocess.run([sys.executable, "src/interface/app.py"])
        return

    logger.info(str(args))
    logger.info(f"Using device: {device}")

    if args.algo == "dreamer":
        from dreamer.train import train_dreamer

        logger.info("Running DreamerV3-lite (latent imagination).")
        train_dreamer(
            env_name=args.env,
            device=device,
            total_steps=args.total_steps,
            prefill=args.prefill,
            train_every=args.train_every,
            seq_len=args.seq_len,
            batch_size=args.dreamer_batch,
            action_repeat=args.action_repeat,
            horizon=args.horizon,
            deter_dim=args.deter_dim,
            cnn_depth=args.cnn_depth,
            entropy_scale=args.entropy_scale,
            seed=args.seed,
            save_path=Path(args.save_path),
            load_path=args.load_path,
            use_tensorboard=args.tensorboard,
        )
        return

    try:
        envs = gym.make_vec(args.env, num_envs=env_batch_size, render_mode=args.render_mode)

        log_messages: dict[str, list[str]] = {"info": [], "warning": [], "error": []}
        try:
            world_model, obs_space, action_space, log_messages = create_world_model(
                args, VISION_REGISTRY, MEMORY_REGISTRY, CONTROLLER_REGISTRY, device, log_messages
            )
        except Exception as e:
            raise e
        finally:
            for info_message in log_messages["info"]:
                logger.info(info_message)

            for warning_message in log_messages["warning"]:
                logger.warning(warning_message)

            for error_message in log_messages["error"]:
                logger.error(error_message)

        logger.info(f"Vision model: {world_model.vision.__class__.__name__}")
        logger.info(f"Memory model: {world_model.memory.__class__.__name__}")
        logger.info(f"Controller model: {world_model.controller.__class__.__name__}")

        if args.load_path:
            print(f"Loading model from {args.load_path}")
            world_model.load(args.load_path, obs_space=obs_space, action_space=action_space, device=device)

        if args.patch_load_path:
            patches = []
            if "v" in args.patch:
                patches.append("vision")
            if "m" in args.patch:
                patches.append("memory")
            if "c" in args.patch:
                patches.append("controller")

            logger.info(f"Patching {', '.join(patches)} of model with {args.patch_load_path}")
            world_model.patch_load(
                args.patch_load_path, args.patch, obs_space=obs_space, action_space=action_space, device=device
            )

        if args.infer:
            if not args.load_path or args.patch_load_path:
                logger.warning("World model is not initialized. Infering with random weights.")
            world_model.eval()
            evaluate(world_model, args.env, num_episodes=args.episodes, render_mode=args.render_mode)
        else:
            if args.load_path:
                for param in world_model.parameters():
                    param.requires_grad = True
                world_model.train()
            train(
                world_model,
                envs,
                max_iter=args.epochs,
                device=device,
                rollout_steps=args.rollout_steps,
                num_ppo_epochs=args.ppo_epochs,
                batch_size=args.ppo_batch_size,
                clip_range=args.ppo_clip_range,
                clip_range_vf=args.ppo_range_vf,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                learning_rate=args.lr,
                policy_lr=args.ppo_lr,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
                train_world_model=not args.no_train_world_model,
                world_model_epochs=args.world_model_epochs,
                patience=args.patience,
                use_tensorboard=args.tensorboard,
                save_path=Path(args.save_path),
                save_freq=args.save_freq,
                log_freq=args.log_freq,
                render_mode=args.render_mode,
            )

            save_name = Path(f"{args.save_path}{args.env}_{datetime.now().isoformat(timespec='minutes')}.pt")
            world_model.save(save_name, obs_space=obs_space, action_space=action_space)

            logger.info(f"Model saved to {save_name}")

        envs.close()
        logger.info("Environment closed.")
    except Exception as e:
        logger.exception(f"Exception during training: {e}")
        raise e


if __name__ == "__main__":
    main()
