"""PPO (Proximal Policy Optimization) training loop for World Model."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from buffer import RolloutBuffer
from WorldModel import WorldModel
from utils.gym_tools import state_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("train.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train(
    model: WorldModel,
    envs: Any,
    max_iter: int = 1000,
    device: torch.device = torch.device("cpu"),
    # PPO hyperparameters
    rollout_steps: int = 128,
    num_ppo_epochs: int = 4,
    batch_size: int = 64,
    clip_range: float = 0.2,
    clip_range_vf: float | None = None,  # None = no clipping
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    # Learning rates
    learning_rate: float = 3e-4,
    policy_lr: float | None = None,  # None = use learning_rate
    # Loss coefficients
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    # World model training
    train_world_model: bool = True,
    world_model_epochs: int = 1,
    # Early stopping
    patience: int = 0,  # 0 = disabled; stop after this many epochs without reward improvement
    # Logging & saving
    use_tensorboard: bool = True,
    save_path: Path = Path("./"),
    save_freq: int = 50,
    log_freq: int = 1,
    render_mode: str = "",
) -> dict[str, list]:
    """
    Train world model with PPO algorithm.

    PPO Algorithm:
        1. Collect rollout_steps of experience with current policy
        2. Compute GAE advantages
        3. For num_ppo_epochs:
            - Sample mini-batches
            - Compute clipped surrogate loss
            - Update policy and value function
        4. Optionally train world model components
        5. Repeat

    Args:
        model: WorldModel instance
        envs: Vectorized gymnasium environments
        max_iter: Maximum number of training epochs (each epoch = 1 rollout)
        device: Torch device
        rollout_steps: Number of steps to collect per rollout
        num_ppo_epochs: Number of PPO update epochs per rollout
        batch_size: Mini-batch size for PPO updates
        clip_range: PPO clipping parameter (epsilon)
        clip_range_vf: Value function clipping (None = disabled)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        learning_rate: Learning rate for world model
        policy_lr: Learning rate for policy (defaults to learning_rate)
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        train_world_model: Whether to train vision/memory components
        world_model_epochs: Epochs to train world model per rollout
        use_tensorboard: Enable tensorboard logging
        save_path: Path to save checkpoints
        save_freq: Save model every N epochs
        log_freq: Log metrics every N epochs
        render_mode: Render mode (unused, kept for API compatibility)

    Returns:
        Dictionary of training history
    """
    num_envs = envs.num_envs
    obs_space = envs.single_observation_space
    action_space = envs.single_action_space
    is_image_based = len(obs_space.shape) == 3

    # Determine observation shape for buffer
    if is_image_based:
        obs_shape = (obs_space.shape[2], obs_space.shape[0], obs_space.shape[1])
    else:
        obs_shape = obs_space.shape

    policy_lr = policy_lr or learning_rate

    # Setup optimizers
    world_params = list(model.vision.parameters()) + list(model.memory.parameters())

    world_optimizer = torch.optim.Adam(world_params, lr=learning_rate)
    policy_optimizer = torch.optim.Adam(model.controller.parameters(), lr=policy_lr)

    # Cosine-annealing LR schedulers — smoothly decay to lr/10 over the full run
    world_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        world_optimizer, T_max=max(max_iter, 1), eta_min=learning_rate / 10
    )
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer, T_max=max(max_iter, 1), eta_min=policy_lr / 10
    )

    # Detect discrete vs continuous action space
    is_discrete = hasattr(action_space, "n")
    action_dim = action_space.n if is_discrete else action_space.shape[0]

    # Setup buffer
    buffer = RolloutBuffer(
        buffer_size=rollout_steps,
        num_envs=num_envs,
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=device,
        gamma=gamma,
        gae_lambda=gae_lambda,
        is_discrete=is_discrete,
    )

    # Setup logging
    writer = SummaryWriter() if use_tensorboard else None

    # Training history
    history: dict[str, list] = {
        "epoch": [],
        "reward": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "world_loss": [],
        "vision_loss": [],
        "memory_loss": [],
    }

    # Initialize
    state, _ = envs.reset()
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_episodes = 0
    total_steps = 0

    best_reward = float("-inf")
    best_mean_reward = float("-inf")
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("PPO Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {max_iter}, Rollout steps: {rollout_steps}")
    logger.info(f"PPO epochs: {num_ppo_epochs}, Batch size: {batch_size}")
    logger.info(f"Clip range: {clip_range}, gamma: {gamma}, lambda: {gae_lambda}")
    logger.info(f"Entropy coef: {entropy_coef}, Value coef: {value_coef}")
    logger.info("=" * 60)

    for epoch in range(max_iter):
        # ============ COLLECT ROLLOUT ============
        buffer.reset()
        epoch_completed_rewards: list[float] = []  # episode rewards finishing in this rollout

        for _step in range(rollout_steps):
            state_tensor = state_transform(state, is_image_based, device)

            with torch.no_grad():
                # Get model outputs
                output_dict = model(
                    state_tensor,
                    action_space=action_space,
                    is_image_based=is_image_based,
                    return_losses=True,
                    last_reward=episode_rewards,
                )

                action = output_dict["action"]
                log_prob = output_dict["log_probs"].squeeze(-1)
                value = output_dict["value"].squeeze(-1)

                # Reuse z_t already computed inside WorldModel.forward() — no extra encode pass.
                z_t = output_dict["z_t"]
                h_t = output_dict["memory_hidden"]

            # Step environment
            actions_np = action.cpu().numpy()
            next_state, reward, terminated, truncated, _ = envs.step(actions_np)
            done = terminated | truncated
            done_t = torch.from_numpy(done).to(device, dtype=torch.bool)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)

            # Compute next latent state for memory training
            with torch.no_grad():
                next_state_tensor = state_transform(next_state, is_image_based, device)
                z_e_next = model.vision.encode(next_state_tensor, is_image_based=is_image_based)
                z_next = z_e_next.mean(dim=(2, 3)) if is_image_based else z_e_next

            # Store in buffer
            buffer.add(
                obs=state_tensor,
                action=action,
                reward=reward_t,
                value=value,
                log_prob=log_prob,
                done=done_t,
                z_t=z_t,
                h_t=h_t,
                z_next=z_next,
            )

            # Track episode stats
            episode_rewards += reward_t
            episode_lengths += 1
            total_steps += num_envs
            model.iter_num += num_envs

            # Handle episode ends
            for env_idx in torch.where(done_t)[0]:
                completed_episodes += 1
                model.nb_experiments += 1
                ep_reward = episode_rewards[env_idx].item()
                epoch_completed_rewards.append(ep_reward)

                if writer:
                    writer.add_scalar("rollout/episode_reward", ep_reward, completed_episodes)
                    writer.add_scalar("rollout/episode_length", episode_lengths[env_idx].item(), completed_episodes)

                # Track best single-episode reward
                if ep_reward > best_reward:
                    best_reward = ep_reward

                episode_rewards[env_idx] = 0
                episode_lengths[env_idx] = 0
                model.reset_env_memory(env_idx)

            state = next_state

        # ============ COMPUTE ADVANTAGES ============
        with torch.no_grad():
            state_tensor = state_transform(state, is_image_based, device)
            output_dict = model(
                state_tensor,
                action_space=action_space,
                is_image_based=is_image_based,
                return_losses=True,
                last_reward=episode_rewards,
            )
            last_value = output_dict["value"]
            last_done = torch.from_numpy(terminated | truncated).to(device, dtype=torch.bool)

        buffer.compute_gae(last_value, last_done)

        # Normalize advantages
        adv_mean = buffer.advantages.mean()
        adv_std = buffer.advantages.std() + 1e-8
        buffer.advantages = (buffer.advantages - adv_mean) / adv_std

        # ============ PPO UPDATE ============
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        model.train()  # Ensure model is in training mode
        for _ppo_epoch in range(num_ppo_epochs):
            for batch in buffer.get_batches(batch_size):
                # Get batch data
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"].detach()
                old_values = batch["old_values"].detach()
                advantages = batch["advantages"].detach()
                returns = batch["returns"].detach()
                z_t = batch["latent_states"].detach()
                h_t = batch["hidden_states"].detach()

                # Evaluate actions using the controller's evaluate_actions method
                new_log_probs, new_values, entropy = model.controller.evaluate_actions(z_t, h_t, actions)

                # ============ POLICY LOSS (Clipped Surrogate) ============
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # ============ VALUE LOSS ============
                if clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range_vf, clip_range_vf)
                    value_loss_1 = (new_values - returns) ** 2
                    value_loss_2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # ============ ENTROPY BONUS ============
                entropy_loss = -entropy.mean()

                # ============ TOTAL LOSS ============
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                # ============ UPDATE ============
                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.controller.parameters(), max_grad_norm)
                policy_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Average losses
        avg_policy_loss = total_policy_loss / max(num_updates, 1)
        avg_value_loss = total_value_loss / max(num_updates, 1)
        avg_entropy = total_entropy / max(num_updates, 1)

        # ============ WORLD MODEL UPDATE ============
        # Train vision and memory using collected observations
        avg_world_loss = 0.0
        avg_vision_loss = 0.0
        avg_memory_loss = 0.0
        # Only train if there are trainable vision/memory parameters
        has_trainable_params = len(world_params) > 0 and any(p.requires_grad for p in world_params)
        if train_world_model and has_trainable_params:
            wm_updates = 0
            for _wm_epoch in range(world_model_epochs):
                for batch in buffer.get_batches(batch_size):
                    obs = batch["observations"]
                    z_t = batch["latent_states"]
                    h_t = batch["hidden_states"]
                    z_next_actual = batch["next_latent_states"]
                    actions = batch["actions"]

                    # Convert discrete actions to one-hot for memory
                    if is_discrete:
                        a_t = torch.nn.functional.one_hot(actions.long(), num_classes=action_dim).float()
                    else:
                        a_t = actions

                    # Vision forward pass (3-tuple: recon, z, loss)
                    recon, _z_e, vq_loss = model.vision(obs)

                    # Vision loss (reconstruction for VQ-VAE/VAE, or predictive for JEPA)
                    if recon.shape == obs.shape:
                        vision_loss = torch.nn.functional.mse_loss(recon, obs)
                    else:
                        vision_loss = vq_loss.mean() if vq_loss.numel() > 0 else torch.tensor(0.0, device=device)

                    # Memory loss: predict next latent state
                    z_next_pred = model.memory.predict_next(z_t, a_t, h_t)
                    memory_loss = torch.nn.functional.mse_loss(z_next_pred, z_next_actual)

                    # Extra memory loss (e.g. KL divergence for RSSM)
                    extra_memory_loss = model.memory.get_extra_loss()

                    # CPC (InfoNCE) auxiliary loss (e.g. TemporalTransformer)
                    cpc_loss = model.memory.compute_cpc_loss(h_t, z_next_actual)

                    world_loss = vision_loss + vq_loss.mean() + memory_loss
                    if extra_memory_loss is not None:
                        world_loss = world_loss + extra_memory_loss
                    if cpc_loss is not None:
                        world_loss = world_loss + cpc_loss

                    # Only backprop if loss requires grad
                    if world_loss.requires_grad:
                        world_optimizer.zero_grad()
                        world_loss.backward()
                        nn.utils.clip_grad_norm_(world_params, max_grad_norm)
                        world_optimizer.step()
                        avg_world_loss += world_loss.item()
                        avg_vision_loss += vision_loss.item()
                        avg_memory_loss += memory_loss.item()
                        wm_updates += 1

            avg_world_loss /= max(wm_updates, 1)
            avg_vision_loss /= max(wm_updates, 1)
            avg_memory_loss /= max(wm_updates, 1)

        # ============ MEAN EPOCH REWARD + EARLY STOPPING ============
        mean_epoch_reward = float(np.mean(epoch_completed_rewards)) if epoch_completed_rewards else float("nan")

        if not np.isnan(mean_epoch_reward):
            if mean_epoch_reward > best_mean_reward:
                best_mean_reward = mean_epoch_reward
                patience_counter = 0
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch}: no reward improvement for {patience} consecutive epochs."
                    )
                    # Save a checkpoint before stopping
                    model.save(save_path / "ppo_early_stop.pt", obs_space, action_space)
                    break

        # ============ LR SCHEDULER STEP ============
        world_scheduler.step()
        policy_scheduler.step()

        # ============ LOGGING ============
        if epoch % log_freq == 0:
            logger.info(
                f"Epoch {epoch:4d} | "
                f"Episodes: {completed_episodes:5d} | "
                f"Best: {best_reward:7.1f} | "
                f"MeanRew: {mean_epoch_reward:7.2f} | "
                f"Policy: {avg_policy_loss:.4f} | "
                f"Value: {avg_value_loss:.4f} | "
                f"Entropy: {avg_entropy:.4f}"
            )

        if writer:
            writer.add_scalar("train/policy_loss", avg_policy_loss, epoch)
            writer.add_scalar("train/value_loss", avg_value_loss, epoch)
            writer.add_scalar("train/entropy", avg_entropy, epoch)
            writer.add_scalar("train/world_loss", avg_world_loss, epoch)
            writer.add_scalar("train/vision_loss", avg_vision_loss, epoch)
            writer.add_scalar("train/memory_loss", avg_memory_loss, epoch)
            writer.add_scalar("train/total_steps", total_steps, epoch)
            writer.add_scalar("train/mean_epoch_reward", mean_epoch_reward, epoch)
            writer.add_scalar("train/lr_world", world_scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("train/lr_policy", policy_scheduler.get_last_lr()[0], epoch)

        # Record history
        history["epoch"].append(epoch)
        history["reward"].append(mean_epoch_reward)
        history["policy_loss"].append(avg_policy_loss)
        history["value_loss"].append(avg_value_loss)
        history["entropy"].append(avg_entropy)
        history["world_loss"].append(avg_world_loss)
        history["vision_loss"].append(avg_vision_loss)
        history["memory_loss"].append(avg_memory_loss)

        # ============ SAVE ============
        if (epoch + 1) % save_freq == 0:
            model.save(save_path / f"ppo_epoch_{epoch + 1}.pt", obs_space, action_space)

    # Final save
    model.save(save_path / "ppo_final.pt", obs_space, action_space)

    if writer:
        writer.close()

    logger.info("=" * 60)
    logger.info(f"Training complete! Best reward: {best_reward:.1f}")
    logger.info(f"Total episodes: {completed_episodes}, Total steps: {total_steps}")
    logger.info("=" * 60)

    return history
