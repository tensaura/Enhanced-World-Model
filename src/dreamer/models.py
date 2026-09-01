"""The DreamerV3-lite agent: world model + actor-critic trained in imagination."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer.networks import (
    ACTOR_REGISTRY,
    CRITIC_REGISTRY,
    DECODER_REGISTRY,
    DYNAMICS_REGISTRY,
    ENCODER_REGISTRY,
    ContinueHead,
    RewardHead,
)
from dreamer.utils import categorical_kl, lambda_return, symexp, symlog


@dataclass
class DreamerConfig:
    """Hyperparameters for the Dreamer agent. Defaults are tuned to train on an
    Apple-Silicon / MPS machine; scale ``deter_dim`` / ``cnn_depth`` up for the long run."""

    # Spaces (filled in from the env).
    obs_shape: tuple[int, ...] = (4,)
    action_dim: int = 2
    is_image: bool = False
    is_discrete: bool = True

    # World model.
    deter_dim: int = 256
    num_categoricals: int = 32
    num_classes: int = 32
    cnn_depth: int = 32
    hidden: int = 256
    mlp_layers: int = 2

    # Pluggable component selection by registry name (empty = auto-pick by modality).
    encoder: str = ""
    decoder: str = ""
    dynamics: str = ""
    actor: str = ""
    critic: str = ""

    # Loss weights / KL.
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    free_bits: float = 1.0

    # Imagination / actor-critic.
    horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 1e-3
    slow_critic_tau: float = 0.02

    # Optimisation.
    model_lr: float = 1e-4
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    eps: float = 1e-8
    grad_clip: float = 100.0

    metadata: dict = field(default_factory=dict)


def _stack_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of per-timestep state dicts into (B, L, ...) tensors."""
    return {k: torch.stack([s[k] for s in states], dim=1) for k in states[0]}


class WorldModel(nn.Module):
    """Encoder + RSSM + decoder + reward/continue heads."""

    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Build pluggable components from the registries. Names come from cfg; an
        # empty selection falls back to the modality-appropriate default. Order
        # matters: the dynamics needs the encoder's embed_dim, the decoder needs
        # the dynamics' feat_dim.
        enc_name = getattr(cfg, "encoder", "") or ("ConvEncoder" if cfg.is_image else "MLPEncoder")
        self.encoder = ENCODER_REGISTRY[enc_name].from_config(cfg)
        embed_dim = self.encoder.embed_dim

        dyn_name = getattr(cfg, "dynamics", "") or "RSSM"
        self.rssm = DYNAMICS_REGISTRY[dyn_name].from_config(cfg, embed_dim)
        feat_dim = self.rssm.feat_dim

        dec_name = getattr(cfg, "decoder", "") or ("ConvDecoder" if cfg.is_image else "MLPDecoder")
        self.decoder = DECODER_REGISTRY[dec_name].from_config(cfg, feat_dim)

        self.reward_head = RewardHead(feat_dim, hidden=cfg.hidden, layers=cfg.mlp_layers)
        self.continue_head = ContinueHead(feat_dim, hidden=cfg.hidden, layers=cfg.mlp_layers)

    # ------------------------------------------------------------ preprocessing

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw observation tensor → network input.

        Images (uint8 NHWC) → float NCHW in ``[-0.5, 0.5]``; vectors → symlog space.
        """
        if self.cfg.is_image:
            obs = obs.float() / 255.0 - 0.5
            if obs.dim() == 4:  # (N, H, W, C) → (N, C, H, W)
                obs = obs.permute(0, 3, 1, 2)
            return obs
        return symlog(obs.float())

    def encode(self, obs_processed: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_processed)

    def decode_obs(self, recon: torch.Tensor) -> torch.Tensor:
        """Network output → displayable observation (inverse of :meth:`preprocess`)."""
        if self.cfg.is_image:
            img = torch.clamp(recon + 0.5, 0.0, 1.0)
            return img.permute(0, 2, 3, 1)  # NCHW → NHWC
        return symexp(recon)

    # ----------------------------------------------------------------- observe

    def observe(
        self, obs: torch.Tensor, actions: torch.Tensor, is_first: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Roll the RSSM over a (B, L) sequence. ``actions[:, t]`` is the action that
        leads into ``obs[:, t]``; ``is_first[:, t]`` flags episode starts."""
        b, length = actions.shape[0], actions.shape[1]
        device = obs.device

        flat = self.preprocess(obs.reshape(b * length, *obs.shape[2:]))
        embed = self.encode(flat).reshape(b, length, -1)

        state = self.rssm.initial(b, device)
        posts: list[dict[str, torch.Tensor]] = []
        priors: list[dict[str, torch.Tensor]] = []
        for t in range(length):
            post, prior = self.rssm.obs_step(state, actions[:, t], embed[:, t], is_first[:, t])
            posts.append(post)
            priors.append(prior)
            state = post
        return _stack_states(posts), _stack_states(priors)

    # -------------------------------------------------------------------- loss

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        recon_hook: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Compute the world-model loss on a replay batch.

        Returns ``(total_loss, metrics, post_states)`` where ``post_states`` are the
        posterior states used as imagination start points. ``recon_hook(recon, target)``
        may return an extra loss term on the decoded frames (e.g. a perceptual loss).
        """
        cfg = self.cfg
        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        is_first = batch["is_first"]
        is_terminal = batch["is_terminal"]

        post, prior = self.observe(obs, actions, is_first)
        feat = self.rssm.get_feat(post)  # (B, L, feat)
        b, length = feat.shape[0], feat.shape[1]
        feat_flat = feat.reshape(b * length, -1)

        # Reconstruction.
        target = self.preprocess(obs.reshape(b * length, *obs.shape[2:]))
        recon = self.decoder(feat_flat)
        recon_loss = 0.5 * F.mse_loss(recon, target, reduction="none")
        recon_loss = recon_loss.reshape(b * length, -1).sum(-1).mean()

        # Reward (two-hot symlog).
        reward_dist = self.reward_head(feat_flat)
        reward_loss = -reward_dist.log_prob(rewards.reshape(b * length)).mean()

        # Continue (Bernoulli).
        cont_target = (1.0 - is_terminal).reshape(b * length)
        cont_dist = self.continue_head(feat_flat)
        cont_loss = -cont_dist.log_prob(cont_target).mean()

        # KL with balancing + free bits.
        post_logits = post["logits"]
        prior_logits = prior["logits"]
        kl_dyn = categorical_kl(post_logits.detach(), prior_logits).sum(-1)  # train prior
        kl_rep = categorical_kl(post_logits, prior_logits.detach()).sum(-1)  # regularise post
        kl_dyn = torch.clamp(kl_dyn, min=cfg.free_bits).mean()
        kl_rep = torch.clamp(kl_rep, min=cfg.free_bits).mean()

        total = cfg.beta_pred * (recon_loss + reward_loss + cont_loss) + cfg.beta_dyn * kl_dyn + cfg.beta_rep * kl_rep

        extra_metrics: dict[str, float] = {}
        if recon_hook is not None:
            extra = recon_hook(recon, target)
            total = total + extra
            extra_metrics["wm/recon_hook"] = extra.item()

        metrics = {
            **extra_metrics,
            "wm/recon": recon_loss.item(),
            "wm/reward": reward_loss.item(),
            "wm/continue": cont_loss.item(),
            "wm/kl_dyn": kl_dyn.item(),
            "wm/kl_rep": kl_rep.item(),
            "wm/total": total.item(),
        }
        return total, metrics, post


class Dreamer(nn.Module):
    """Full agent: world model, actor, critic, and a slow (EMA) target critic."""

    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wm = WorldModel(cfg)
        feat_dim = self.wm.rssm.feat_dim
        actor_name = getattr(cfg, "actor", "") or "Actor"
        self.actor = ACTOR_REGISTRY[actor_name].from_config(cfg, feat_dim)
        critic_name = getattr(cfg, "critic", "") or "Critic"
        self.critic = CRITIC_REGISTRY[critic_name].from_config(cfg, feat_dim)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)

        # EMA range used to normalise returns (DreamerV3 percentile normalisation).
        self.register_buffer("ret_lo", torch.zeros(()))
        self.register_buffer("ret_hi", torch.zeros(()))
        self.ret_lo: torch.Tensor
        self.ret_hi: torch.Tensor

    # ------------------------------------------------------------- interaction

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        prev_action: torch.Tensor,
        is_first: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One environment step: encode obs, update posterior, sample an action."""
        if state is None:
            state = self.wm.rssm.initial(obs.shape[0], obs.device)
        embed = self.wm.encode(self.wm.preprocess(obs))
        post, _ = self.wm.rssm.obs_step(state, prev_action, embed, is_first)
        feat = self.wm.rssm.get_feat(post)
        dist = self.actor(feat)
        action = dist.sample() if training else dist.mode()
        return action, post

    # ------------------------------------------------------------- imagination

    def imagine(self, start: dict[str, torch.Tensor], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll the actor forward in latent space from ``start`` states.

        Returns ``(feats, actions)`` with shapes ``(H + 1, N, feat)`` and ``(H, N, A)``.
        """
        state = {k: v.reshape(-1, v.shape[-1]).detach() for k, v in start.items() if k != "logits"}
        feats = [self.wm.rssm.get_feat(state)]
        actions = []
        for _ in range(horizon):
            dist = self.actor(feats[-1])
            action = dist.sample()
            actions.append(action)
            state = self.wm.rssm.img_step(state, action)
            feats.append(self.wm.rssm.get_feat(state))
        return torch.stack(feats, dim=0), torch.stack(actions, dim=0)

    def _normalise(self, returns: torch.Tensor) -> torch.Tensor:
        """Update and apply the percentile-range return normaliser.

        Percentiles are computed on CPU since ``torch.quantile`` is unavailable on MPS.
        """
        flat = returns.detach().reshape(-1).float().cpu()
        lo = torch.quantile(flat, 0.05).to(self.ret_lo.device)
        hi = torch.quantile(flat, 0.95).to(self.ret_hi.device)
        decay = 0.99
        self.ret_lo = decay * self.ret_lo + (1 - decay) * lo
        self.ret_hi = decay * self.ret_hi + (1 - decay) * hi
        scale = torch.clamp(self.ret_hi - self.ret_lo, min=1.0)
        return scale

    def actor_critic_loss(self, start: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Imagine from ``start`` states and compute actor and critic losses."""
        cfg = self.cfg
        feats, actions = self.imagine(start, cfg.horizon)  # (H+1,N,F), (H,N,A)

        # Predicted rewards/continues at the imagined arrival states s_1..s_H.
        reward = self.wm.reward_head(feats[1:]).mean().squeeze(-1)  # (H, N)
        cont = self.wm.continue_head(feats[1:]).mean  # Bernoulli prob (H, N)
        discount = cfg.gamma * cont

        # Bootstrap values from the slow target critic (detached targets).
        slow_values = self.slow_critic(feats).mean().squeeze(-1)  # (H+1, N)
        returns = lambda_return(
            reward.detach(), slow_values.detach(), discount.detach(), cfg.lambda_
        )  # (H, N) detached targets

        scale = self._normalise(returns)

        # ---- Actor ----
        entropy = self.actor(feats[:-1].detach()).entropy()  # (H, N)
        if cfg.is_discrete:
            logpi = self.actor(feats[:-1].detach()).log_prob(actions)  # (H, N)
            # REINFORCE: standardise advantages (zero-mean/unit-std) so the policy
            # gradient stays O(1) and cannot overwhelm the entropy bonus before the
            # critic catches up — the same trick PPO uses to prevent premature collapse.
            advantage = (returns - slow_values[:-1]).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            actor_loss = -(logpi * advantage).mean()
        else:
            # Pathwise (dynamics) gradient: reward/continue keep the graph through actions,
            # bootstrap values are detached so critic params stay out of the actor update.
            reward_g = self.wm.reward_head(feats[1:]).mean().squeeze(-1)
            cont_g = self.wm.continue_head(feats[1:]).mean
            returns_g = lambda_return(reward_g, slow_values.detach(), (cfg.gamma * cont_g), cfg.lambda_)
            actor_loss = -(returns_g / scale).mean()
        actor_loss = actor_loss - cfg.entropy_scale * entropy.mean()

        # ---- Critic ----
        value_dist = self.critic(feats[:-1].detach())  # predict at s_1..s_H states
        critic_loss = -value_dist.log_prob(returns.detach()).mean()

        metrics = {
            "ac/return": returns.mean().item(),
            "ac/value": slow_values.mean().item(),
            "ac/reward_imag": reward.mean().item(),
            "ac/entropy": entropy.mean().item(),
            "ac/actor_loss": actor_loss.item(),
            "ac/critic_loss": critic_loss.item(),
            "ac/ret_scale": scale.item(),
        }
        return actor_loss, critic_loss, metrics

    @torch.no_grad()
    def update_slow_critic(self) -> None:
        tau = self.cfg.slow_critic_tau
        for slow, fast in zip(self.slow_critic.parameters(), self.critic.parameters(), strict=False):
            slow.data.mul_(1 - tau).add_(fast.data, alpha=tau)

    # -------------------------------------------------------------- save / load

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.cfg}, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> Dreamer:
        ckpt = torch.load(path, weights_only=False, map_location=device)
        model: Dreamer = Dreamer(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        return model
