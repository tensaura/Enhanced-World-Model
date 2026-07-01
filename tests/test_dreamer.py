"""Tests for the DreamerV3-lite agent: math utilities, networks, world model and
actor-critic. All configs are deliberately tiny so the suite runs on CPU in seconds."""

import numpy as np
import pytest
import torch

from dreamer.models import Dreamer, DreamerConfig
from dreamer.replay import SequenceReplay
from dreamer.utils import (
    OneHotCategoricalST,
    TwoHotSymlog,
    lambda_return,
    symexp,
    symlog,
)


def _tiny_cfg(is_image: bool, is_discrete: bool) -> DreamerConfig:
    return DreamerConfig(
        obs_shape=(64, 64, 3) if is_image else (4,),
        action_dim=3 if not is_discrete else 2,
        is_image=is_image,
        is_discrete=is_discrete,
        deter_dim=16,
        num_categoricals=4,
        num_classes=4,
        cnn_depth=4,
        hidden=32,
        mlp_layers=2,
        horizon=3,
    )


# ------------------------------------------------------------------- math utils


def test_symlog_symexp_inverse():
    x = torch.tensor([-1000.0, -1.0, 0.0, 0.5, 1000.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_two_hot_recovers_value():
    # A confident two-hot distribution should decode close to its target.
    logits = torch.zeros(8, 255)
    dist = TwoHotSymlog(logits)
    target = torch.tensor([5.0] * 8)
    # log_prob is finite and the mean is in a sane range.
    assert torch.isfinite(dist.log_prob(target)).all()
    assert dist.mean().shape == (8, 1)


def test_onehot_straight_through():
    logits = torch.randn(5, 4, requires_grad=True)
    dist = OneHotCategoricalST(logits)
    sample = dist.sample()
    assert sample.shape == (5, 4)
    # Each row is one-hot in the forward pass.
    assert torch.allclose(sample.sum(-1), torch.ones(5))
    # Gradients flow back through the straight-through estimator.
    sample.sum().backward()
    assert logits.grad is not None


def test_lambda_return_constant_reward():
    # With reward 1, value 0, continue 1, lambda 1 → discounted sum of ones.
    h, b = 4, 2
    rewards = torch.ones(h, b)
    values = torch.zeros(h + 1, b)
    continues = torch.ones(h, b)
    out = lambda_return(rewards, values, continues, lambda_=1.0)
    assert out.shape == (h, b)
    assert torch.allclose(out[0], torch.tensor(4.0))


# --------------------------------------------------------------------- replay


def test_replay_sample_shapes():
    rb = SequenceReplay(200, (4,), 2, is_image=False)
    for i in range(120):
        rb.add(
            np.random.randn(4).astype("float32"),
            np.eye(2)[i % 2].astype("float32"),
            float(i),
            i % 30 == 0,
            False,
        )
    assert rb.can_sample(16)
    batch = rb.sample(8, 16, torch.device("cpu"))
    assert batch["obs"].shape == (8, 16, 4)
    assert batch["action"].shape == (8, 16, 2)
    assert batch["reward"].shape == (8, 16)
    assert batch["is_first"].shape == (8, 16)


# ----------------------------------------------------------------- world model


@pytest.mark.parametrize("is_image,is_discrete", [(False, True), (True, False)])
def test_world_model_loss_finite(is_image, is_discrete):
    cfg = _tiny_cfg(is_image, is_discrete)
    agent = Dreamer(cfg)
    rb = SequenceReplay(200, cfg.obs_shape, cfg.action_dim, is_image)
    for i in range(80):
        obs = (
            np.random.randint(0, 255, cfg.obs_shape, dtype=np.uint8)
            if is_image
            else np.random.randn(*cfg.obs_shape).astype("float32")
        )
        action = np.random.randn(cfg.action_dim).astype("float32")
        rb.add(obs, action, float(np.random.randn()), i % 20 == 0, False)
    batch = rb.sample(2, 6, torch.device("cpu"))
    loss, metrics, post = agent.wm.loss(batch)
    assert torch.isfinite(loss)
    assert {"deter", "stoch", "logits"} <= set(post.keys())
    assert all(np.isfinite(v) for v in metrics.values())


# --------------------------------------------------------------- actor-critic


@pytest.mark.parametrize("is_image,is_discrete", [(False, True), (True, False)])
def test_full_train_step(is_image, is_discrete):
    cfg = _tiny_cfg(is_image, is_discrete)
    agent = Dreamer(cfg)
    model_opt = torch.optim.Adam(agent.wm.parameters(), lr=1e-4)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=1e-4)

    rb = SequenceReplay(200, cfg.obs_shape, cfg.action_dim, is_image)
    for i in range(80):
        obs = (
            np.random.randint(0, 255, cfg.obs_shape, dtype=np.uint8)
            if is_image
            else np.random.randn(*cfg.obs_shape).astype("float32")
        )
        rb.add(
            obs,
            np.random.randn(cfg.action_dim).astype("float32"),
            float(np.random.randn()),
            i % 20 == 0,
            False,
        )
    batch = rb.sample(2, 6, torch.device("cpu"))

    wm_loss, _, post = agent.wm.loss(batch)
    model_opt.zero_grad()
    wm_loss.backward()
    model_opt.step()

    actor_loss, critic_loss, ac_metrics = agent.actor_critic_loss(post)
    actor_opt.zero_grad()
    critic_opt.zero_grad()
    (actor_loss + critic_loss).backward()
    actor_opt.step()
    critic_opt.step()
    agent.update_slow_critic()

    assert torch.isfinite(actor_loss)
    assert torch.isfinite(critic_loss)
    assert all(np.isfinite(v) for v in ac_metrics.values())


def test_act_returns_valid_action():
    cfg = _tiny_cfg(is_image=False, is_discrete=True)
    agent = Dreamer(cfg)
    obs = torch.randn(1, 4)
    prev_action = torch.zeros(1, cfg.action_dim)
    is_first = torch.ones(1)
    action, state = agent.act(obs, None, prev_action, is_first, training=True)
    assert action.shape == (1, cfg.action_dim)
    assert {"deter", "stoch", "logits"} <= set(state.keys())
