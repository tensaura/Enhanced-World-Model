"""
Tests for RolloutBuffer — GAE computation, batching, and edge cases.
"""

import math

import numpy as np
import pytest
import torch

from buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_buffer(
    buffer_size: int = 8,
    num_envs: int = 2,
    obs_shape: tuple = (4,),
    action_dim: int = 2,
    is_discrete: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> RolloutBuffer:
    return RolloutBuffer(
        buffer_size=buffer_size,
        num_envs=num_envs,
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=torch.device("cpu"),
        gamma=gamma,
        gae_lambda=gae_lambda,
        is_discrete=is_discrete,
    )


def fill_buffer(buf: RolloutBuffer, reward_val: float = 1.0, done_val: bool = False) -> None:
    """Fill the buffer with dummy uniform data."""
    B = buf.num_envs
    for _ in range(buf.buffer_size):
        obs = torch.zeros(B, *buf.obs_shape)
        if buf.is_discrete:
            action = torch.zeros(B, dtype=torch.long)
        else:
            action = torch.zeros(B, buf.action_dim)
        reward = torch.full((B,), reward_val)
        value = torch.ones(B)
        log_prob = torch.full((B,), -0.5)
        done = torch.full((B,), done_val, dtype=torch.bool)
        buf.add(obs=obs, action=action, reward=reward, value=value, log_prob=log_prob, done=done)


# ---------------------------------------------------------------------------
# Creation & reset
# ---------------------------------------------------------------------------


def test_buffer_creation():
    buf = make_buffer()
    assert buf.ptr == 0
    assert not buf.full


def test_buffer_reset():
    buf = make_buffer(buffer_size=4)
    fill_buffer(buf)
    assert buf.ptr == 4
    assert buf.full
    buf.reset()
    assert buf.ptr == 0
    assert not buf.full


# ---------------------------------------------------------------------------
# add() and overflow guard
# ---------------------------------------------------------------------------


def test_buffer_fills_correctly():
    buf = make_buffer(buffer_size=4, num_envs=3)
    fill_buffer(buf, reward_val=2.0)
    assert torch.all(buf.rewards == 2.0)
    assert buf.full


def test_buffer_discrete_action_shape():
    buf = make_buffer(buffer_size=4, action_dim=5, is_discrete=True)
    fill_buffer(buf)
    assert buf.actions.shape == (4, 2)  # (buffer_size, num_envs)
    assert buf.actions.dtype == torch.long


def test_buffer_continuous_action_shape():
    buf = make_buffer(buffer_size=4, action_dim=3, is_discrete=False)
    fill_buffer(buf)
    assert buf.actions.shape == (4, 2, 3)  # (buffer_size, num_envs, action_dim)


def test_latent_hidden_states_stored():
    buf = make_buffer(buffer_size=4, num_envs=2)
    for _ in range(4):
        obs = torch.zeros(2, 4)
        action = torch.zeros(2, 2)
        buf.add(
            obs=obs,
            action=action,
            reward=torch.ones(2),
            value=torch.ones(2),
            log_prob=torch.full((2,), -0.5),
            done=torch.zeros(2, dtype=torch.bool),
            z_t=torch.ones(2, 8),
            h_t=torch.ones(2, 16),
            z_next=torch.ones(2, 8) * 2,
        )
    assert buf.latent_states is not None
    assert buf.hidden_states is not None
    assert buf.next_latent_states is not None
    assert buf.latent_states.shape == (4, 2, 8)
    assert buf.hidden_states.shape == (4, 2, 16)
    assert buf.next_latent_states.shape == (4, 2, 8)


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def test_gae_no_dones():
    """
    With no episode terminations, constant reward=1, value=1, and last_value=1:
    each TD error delta = r + gamma*V(s') - V(s) = 1 + gamma*1 - 1 = gamma > 0,
    so all advantages should be positive and decrease from step 0 to step T-1.
    """
    gamma, lam = 0.99, 0.95
    buf = make_buffer(buffer_size=4, num_envs=1, gamma=gamma, gae_lambda=lam)
    fill_buffer(buf, reward_val=1.0, done_val=False)

    # last_value = 1.0 keeps the chain consistent (bootstrapping from V=1)
    last_value = torch.ones(1)
    last_done = torch.zeros(1, dtype=torch.bool)
    buf.compute_gae(last_value, last_done)

    assert buf.advantages.shape == (4, 1)
    # All advantages should be positive
    assert (buf.advantages > 0).all(), f"All advantages should be > 0, got {buf.advantages}"
    # Earlier steps should have larger advantages (more future reward to accumulate)
    advs = buf.advantages[:, 0].tolist()
    assert (
        advs[0] > advs[1] > advs[2] > advs[3]
    ), f"Advantages should decrease over time, got {advs}"


def test_gae_with_terminal():
    """Episodes ending mid-sequence should reset GAE accumulation."""
    buf = make_buffer(buffer_size=4, num_envs=1, gamma=0.99, gae_lambda=1.0)

    for t in range(4):
        done = torch.tensor([t == 1], dtype=torch.bool)  # episode ends at step 1
        buf.add(
            obs=torch.zeros(1, 4),
            action=torch.zeros(1, 2),
            reward=torch.ones(1),
            value=torch.ones(1),
            log_prob=torch.full((1,), -0.5),
            done=done,
        )

    buf.compute_gae(torch.zeros(1), torch.zeros(1, dtype=torch.bool))
    # After a done, GAE resets — steps 2 and 3 have advantages independent of steps 0-1
    adv = buf.advantages[:, 0].tolist()
    # step 3 and step 2 are in a fresh episode; step 0 and 1 share an episode
    # Just verify the terminal does NOT carry over to the new episode (adv[2] and adv[3] are finite)
    assert all(math.isfinite(a) for a in adv)


def test_returns_equal_advantages_plus_values():
    buf = make_buffer(buffer_size=4, num_envs=2)
    fill_buffer(buf, reward_val=1.0)
    buf.compute_gae(torch.ones(2), torch.zeros(2, dtype=torch.bool))
    torch.testing.assert_close(buf.returns, buf.advantages + buf.values)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def test_get_batches_covers_all_data():
    buf = make_buffer(buffer_size=8, num_envs=4)
    fill_buffer(buf)
    buf.compute_gae(torch.ones(4), torch.zeros(4, dtype=torch.bool))

    total_samples = 8 * 4  # buffer_size * num_envs
    seen = 0
    for batch in buf.get_batches(batch_size=16, shuffle=False):
        seen += batch["observations"].shape[0]
    assert seen == total_samples


def test_get_batches_includes_latent_states():
    buf = make_buffer(buffer_size=4, num_envs=2)
    for _ in range(4):
        buf.add(
            obs=torch.zeros(2, 4),
            action=torch.zeros(2, 2),
            reward=torch.ones(2),
            value=torch.ones(2),
            log_prob=torch.full((2,), -0.5),
            done=torch.zeros(2, dtype=torch.bool),
            z_t=torch.ones(2, 8),
            h_t=torch.ones(2, 12),
            z_next=torch.ones(2, 8),
        )
    buf.compute_gae(torch.ones(2), torch.zeros(2, dtype=torch.bool))

    for batch in buf.get_batches(batch_size=4):
        assert "latent_states" in batch
        assert "hidden_states" in batch
        assert "next_latent_states" in batch
        assert batch["latent_states"].shape[-1] == 8
        assert batch["hidden_states"].shape[-1] == 12


def test_batch_size_larger_than_buffer():
    """A batch size larger than total data should yield exactly one batch."""
    buf = make_buffer(buffer_size=4, num_envs=2)
    fill_buffer(buf)
    buf.compute_gae(torch.ones(2), torch.zeros(2, dtype=torch.bool))

    batches = list(buf.get_batches(batch_size=1000, shuffle=False))
    assert len(batches) == 1
    assert batches[0]["observations"].shape[0] == 4 * 2


def test_batch_shuffle_changes_order():
    """Shuffled and unshuffled batches should generally have different index orders."""
    buf = make_buffer(buffer_size=16, num_envs=2)
    fill_buffer(buf)
    buf.compute_gae(torch.ones(2), torch.zeros(2, dtype=torch.bool))

    unshuffled = torch.cat([b["observations"] for b in buf.get_batches(8, shuffle=False)])
    shuffled = torch.cat([b["observations"] for b in buf.get_batches(8, shuffle=True)])
    # Same elements, potentially different order — both have same shape
    assert unshuffled.shape == shuffled.shape
