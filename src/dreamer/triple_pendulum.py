"""Inverted **triple** pendulum on a cart — a custom MuJoCo Gymnasium environment.

Extends Gymnasium's ``InvertedDoublePendulum`` to a three-link pole. The agent applies
a horizontal force to the cart and must balance the 3-link chain upright. Fully vertical
the tip reaches z = 1.8 m; the episode ends if the tip drops below ``term_height``.

Registered as ``InvertedTriplePendulum-v0`` on import. Observation is a 12-vector:
``[cart_x, sin(θ1..3), cos(θ1..3), qvel(4), cart_constraint_force]``; action is the
1-D cart force in ``[-1, 1]``.
"""

from __future__ import annotations

import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register, registry
from gymnasium.spaces import Box

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "inverted_triple_pendulum.xml")

_FULL_HEIGHT = 1.8  # tip height when perfectly upright (3 links x 0.6 m)


class InvertedTriplePendulumEnv(MujocoEnv, utils.EzPickle):
    """Balance a three-link inverted pendulum on a cart."""

    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(
        self,
        xml_file: str = _ASSET,
        frame_skip: int = 5,
        default_camera_config: dict | None = None,
        healthy_reward: float = 10.0,
        upright_weight: float = 3.0,
        term_height: float = 1.0,
        reset_noise_scale: float = 0.1,
        **kwargs,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            healthy_reward,
            upright_weight,
            term_height,
            reset_noise_scale,
            **kwargs,
        )
        self._healthy_reward = healthy_reward
        self._upright_weight = upright_weight
        self._term_height = term_height
        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config or {},
            **kwargs,
        )
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        x, _, z = self.data.site_xpos[0]  # tip site: x and height z
        obs = self._get_obs()
        terminated = bool(z <= self._term_height)
        reward, info = self._get_rew(x, z, terminated)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def _get_rew(self, x, z, terminated):
        # Reward EVERY link being vertical (not just the tip): a far cleaner
        # "stand all three up" gradient than tip-height alone. The absolute angle
        # of link k from vertical is the cumulative sum of the relative hinge angles,
        # so sum(cos(.)) is 3 when the whole chain is perfectly upright.
        abs_angles = np.cumsum(self.data.qpos[1:4])
        upright = float(np.sum(np.cos(abs_angles)))  # in [-3, 3]
        omega = self.data.qvel[1:4]  # the three hinge angular velocities

        alive_bonus = self._healthy_reward * int(not terminated)
        upright_reward = self._upright_weight * upright
        ctrl_penalty = 0.01 * x**2
        vel_penalty = 1e-3 * float(np.sum(omega**2))

        reward = alive_bonus + upright_reward - ctrl_penalty - vel_penalty
        return reward, {
            "reward_survive": alive_bonus,
            "reward_upright": upright_reward,
            "ctrl_penalty": -ctrl_penalty,
            "velocity_penalty": -vel_penalty,
            "tip_height": z,
        }

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart position
                np.sin(self.data.qpos[1:]),  # three link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),  # cart + three hinge velocities
                np.clip(self.data.qfrc_constraint, -10, 10)[:1],  # cart constraint force
            ]
        ).ravel()

    def reset_model(self):
        n = self._reset_noise_scale
        self.set_state(
            self.init_qpos + self.np_random.uniform(-n, n, self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * n,
        )
        return self._get_obs()


if "InvertedTriplePendulum-v0" not in registry:
    register(
        id="InvertedTriplePendulum-v0",
        entry_point="dreamer.triple_pendulum:InvertedTriplePendulumEnv",
        max_episode_steps=1000,
        reward_threshold=8000.0,
    )
