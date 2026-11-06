"""Causality, recurrent training boundaries and deployment interface checks."""
import tempfile
from pathlib import Path
import unittest

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np
import torch

from wheel_manipulation.exploration.env import CentralizedWheelEnv, ExplorationConfig
from wheel_manipulation.exploration.policy import ActorCritic, export_actor, generalized_advantages


class CentralizedExplorationTests(unittest.TestCase):
    def make_env(self, **kwargs):
        config = ExplorationConfig(episode_seconds=2, randomization_strength=0,
                                   closure_jitter_rad=0, settle_jitter_seconds=0, **kwargs)
        env = CentralizedWheelEnv(config)
        self.addCleanup(env.close)
        return env

    def test_gym_contract_and_command_previews(self):
        env = self.make_env(target_mode="step")
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=7, options={"target_deg": 3})
        self.assertEqual(obs["actor"].shape, (140,))
        self.assertEqual(obs["privileged"].shape, (54,))
        self.assertEqual(env.action_space.shape, (24,))
        for _ in range(20):
            obs, _, _, _, _ = env.step(np.zeros(24))
        expected = (env._preview() - obs["actor"][120]) / np.deg2rad(10)
        np.testing.assert_allclose(obs["actor"][122:128], expected, atol=1e-6)
        self.assertLessEqual(np.max(np.abs(env.command_velocities)), np.deg2rad(5) + 1e-9)
        self.assertLessEqual(np.max(np.abs(np.diff(env.command_velocities) / env.dt)), np.deg2rad(40) + 1e-9)

    def test_no_angle_servo_and_no_controller_physics_writes(self):
        env = self.make_env(target_mode="step", no_contact=True)
        env.reset(seed=7, options={"target_deg": 3})
        c = env.controller
        initial = c.wrist_positions.copy()
        for _ in range(40):
            _, _, terminated, _, _ = env.step(np.zeros(24))
            self.assertFalse(terminated)
        np.testing.assert_allclose(c.wrist_positions, initial, atol=1e-10)
        self.assertLess(abs(env.wheel_angle), 1e-10)
        action = np.zeros(24)
        action[2] = .5
        before = env.data.qpos.copy()
        c.update(env.data, action, env.robot_dt)
        np.testing.assert_array_equal(env.data.qpos, before)
        self.assertGreater(np.linalg.norm(c.wrist_positions[0] - initial[0]), 0)
        np.testing.assert_allclose(c.wrist_positions[1], initial[1], atol=1e-10)
        self.assertNotIn(env.model.joint("wheel_joint").id, env.model.actuator_trnid[:, 0])

    def test_full_hand_range_and_no_critic_or_mimic_leak(self):
        env = self.make_env(target_mode="hold")
        obs, _ = env.reset(seed=7)
        c = env.controller
        np.testing.assert_allclose(c.hand_targets(-np.ones(12)), c.hand_limits[:, 0] + c.config.joint_margin)
        np.testing.assert_allclose(c.hand_targets(np.ones(12)), c.hand_limits[:, 1] - c.config.joint_margin)
        before = env._sensor_frame(initialize=True)
        env.contact_forces[:] = 1000
        env.enclosure[:] = .123
        env.data.qpos[env.dependent_q] += .1
        after = env._sensor_frame(initialize=True)
        np.testing.assert_array_equal(before, after)

    def test_gru_reset_and_export_parity(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        network = ActorCritic()
        with torch.no_grad():
            network.actor.mean.weight.normal_(0, .1)
        x = torch.randn(2, 140)
        h = torch.randn(2, 128)
        reset = torch.ones(2)
        with torch.no_grad():
            mean, new_h = network.actor(x, h, reset)
            zero_mean, zero_h = network.actor(x, torch.zeros_like(h), reset)
            _, remembered_h = network.actor(x, h, torch.zeros(2))
        torch.testing.assert_close(mean, zero_mean)
        torch.testing.assert_close(new_h, zero_h)
        self.assertGreater(float(torch.linalg.norm(remembered_h - new_h)), .1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            export_actor(network, path)
            actor = torch.jit.load(str(path))
            action, actual_h = actor(x, h, reset)
            torch.testing.assert_close(action, torch.tanh(mean))
            torch.testing.assert_close(actual_h, new_h)

    def test_timeouts_bootstrap_without_cross_episode_gae(self):
        # Step 0 is a timeout: bootstrap its terminal observation value 10,
        # while the next episode's reward 100 must not contaminate its GAE.
        advantages, returns = generalized_advantages(torch.tensor([1., 100.]),
            torch.tensor([2., 0.]), torch.tensor([10., 0.]), torch.tensor([1., 1.]), gamma=.9, lam=.95)
        torch.testing.assert_close(advantages, torch.tensor([8., 100.]))
        torch.testing.assert_close(returns, torch.tensor([10., 100.]))

    def test_physical_preparation_varies_reproducibly(self):
        config = ExplorationConfig(episode_seconds=1, closure_jitter_rad=.06, settle_jitter_seconds=.1)
        env = CentralizedWheelEnv(config)
        self.addCleanup(env.close)
        _, first = env.reset(seed=7)
        _, repeat = env.reset(seed=7)
        _, other = env.reset(seed=17)
        np.testing.assert_array_equal(first["preparation_closed"], repeat["preparation_closed"])
        self.assertFalse(np.allclose(first["preparation_closed"], other["preparation_closed"]))
        self.assertTrue(np.all(np.asarray(first["preparation_contact_forces"]) > 0))


if __name__ == "__main__":
    unittest.main()
