"""Verify the narrowed learning boundary, physics causality and export contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np
from stable_baselines3 import PPO
import torch

from wheel_manipulation.common.policy import AsymmetricPolicy, export_actor
from wheel_manipulation.artifacts import require_matching_checkpoint
from wheel_manipulation.env import ContactResidualEnv, ContactTaskConfig, HandsOnlyContactEnv
from wheel_manipulation.observations import deployment_observation
from wheel_manipulation.common.policy import load_policy
from wheel_manipulation.common.kinematics import kinematic_sweep
from wheel_manipulation.common.resources import training_slot
from wheel_manipulation.common.acceptance import aggregate
from wheel_manipulation.paths import ROOT


class ContactResidualTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ContactResidualEnv(ContactTaskConfig(episode_seconds=2, **kwargs))
        self.addCleanup(env.close)
        return env

    def test_gym_contract_and_goal_reference_fields(self):
        env = self.make_env()
        check_env(env, skip_render_check=True)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation["actor"].shape, (8, 122))
        env.set_target(np.deg2rad(10))
        observation, _, _, _, _ = env.step(np.zeros(18))
        self.assertAlmostEqual(float(observation["actor"][-1, 116]), env.target_angle)
        self.assertAlmostEqual(float(observation["actor"][-1, 120]), env.controller.nominal_phase)
        self.assertAlmostEqual(float(observation["actor"][-1, 121]), env.controller.nominal_velocity)
        for invalid in (np.zeros(12), np.full(18, np.nan)):
            with self.assertRaises(ValueError):
                env.step(invalid)

    def test_zero_policy_turns_arms_but_cannot_turn_disconnected_wheel(self):
        env = self.make_env(no_contact=True)
        env.reset(seed=7, options={"target_deg": 10})
        for _ in range(40):
            _, _, terminated, _, _ = env.step(np.zeros(18))
            self.assertFalse(terminated)
        self.assertGreater(env.controller.nominal_phase, np.deg2rad(5))
        self.assertLess(abs(env.wheel_angle), 1e-9)
        self.assertNotIn(env.model.joint("wheel_joint").id, env.model.actuator_trnid[:, 0])
        self.assertIsNone(env.controller.table)

    def test_constant_offsets_do_not_accumulate_and_do_not_write_physics(self):
        env = self.make_env(no_contact=True, settle_seconds=0)
        data = mujoco.MjData(env.model)
        c = env.controller
        data.qpos[c.arm_q] = c.initial_arm
        data.qpos[c.hand_q] = c.closed_hand
        mujoco.mj_forward(env.model, data)
        c.reset_reference(data)
        action = np.zeros(18)
        action[12:] = 1
        for _ in range(400):
            before = data.qpos.copy()
            c.update(data, action, env.robot_dt)
            np.testing.assert_array_equal(data.qpos, before)
            # Explicit scratch kinematics, not a dynamic rollout or wheel write.
            data.qpos[c.motor_q] = c.target
            mujoco.mj_forward(env.model, data)
        np.testing.assert_allclose(np.rad2deg(c.phase_residual), [1, .5], atol=1e-6)
        self.assertAlmostEqual(c.nominal_phase, 0)
        np.testing.assert_allclose(c.radial, .002, atol=1e-8)
        np.testing.assert_allclose(c.axial, .002, atol=1e-8)
        self.assertLessEqual(abs(np.rad2deg(c.phase[0] - c.phase[1])), 1 + 1e-8)
        phase = c.phase.copy()
        for _ in range(100):
            c.update(data, action, env.robot_dt)
            data.qpos[c.motor_q] = c.target
            mujoco.mj_forward(env.model, data)
        np.testing.assert_allclose(c.phase, phase, atol=1e-8)

    def test_actor_isolation_export_and_separately_trainable_hand_ablation(self):
        torch.set_num_threads(1)
        for cls, dimension in ((ContactResidualEnv, 18), (HandsOnlyContactEnv, 12)):
            env = cls(ContactTaskConfig(episode_seconds=2))
            self.addCleanup(env.close)
            model = PPO(AsymmetricPolicy, env, n_steps=16, batch_size=16, seed=7,
                        policy_kwargs=dict(net_arch=dict(pi=[32], vf=[64])))
            model.learn(32)
            obs, _ = env.reset(seed=7)
            action = model.predict(obs, deterministic=True)[0]
            self.assertEqual(action.shape, (dimension,))
            changed = deployment_observation(obs["actor"])
            changed["privileged"][:] = 1000
            np.testing.assert_array_equal(action, model.predict(changed, deterministic=True)[0])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "actor.pt"
                export_actor(model, path)
                actor = torch.jit.load(str(path))
                output = actor(torch.as_tensor(obs["actor"][None])).detach().numpy()[0]
                np.testing.assert_allclose(action, output, atol=1e-6)

    def test_velocity_checkpoint_is_rejected_even_with_same_action_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"interface_version": "dexsteer-v1", "scene_sha256": "same"}))
            with self.assertRaisesRegex(ValueError, "different semantics"):
                require_matching_checkpoint(root / "model.zip", "same")

    def test_implicit_motor_law_preserves_total_pd_force_limits(self):
        env = self.make_env()
        env.reset(seed=7)
        c, data, model = env.controller, env.data, env.model
        data.qvel[c.motor_v] = np.linspace(-10, 10, 26)
        mujoco.mj_forward(model, data)
        desired = c.target + .3
        raw = c.kp * (desired - data.qpos[c.motor_q]) - c.kd * data.qvel[c.motor_v] + data.qfrc_bias[c.motor_v]
        expected = np.clip(raw, model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
        data.ctrl[:] = c.torque(data, desired)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, expected, atol=1e-8)
        self.assertTrue(np.all(model.actuator_forcelimited))
        self.assertTrue(np.all(~model.actuator_ctrllimited.astype(bool)))
        np.testing.assert_array_equal(data.qfrc_applied, 0)

    def test_kinematic_continuity_uses_current_scene(self):
        result = kinematic_sweep()
        self.assertTrue(result["passed"], result)

    def test_training_lock_rejects_overlap_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                with training_slot(workspace):
                    with self.assertRaisesRegex(RuntimeError, "Another manipulation trainer"):
                        with training_slot(workspace):
                            self.fail("Two concurrent training locks were granted")
                    raise RuntimeError("intentional failure")
            with training_slot(workspace):
                pass

    def test_incomplete_suites_cannot_pass_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = []
            for seed in (7, 17, 27):
                path = Path(directory) / f"s{seed}.json"
                path.write_text(json.dumps({"training_seed": seed, "cases": [],
                                            "single_checkpoint_gate_passed": True}))
                reports.append(path)
            self.assertFalse(aggregate(reports)["accepted_in_simulation"])

    def test_preserved_checkpoint_loads_without_deleted_module(self):
        # Local checkpoints are intentionally not versioned. Exercise the actual
        # old SB3 class reference and compare with its preserved deployment graph.
        checkpoint = ROOT / "runs/implicit_reach10_s7_pilot/best/best_model.zip"
        if not checkpoint.exists():
            self.skipTest("Local pilot checkpoint is not installed")
        env = self.make_env()
        require_matching_checkpoint(checkpoint, hashlib.sha256(env.scene_path.read_bytes()).hexdigest())
        model = load_policy(checkpoint, device="cpu")
        obs, _ = env.reset(seed=7)
        actor = torch.jit.load(str(checkpoint.with_name("actor.pt")))
        actual = actor(torch.as_tensor(obs["actor"][None])).detach().numpy()[0]
        np.testing.assert_allclose(model.predict(obs, deterministic=True)[0], actual, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
