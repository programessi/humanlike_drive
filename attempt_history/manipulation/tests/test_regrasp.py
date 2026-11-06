"""Meaningful boundaries for learned contact motion and deployable sensing."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from wheel_manipulation.regrasp.env import RegraspEnv, RegraspConfig
from wheel_manipulation.regrasp.policy import TeacherPolicy, TeacherActor, HistoryStudent
from wheel_manipulation.regrasp.scene import SCENE_PATH, build_scene


class RegraspTests(unittest.TestCase):
    def env(self, **kwargs):
        e = RegraspEnv(RegraspConfig(randomization_strength=0, closure_jitter_rad=0,
                                    settle_jitter_seconds=0, episode_seconds=2, **kwargs))
        self.addCleanup(e.close)
        return e

    def test_actor_cannot_read_contacts_torque_or_passive_joints(self):
        e = self.env()
        o, _ = e.reset(seed=7)
        self.assertEqual(o["actor"].shape, (152,))
        self.assertEqual(o["privileged"].shape, (54,))
        first = e._sensor_frame(initialize=True)
        e.contact_forces[:] = 1000
        e.enclosure[:] = .123
        e.data.qpos[e.dependent_q] += .1
        e.data.qfrc_actuator[:] = 999
        second = e._sensor_frame(initialize=True)
        np.testing.assert_array_equal(first, second)

    def test_no_contact_and_zero_command_cannot_steer(self):
        e = self.env(no_contact=True, target_mode="step")
        e.reset(seed=7, options={"direction": 1})
        c = e.controller
        origin = c.wrist_positions.copy()
        for _ in range(25):
            _, _, done, _, _ = e.step(np.zeros(24))
            self.assertFalse(done)
        np.testing.assert_allclose(c.wrist_positions, origin, atol=1e-9)
        self.assertLess(abs(e.wheel_angle), 1e-10)
        self.assertNotIn(e.model.joint("wheel_joint").id, e.model.actuator_trnid[:, 0])
        before = e.data.qpos.copy()
        a = np.zeros(24); a[0] = .5
        c.update(e.data, a, e.robot_dt)
        np.testing.assert_array_equal(e.data.qpos, before)
        self.assertGreater(np.linalg.norm(c.wrist_positions[0] - origin[0]), 0)
        np.testing.assert_allclose(c.wrist_positions[1], origin[1], atol=1e-9)

    def test_time_reference_is_independent_of_actual_progress(self):
        e = self.env(target_mode="step", no_contact=True)
        e.reset(seed=7, options={"direction": -1})
        commands = e.command_angles.copy()
        velocities = e.command_velocities.copy()
        for _ in range(25):
            e.step(np.zeros(24))
        self.assertLess(e.target_angle, commands[0])
        self.assertAlmostEqual(e.target_angle, commands[25])
        self.assertLessEqual(np.max(abs(velocities)), np.deg2rad(5) + 1e-8)
        self.assertLessEqual(np.max(abs(np.diff(velocities) / e.dt)), np.deg2rad(40) + 1e-8)

    def test_command_integrator_state_is_observable(self):
        e = self.env()
        e.reset(seed=7)
        a = e._sensor_frame(initialize=True)
        e.controller.wrist_positions[0, 0] += .005
        b = e._sensor_frame(initialize=True)
        np.testing.assert_array_equal(a[:140], b[:140])
        self.assertGreater(np.linalg.norm(a[140:] - b[140:]), .1)

    def test_static_contact_and_open_close_do_not_count_as_regrasp(self):
        e = self.env()
        e.reset(seed=7)
        e.initial_contacts[:] = True
        initial = e.material_anchor.copy()
        e._material_phase = lambda: initial + np.deg2rad(20)
        e.contact_forces[:] = 1
        for _ in range(20): e._events()
        self.assertEqual(e.regrasp_count.sum(), 0)
        # Losing both hands must not be called a successful support transfer.
        e.contact_forces[:] = 0
        for _ in range(6): e._events()
        e.contact_forces[:] = 1
        for _ in range(10): e._events()
        self.assertEqual(e.regrasp_count.sum(), 0)

    def test_student_interface_and_torchscript_reset_parity(self):
        torch.set_num_threads(1)
        e = self.env()
        policy = TeacherPolicy(e.observation_space, e.action_space, lambda _: 3e-4)
        teacher = TeacherActor(policy).eval()
        student = HistoryStudent(teacher).eval()
        x, h, starts = torch.randn(2, 152), torch.randn(2, 128), torch.ones(2)
        a, next_h = student(x, h, starts)
        b, zero_h = student(x, torch.zeros_like(h), starts)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(next_h, zero_h)
        self.assertEqual(a.shape, (2, 24))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "actor.pt"
            torch.jit.script(student).save(str(path))
            loaded = torch.jit.load(str(path))
            actual, actual_h = loaded(x, h, starts)
            torch.testing.assert_close(actual, a)
            torch.testing.assert_close(actual_h, next_h)

    def test_interfinger_collisions_restored(self):
        if not SCENE_PATH.exists(): build_scene()
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        cross_finger = model.body("L_thumb_distal").id, model.body("L_index_intermediate").id
        signature = min(cross_finger) * (1 << 16) + max(cross_finger)
        self.assertNotIn(signature, model.exclude_signature.tolist())

    def test_recovery_reward_cannot_be_farmed_by_closing_static_hand(self):
        e = self.env(curriculum="B")
        e.reset(seed=7, options={"exact_curriculum": True, "free_side": 0})
        e.initial_contacts[0] = False
        e.contact_forces[:] = 10
        e.enclosure[:] = 1
        e.target_angle = e.wheel_angle
        e.target_velocity = 0
        e._wheel_contact_torques = lambda: np.array([.03, .03, 0])
        for _ in range(20): e._events()
        self.assertEqual(e.recovery_count.sum(), 0)
        e.target_velocity = e.data.qvel[e.wheel_v] = .05
        for _ in range(20): e._events()
        self.assertEqual(e.recovery_count.tolist(), [1, 0])
        for _ in range(20): e._events()
        self.assertEqual(e.new_recovery, 0)

    def test_teacher_export_matches_frozen_policy(self):
        e = self.env()
        obs, _ = e.reset(seed=7)
        policy = TeacherPolicy(e.observation_space, e.action_space, lambda _: 3e-4)
        actor = TeacherActor(policy).eval()
        expected, _ = policy.predict(obs, deterministic=True)
        with torch.no_grad():
            actual = actor(torch.from_numpy(obs["actor"])[None], torch.from_numpy(obs["privileged"])[None])
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-6)

    def test_excessive_policy_kl_reduces_learning_rate_with_bounds(self):
        from wheel_manipulation.regrasp.train import EvidenceCallback
        callback = EvidenceCallback(Path("unused"), kl_target=.01, max_lr=3e-4)
        model = SimpleNamespace(logger=SimpleNamespace(name_to_value={"train/approx_kl": .1}),
                                lr_schedule=lambda _: 3e-4)
        callback.model = model
        callback._on_rollout_start()
        self.assertAlmostEqual(model.lr_schedule(1), 2e-4)
        for _ in range(50): callback._on_rollout_start()
        self.assertAlmostEqual(model.lr_schedule(1), 1e-5)
        model.logger.name_to_value["train/approx_kl"] = .001
        for _ in range(50): callback._on_rollout_start()
        self.assertAlmostEqual(model.lr_schedule(1), 3e-4)

    def test_advanced_preparation_hold_and_previous_course_envelopes(self):
        e = self.env(curriculum="C", amplitude_deg=80, target_mode="hold", no_contact=True,
                     prior_task_fraction=1.)
        e.reset(seed=7, options={"exact_curriculum": True, "phase_deg": 40})
        np.testing.assert_allclose(e.command_angles, np.deg2rad(40))
        e.config.target_mode = "step"
        e.config.episode_seconds = 120
        e.reset(seed=7)
        self.assertIn(e.active_curriculum, ("A", "B"))
        self.assertLessEqual(np.max(abs(e.command_angles)), np.deg2rad(10) + 1e-8)

    def test_long_course_keeps_commanding_after_first_cycle(self):
        e = self.env(curriculum="D", amplitude_deg=80, target_mode="sequence", no_contact=True,
                     prior_task_fraction=0.)
        e.config.episode_seconds = 300
        e.reset(seed=7, options={"exact_curriculum": True})
        late = e.command_angles[int(150 / e.dt):]
        self.assertGreater(np.max(late), np.deg2rad(70))
        self.assertLess(np.min(late), -np.deg2rad(70))


if __name__ == "__main__":
    unittest.main()
