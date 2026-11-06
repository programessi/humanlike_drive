"""M1 behavioral contracts, including causal controls and physical regressions."""
import copy
import unittest

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from unified_steering.audit import ROOT
from unified_steering.env import SteeringEnv
from unified_steering.reference import Reference


class ReferenceTests(unittest.TestCase):
    def test_timing_limits_preview_and_accumulated_angle(self):
        reference = Reference(np.deg2rad([0, 400, -400, 0]))
        times = np.linspace(0, reference.duration, 20001)
        values = np.array([reference.sample(t) for t in times])
        self.assertLessEqual(np.max(np.abs(values[:, 1])), np.deg2rad(10)+1e-8)
        self.assertLessEqual(np.max(np.abs(np.diff(values[:, 1])/np.diff(times))), np.deg2rad(30)+1e-5)
        for t in (0, 2.3, 40, reference.duration):
            np.testing.assert_allclose(reference.preview(t), [reference.sample(t+i/10)[0] for i in range(11)], atol=1e-12)
        self.assertEqual(reference.sample(reference.duration+5), (0., 0.))
        with self.assertRaises(ValueError):
            Reference(np.deg2rad([0, 450]))


@unittest.skipUnless((ROOT/'assets/scene.xml').exists() and
                     (ROOT/'assets/initial_states/m1_both_v1/manifest.json').exists(), 'Requires prepared M1 assets')
class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SteeringEnv()

    def setUp(self):
        self.e = self.env
        self.e.reset(seed=7, options={"support": "left", "state_index": 0})

    def test_gymnasium_and_shapes(self):
        check_env(self.e, skip_render_check=True)
        obs, _ = self.e.reset(seed=7)
        self.assertEqual(obs['actor'].shape, (152,))
        self.assertEqual(obs['critic'].shape, (172,))
        self.assertEqual(self.e.action_space.shape, (26,))
        obs, reward, term, trunc, info = self.e.step(self.e.hold_action())
        self.assertTrue(self.e.observation_space.contains(obs))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(term or trunc)
        self.assertEqual(info['physics_steps'], 25)
        self.assertEqual(info['elapsed_seconds'], .05)

    def test_all_validated_states_hold_through_action_interface(self):
        for mode, count in (("left", 5), ("right", 5), ("both", 1)):
            for index in range(count):
                _, info = self.e.reset(options={"support": mode, "state_index": index})
                theta = info['initial_angle_rad']
                self.e.set_reference([theta])
                peak = 0.
                for _ in range(60):
                    _, _, terminated, truncated, step = self.e.step(self.e.hold_action())
                    self.assertFalse(terminated or truncated, step['termination_reason'])
                    peak = max(peak, abs(step['error_rad']))
                self.assertLess(np.rad2deg(peak), 2., (mode, index, np.rad2deg(peak)))

    def test_actor_has_no_privileged_state_dependency(self):
        s = self.e.sim
        actor = self.e.encoder.actor(self.e.reference, 0)
        old_friction = s.model.geom_friction.copy()
        old_q = s.data.qpos.copy()
        try:
            s.model.geom_friction[:] *= 2
            passive = np.setdiff1d(np.arange(s.model.nq), np.r_[s.qidx, s.wheel_q])
            s.data.qpos[passive] += .1
            s.data.qfrc_applied[:] = 50
            mujoco.mj_forward(s.model, s.data)
            np.testing.assert_array_equal(actor, self.e.encoder.actor(self.e.reference, 0))
        finally:
            s.model.geom_friction[:] = old_friction
            s.data.qpos[:] = old_q
            s.data.qfrc_applied[:] = 0

    def test_no_angle_wrap_and_preview(self):
        s = self.e.sim
        s.data.qpos[s.wheel_q] = 0
        zero = self.e.encoder.actor(self.e.reference, 0)
        s.data.qpos[s.wheel_q] = 2*np.pi
        full = self.e.encoder.actor(self.e.reference, 0)
        self.assertAlmostEqual(float(full[134]-zero[134]), 2.)
        np.testing.assert_allclose(full[138:149]*np.pi + 2*np.pi, self.e.reference.preview(0), atol=1e-6)
        np.testing.assert_allclose(zero[136:138], full[136:138], atol=1e-6)

    def test_no_hidden_reference_tracking(self):
        histories = []
        for goal in (0, 100):
            self.e.reset(seed=1, options={"support": "both"})
            initial = float(self.e.sim.data.qpos[self.e.sim.wheel_q])
            self.e.set_reference([initial, np.deg2rad(goal)])
            action = self.e.hold_action()
            history = []
            for _ in range(40):
                _, _, term, trunc, _ = self.e.step(action)
                self.assertFalse(term or trunc)
                history.append(self.e.sim.data.qpos.copy())
            histories.append(history)
        np.testing.assert_array_equal(*histories)

    def test_no_contact_robot_cannot_drive_wheel(self):
        s = self.e.sim
        m = s.model
        wheel = s.clearance.wheel
        contype, affinity = m.geom_contype.copy(), m.geom_conaffinity.copy()
        trajectories = []
        try:
            m.geom_contype[wheel] = 0
            m.geom_conaffinity[wheel] = 0
            for moving in (False, True):
                self.e.reset(options={"support": "left", "state_index": 0})
                s.data.qpos[s.wheel_q] = np.pi/2
                s.data.qvel[s.wheel_v] = 0
                mujoco.mj_forward(m, s.data)
                action = self.e.hold_action()
                if moving:
                    action[7:14] = .1  # Parked free arm moves; support grip may release.
                    action[14:] = -1
                trajectory = []
                for _ in range(20):
                    _, _, term, trunc, info = self.e.step(action)
                    self.assertFalse(term or trunc, info['termination_reason'])
                    self.assertEqual(np.linalg.norm(s.contacts()[0]), 0.)
                    trajectory.append(float(s.data.qpos[s.wheel_q]))
                trajectories.append(trajectory)
            np.testing.assert_allclose(*trajectories, atol=1e-6, rtol=0)
            self.assertLess(trajectories[0][-1], np.pi/2-.1)
        finally:
            m.geom_contype[:] = contype
            m.geom_conaffinity[:] = affinity

    def test_action_order_limits_rate_and_free_arm_control(self):
        e = self.e.executor
        q = self.e.sim.data.qpos[self.e.sim.qidx[:14]].copy()
        target = e.target.copy()
        for i in range(26):
            for sign in (-1, 1):
                e.reset(target)
                action = e.hold_action().astype(float)
                action[i] += sign*.05
                e.begin(action)
                old = e.target.copy()
                self.assertTrue(e.tick(q, self.e.dt))
                change = e.target-old
                self.assertGreater(sign*change[i], 0, i)
                # The public action interface is float32; hand roundtrip adds <2e-8 rad.
                np.testing.assert_allclose(np.delete(change, i), 0., atol=3e-8)
                self.assertLessEqual(abs(change[i]), (e.arm_speed if i < 14 else e.hand_speed[i-14])*self.e.dt+1e-12)
                self.assertTrue(np.all(np.abs(e.target[:14]-q) <= e.arm_lead+1e-12))
        self.assertAlmostEqual(e.limits[15, 1], .5)
        self.assertAlmostEqual(e.limits[21, 1], .5)
        e.reset(target)
        action = e.hold_action()
        action[7] = .2
        self.e.step(action)
        self.assertGreater(e.target[7], target[7])  # No runtime park lock.

    def test_terminal_and_timeout_semantics(self):
        original = self.e.config
        self.e.config = copy.deepcopy(original)
        try:
            self.e.config['episode_seconds'] = .05
            self.e.reset(options={"support": "both"})
            _, _, term, trunc, info = self.e.step(self.e.hold_action())
            self.assertEqual((term, trunc, info['bootstrap_allowed']), (False, True, True))
            with self.assertRaises(RuntimeError):
                self.e.step(self.e.hold_action())
            self.e.reset(options={"support": "both"})
            s = self.e.sim
            s.data.qpos[s.wheel_q] = np.deg2rad(441)
            _, _, term, trunc, info = self.e.step(self.e.hold_action())
            self.assertEqual((term, trunc, info['bootstrap_allowed']), (True, False, False))
            self.assertEqual(info['termination_reason'], 'wheel_envelope')
        finally:
            self.e.config = original

    def test_nonfinite_physics_ends_without_state_repair(self):
        s = self.e.sim
        finite_obs = {key: value.copy() for key, value in self.e.last_obs.items()}
        action = self.e.hold_action()
        s.data.qvel[s.wheel_v] = np.nan  # Fault injection, never used in normal rollouts.
        obs, reward, term, trunc, info = self.e.step(action)
        self.assertTrue(term)
        self.assertFalse(trunc or info['bootstrap_allowed'] or info['observation_valid'])
        self.assertEqual(info['termination_reason'], 'nonfinite_physics')
        self.assertEqual(info['physics_steps'], 0)
        self.assertTrue(np.isnan(s.data.qvel[s.wheel_v]))
        self.assertTrue(np.isfinite(reward))
        for key in obs:
            np.testing.assert_array_equal(obs[key], finite_obs[key])


if __name__ == '__main__':
    unittest.main()
