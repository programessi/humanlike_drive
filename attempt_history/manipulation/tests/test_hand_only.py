"""Physical boundaries of the arm-free task, not reward-implementation tests."""
import unittest

import mujoco
import numpy as np

from wheel_manipulation.hand_only.env import HandOnlyEnv, HandOnlyConfig


class HandOnlyTests(unittest.TestCase):
    def env(self, **kwargs):
        env = HandOnlyEnv(HandOnlyConfig(settle_seconds=.8, **kwargs))
        self.addCleanup(env.close)
        env.reset(seed=7)
        return env

    def test_scene_has_only_two_free_hands_passive_wheel_and_twelve_drives(self):
        env = self.env()
        names = [env.model.joint(i).name for i in range(env.model.njnt)]
        self.assertFalse(any('shoulder' in n or 'elbow' in n or 'wrist_pitch_joint' in n for n in names))
        self.assertEqual(env.model.nu, 12)
        self.assertEqual(env.model.neq, 12)
        self.assertEqual(sum(env.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 2)
        self.assertNotIn(env.model.joint('wheel_joint').id, env.model.actuator_trnid[:, 0])
        self.assertTrue((env.contact_forces > .5).all())

    def test_wrist_commands_do_not_write_physical_state_and_do_not_move_passive_wheel_without_contact(self):
        env = self.env(no_contact=True)
        action = np.zeros(24); action[[0, 6]] = .1; action[[3, 9]] = .05
        before = env.data.qpos.copy()
        env._integrate_action(action)
        np.testing.assert_array_equal(env.data.qpos, before)
        for _ in range(30):
            obs, _, done, _, info = env.step(action)
            self.assertFalse(done)
        self.assertEqual(obs.shape, (137,))
        self.assertLess(abs(info['wheel_deg']), 1e-8)
        self.assertTrue((env.contact_forces == 0).all())

    def test_total_virtual_wrist_wrench_is_bounded_including_gravity_compensation(self):
        env = self.env()
        env.positions += .5
        env.rotations = np.array([m @ np.diag([-1., -1., 1.]) for m in env.rotations])
        env._physics_step()
        self.assertTrue((np.linalg.norm(env.applied_wrenches[:, :3], axis=1) <= env.config.force_limit_n+1e-9).all())
        self.assertTrue((np.linalg.norm(env.applied_wrenches[:, 3:], axis=1) <= env.config.torque_limit_nm+1e-9).all())

    def test_actor_does_not_depend_on_contact_readouts_or_dependent_finger_positions(self):
        env = self.env()
        first = env._observation()
        env.contact_forces[:] = 1000.
        env.data.qfrc_actuator[:] = 999.
        active = set(env.hand_q.tolist())
        for i in range(env.model.njnt):
            if env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                q = env.model.jnt_qposadr[i]
                if q not in active and q != env.wheel_q:
                    env.data.qpos[q] += .1
        # Active positions, wrist/root pose and wheel feedback stay fixed.
        # Recompute FK so this also checks real independence, not stale sites.
        mujoco.mj_forward(env.model, env.data)
        second = env._observation()
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()
