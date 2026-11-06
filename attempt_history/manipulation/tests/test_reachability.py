"""Reachability boundaries and branch continuity, independently of rewards."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from wheel_manipulation.hand_only.reachability import ArmReachabilityGuard


class ReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.guard = ArmReachabilityGuard()
        g = self.guard
        positions = np.array([g.center+g.frame@[-.002, .163, 0.],
                              g.center+g.frame@[-.002, -.163, 0.]])
        self.positions, self.rotations = g.reset(positions, np.tile(g.frame, (2, 1, 1)))

    def test_full_pose_hold_does_not_move_redundant_joints(self):
        g = self.guard
        before = g.q.copy()
        for _ in range(50):
            p, r = g.filter(self.positions, self.rotations, .05)
        np.testing.assert_array_equal(g.q, before)
        np.testing.assert_array_equal(p, self.positions)
        np.testing.assert_array_equal(r, self.rotations)

    def test_unreachable_translation_rolls_back_certificate(self):
        g = self.guard
        before = g.q.copy()
        p, r = g.filter(self.positions+[2., 0., 0.], self.rotations, .05)
        self.assertEqual(g.last_fraction, 0.)
        np.testing.assert_array_equal(g.q, before)
        np.testing.assert_array_equal(p, self.positions)
        np.testing.assert_array_equal(r, self.rotations)

    def test_position_alone_is_not_sufficient_for_orientation_feasibility(self):
        g = self.guard
        before = g.q.copy()
        rotations = self.rotations.copy()
        rotations[0] = Rotation.from_rotvec(g.frame[:, 0]*np.pi).as_matrix() @ rotations[0]
        g.filter(self.positions, rotations, .0001)
        self.assertEqual(g.last_fraction, 0.)
        np.testing.assert_array_equal(g.q, before)

    def test_accepted_motion_has_continuous_joint_witness_and_matching_fk(self):
        g = self.guard
        before = g.q.copy()
        p, r = g.filter(self.positions+[0., 0., .001], self.rotations, .05)
        self.assertGreater(g.last_fraction, 0.)
        self.assertLessEqual(np.abs(g.q-before).max(), g.config.joint_speed_rad_s*.05+1e-10)
        self.assertGreaterEqual(g.metrics['min_joint_margin_deg'], g.config.joint_margin_deg-1e-8)
        self.assertTrue(g.check_segment(before, g.q)[0])
        fk_p, fk_r = g._forward(g.q)
        np.testing.assert_allclose(p, fk_p, atol=1e-12)
        np.testing.assert_allclose(r, fk_r, atol=1e-12)

    def test_joint_limits_alone_do_not_allow_arm_torso_collision(self):
        g = self.guard
        end = g.q.copy()
        end[1] = g.lower[1]  # In joint bounds, but forearm intersects torso.
        self.assertTrue((end >= g.lower).all() and (end <= g.upper).all())
        valid, reason = g.check_segment(g.q, end)
        self.assertFalse(valid)
        self.assertEqual(reason, 'collision_guard')

    def test_actual_unreachable_palm_is_detected_without_mutating_command(self):
        g = self.guard
        before = g.q.copy()
        self.assertFalse(g.assess_actual(self.positions+[1., 0., 0.], self.rotations))
        np.testing.assert_array_equal(g.q, before)


if __name__ == '__main__':
    unittest.main()
