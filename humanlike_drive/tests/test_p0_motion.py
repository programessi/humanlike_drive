"""Schedule and evidence regressions for contact-driven P0."""
import unittest
from types import SimpleNamespace
import numpy as np

from humanlike_drive.controller.handover_plan import HandoverPlan
from humanlike_drive.controller.continuous_plan import ContinuousPlan
from humanlike_drive.env_runner.p0_motion import add_handover_evaluation


class MotionPlanTest(unittest.TestCase):
    def test_full_sequence_is_continuous_and_within_rate_limit(self):
        closed = np.ones((2, 6))
        for mirror in (1., -1.):
            plan = ContinuousPlan(mirror, 180.)
            times = np.arange(0., plan.duration, .02)
            states = [plan.sample(t, closed) for t in times]
            ref = np.array([s['reference'] for s in states])
            self.assertLessEqual(np.max(abs(np.diff(ref)/.02)), np.deg2rad(5.)+1e-8)
            self.assertAlmostEqual(ref[-1], 0.)
            self.assertAlmostEqual(mirror*np.rad2deg(ref).max() if mirror > 0 else
                                   mirror*np.rad2deg(ref).min(), 180.)
            self.assertTrue(np.any(mirror*ref < np.deg2rad(-39.)))
            # Every reference and arm target is continuous across segment edges.
            for segment in plan.segments[:-1]:
                t = segment['end']
                before, after = plan.sample(t-1e-7, closed), plan.sample(t+1e-7, closed)
                self.assertLess(abs(before['reference']-after['reference']), 1e-6)
                np.testing.assert_allclose(before['hand_phases'], after['hand_phases'], atol=1e-5)
            for cycle in plan.cycles:
                state = plan.sample(cycle['single']+4., closed)
                new = cycle['receiving']
                self.assertAlmostEqual(state['hand_phases'][new], state['reference']-cycle['new_epoch'])
                self.assertFalse(state['support'][1-new])

    def test_intermediate_shuffle_keeps_final_task_goal(self):
        plan = ContinuousPlan(1., 180.)
        for cycle in plan.cycles[1:4]:
            state = plan.sample(cycle['approach']+1., np.ones((2, 6)))
            self.assertAlmostEqual(np.rad2deg(state['target']), 180.)

    def test_handover_moves_through_all_four_stages(self):
        for angle in (15., -15.):
            plan = HandoverPlan(angle, moving=True)
            times = np.arange(plan.approach+.51, plan.target_hold-.51, .02)
            ref = np.array([plan.reference(t) for t in times])
            np.testing.assert_allclose(np.diff(ref)/.02, plan.sign*np.deg2rad(1.), atol=1e-8)


class TransferEvidenceTest(unittest.TestCase):
    def fixture(self):
        dt = .002
        n = 6500
        phase = np.r_[np.full(1500, 8), np.full(5000, 10)]
        forces = np.ones((n, 2, 6)); forces[1500:, 0] = 0.
        arrays = dict(time=np.arange(n)*dt, phase=phase, rim_forces=forces,
            load_torque=np.full(n, -.1), hand_wheel_torque=np.tile([0., .1], (n, 1)),
            wheel_angle=np.zeros(n), reference=np.zeros(n))
        plan = SimpleNamespace(receiving=1, original=0, moving=False, new_epoch=0., final_target=0.)
        return arrays, plan, dt

    def test_contact_without_signed_load_transfer_fails(self):
        arrays, plan, dt = self.fixture()
        result = {'checks': {}}
        add_handover_evaluation(result, arrays, plan, dt)
        self.assertTrue(all(result['checks'].values()))
        arrays['hand_wheel_torque'][:, 1] *= -1.
        result = {'checks': {}}
        add_handover_evaluation(result, arrays, plan, dt)
        self.assertFalse(result['checks']['handover_overlap_ge_0p1s'])
        self.assertFalse(result['checks']['handover_new_mean_counterload_torque_ge_0p02Nm'])

    def test_old_hand_contact_or_missing_finger_cannot_pass_solo_support(self):
        arrays, plan, dt = self.fixture()
        arrays['rim_forces'][1500:, 0, 0] = .1
        arrays['rim_forces'][1500:, 1, 2] = 0.
        result = {'checks': {}}
        add_handover_evaluation(result, arrays, plan, dt)
        self.assertFalse(result['checks']['handover_old_hand_clear'])
        self.assertFalse(result['checks']['handover_new_six_parts_support_95pct'])

    def test_paused_reference_cannot_pass_moving_handover(self):
        arrays, plan, dt = self.fixture()
        plan.moving = True; plan.sign = 1.; plan.approach = 0.; plan.target_hold = 13.
        result = {'checks': {}}
        add_handover_evaluation(result, arrays, plan, dt)
        self.assertFalse(result['checks']['handover_reference_keeps_moving'])


if __name__ == '__main__':
    unittest.main()
