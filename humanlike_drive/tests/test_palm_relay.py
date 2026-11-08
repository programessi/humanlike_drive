"""Task-level invariants of the unwrapped-angle relay planner."""

import unittest
import numpy as np
from humanlike_drive.controller.palm_relay import RelayPlan


class RelayTaskTest(unittest.TestCase):
    def test_contact_wait_preserves_past_motion_and_shifts_future_acceptance_windows(
        self,
    ):
        plan = RelayPlan([180, 0], stroke_deg=40, speed_deg_s=1)
        entering = next(
            i for i, s in enumerate(plan.segments) if s["kind"] == "transfer"
        )
        boundary = plan.segments[entering]["start"]
        times = np.linspace(0, boundary - 0.01, 30)
        before = [plan.sample(t)["reference"] for t in times]
        solo = plan.handovers[0]["solo_begin"]
        final = plan.duration
        plan.extend_stationary_wait(entering, 0.5)
        np.testing.assert_allclose(before, [plan.sample(t)["reference"] for t in times])
        self.assertEqual(plan.sample(boundary + 0.2)["kind"], "settle")
        self.assertEqual(plan.sample(boundary + 0.6)["kind"], "transfer")
        self.assertAlmostEqual(plan.handovers[0]["solo_begin"], solo + 0.5)
        self.assertAlmostEqual(plan.duration, final + 0.5)
        self.assertAlmostEqual(
            plan.handovers[0]["end"] - plan.handovers[0]["solo_begin"], 10.0
        )

    def test_mixed_targets_remain_continuous_and_preserve_unwrapped_progress(self):
        plan = RelayPlan([360, 180, -90, 0], stroke_deg=40)
        holds = [s["ref1"] for s in plan.segments if s["kind"] == "goal_hold"]
        np.testing.assert_allclose(np.rad2deg(holds), [360, 180, -90, 0], atol=1e-9)
        for before, after in zip(plan.segments, plan.segments[1:]):
            self.assertAlmostEqual(before["ref1"], after["ref0"])
            np.testing.assert_allclose(before["phases1"], after["phases0"])
            np.testing.assert_allclose(before["retreat1"], after["retreat0"])
        self.assertAlmostEqual(plan.sample(plan.duration)["reference"], 0)

    def test_upper_sectors_and_motion_are_consistent_without_wheel_assistance(self):
        plan = RelayPlan([-180, -75, 30, 0], stroke_deg=40)
        for segment in plan.segments:
            p = np.array([segment["phases0"], segment["phases1"]])
            self.assertTrue(np.all(p[:, 0] >= -1e-10))
            self.assertTrue(np.all(p[:, 0] <= np.deg2rad(40) + 1e-10))
            self.assertTrue(np.all(p[:, 1] >= -np.deg2rad(40) - 1e-10))
            self.assertTrue(np.all(p[:, 1] <= 1e-10))
            if segment["kind"] == "move":
                active = segment["active"]
                self.assertAlmostEqual(
                    segment["ref1"] - segment["ref0"],
                    segment["phases1"][active] - segment["phases0"][active],
                )
                self.assertEqual(segment["retreat0"][active], 0)
            else:
                self.assertAlmostEqual(segment["ref1"], segment["ref0"])

    def test_every_handover_has_a_full_independent_hold_before_resuming(self):
        plan = RelayPlan([180, 0], stroke_deg=40)
        self.assertGreater(len(plan.handovers), 1)
        for event in plan.handovers:
            self.assertAlmostEqual(event["end"] - event["solo_begin"], 10.0)
            state = plan.sample(event["solo_begin"] + 5.0)
            self.assertTrue(state["support"][event["new"]])
            self.assertFalse(state["support"][event["old"]])
            self.assertAlmostEqual(state["retreat"][event["old"]], 1.0)


if __name__ == "__main__":
    unittest.main()
