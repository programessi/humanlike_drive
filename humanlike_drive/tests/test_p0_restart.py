"""Regression checks for the new clearance grasp and shared execution boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from humanlike_drive.controller.basic_grasps import apply
from humanlike_drive.controller.clearance_grasp import Clearance
from humanlike_drive.controller.p0_control import RateExecutor, RoundTrip, release_motion
from humanlike_drive.env.logitech_scene import PROJECT, SCENE
from humanlike_drive.env.p0_scene import build_scene
from humanlike_drive.env_runner.p0_restart import Recorder


class RestartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original_hash = hashlib.sha256(SCENE.read_bytes()).hexdigest()
        cls.model = mujoco.MjModel.from_xml_path(str(build_scene(Path(cls.temp.name)/"scene.xml")))
        cls.grasp = json.loads((PROJECT/"assets/scenes/p0_clear_grasp_v1.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        apply(self.model, self.data, "clear", {"clear": self.grasp})

    def test_inactive_control_range_does_not_zero_arm_targets(self):
        executor = RateExecutor(self.model, self.data)
        np.testing.assert_allclose(executor.target, self.data.ctrl, atol=1e-12)
        np.testing.assert_allclose(executor.follow(self.data.ctrl.copy(), self.data.qpos[executor.qids]),
                                   self.data.ctrl, atol=1e-12)

    def test_blocked_targets_keep_rate_range_and_preload_bounds(self):
        executor = RateExecutor(self.model, self.data)
        actual = self.data.qpos[executor.qids].copy()
        for _ in range(100):
            previous = executor.target.copy()
            target = executor.step(np.full(26, 10.), actual)
            self.assertTrue(np.all(abs(target-previous) <= executor.vmax*executor.dt+1e-12))
            self.assertTrue(np.all(target <= executor.bounds[:,1]+1e-12))
        self.assertTrue(np.all(abs(target-actual) <= executor.preload+1e-12))

    def test_new_grasp_clears_paddles_through_full_opening(self):
        result = Clearance(self.model).opening_sweep(self.data, samples=31)
        self.assertGreater(result["closed_paddle_clearance_mm"], 15.)
        self.assertGreater(result["opening_min_paddle_clearance_mm"], 5.)
        self.assertGreater(result["opening_min_spoke_clearance_mm"], 5.)

    def test_scene_remains_passive_and_original_unchanged(self):
        self.assertEqual(hashlib.sha256(SCENE.read_bytes()).hexdigest(), self.original_hash)
        self.assertFalse(np.any(self.model.actuator_trnid[:,0] == self.model.joint("wheel_joint").id))
        robot = self.model.geom_contype == 2
        self.assertTrue(np.all(self.model.geom_conaffinity[robot] & 2))

    def test_recorder_includes_finite_controls_and_clearance_sampling(self):
        executor = RateExecutor(self.model, self.data)
        recorder = Recorder(self.model)
        recorder.sample(self.data, 0., 0, 0., executor, self.data.ctrl.copy(), 0., [True,True])
        self.assertEqual(recorder.rows[0]["drive_total_force"].shape, (26,))
        self.assertTrue(recorder.rows[0]["clearance_sampled"])
        self.assertEqual(recorder.rows[0]["active_support"].tolist(), [True,True])

    def test_reference_respects_speed_and_complete_return(self):
        for angle in (30., -30.):
            ref = RoundTrip(angle, target_hold_s=24.)
            t = np.arange(0., ref.duration, .002)
            values = np.array([ref.sample(s)[0] for s in t])
            self.assertLessEqual(np.max(abs(np.diff(values)/.002)), np.deg2rad(5.)+1e-10)
            self.assertAlmostEqual(values[-1], 0.)
            self.assertAlmostEqual(ref.duration, 42.)

    def test_fingers_stay_curled_near_housing_during_release_and_approach(self):
        # Regression for the housing collision at +/-30 degrees: the intermediate
        # extension of the pinky is safe only after the wrist has withdrawn.
        for closed in (self.data.ctrl[14:20], self.data.ctrl[20:26]):
            for regrasp, duration in ((False, 3.), (True, 5.)):
                for t in np.linspace(0., duration, 201):
                    retreat, hand = release_motion(t, closed, regrasp=regrasp)
                    if retreat < .30:
                        self.assertTrue(np.all(hand[2:] >= .65-1e-12))
                if regrasp:
                    retreat, hand = release_motion(duration, closed, regrasp=True)
                    self.assertEqual(retreat, 0.)
                    np.testing.assert_allclose(hand, closed)
                else:
                    retreat, hand = release_motion(duration, closed)
                    self.assertEqual(retreat, 1.)
                    self.assertTrue(np.all(hand[2:] <= .1+1e-12))


if __name__ == "__main__":
    unittest.main()
