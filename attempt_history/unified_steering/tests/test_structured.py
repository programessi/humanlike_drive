"""Check the new task's physical and information boundaries, not PPO success."""
import unittest
import mujoco
import numpy as np
from unified_steering.structured.env import SteeringTeacherEnv, TaskConfig
from unified_steering.structured.parking import SupportTeacherEnv


class StructuredTests(unittest.TestCase):
    def env(self, **kwargs):
        env = SteeringTeacherEnv(TaskConfig(episode_seconds=1., **kwargs))
        self.addCleanup(env.close)
        env.reset(seed=1007)
        return env

    def test_load_persists_without_contact(self):
        env = self.env(no_contact=True)
        env.data.qpos[env.wheel_q] = .3
        mujoco.mj_forward(env.model, env.data)
        env.measuring = False
        env._physics_step()
        self.assertLess(env.data.qfrc_applied[env.wheel_v], -.08)
        self.assertFalse(any(g in env.wheel_geoms for c in env.data.contact for g in (c.geom1,c.geom2)))

    def test_reference_advances_without_progress(self):
        env = self.env(no_contact=True, target_mode="step")
        expected = env.command_angles.copy()
        wheel = env.wheel_angle
        env.time = .75
        env._update_target()
        self.assertNotEqual(env.target_angle, wheel)
        np.testing.assert_array_equal(expected, env.command_angles)
        self.assertEqual(env.wheel_angle, wheel)

    def test_actual_arm_margin_terminates_at_physics_step(self):
        env = self.env()
        joint = env.controller.arm_jids[0]
        env.data.qpos[env.model.jnt_qposadr[joint]] = env.model.jnt_range[joint,0]+.01
        env._advance_action = lambda action: env._physics_step()
        _, _, terminated, _, info = env.step(np.zeros(24))
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "arm_margin")

    def test_actor_excludes_privileged_contact_and_force(self):
        env = self.env()
        before = env._sensor_frame(initialize=True)
        env.contact_forces[:] = 1000
        env.data.qfrc_actuator[:] = 999
        env.data.qpos[env.dependent_q] += .1
        after = env._sensor_frame(initialize=True)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before.shape, (152,))

    def test_parked_hand_is_clear_but_not_locked_during_execution(self):
        env = SupportTeacherEnv(TaskConfig(episode_seconds=1.))
        self.addCleanup(env.close)
        env.reset(seed=1107, options=dict(free_side=1, parking_profile="axial_near",
                                         phase_deg=-5., free_opening_delta_rad=-.02))
        self.assertGreater(env.free_clearance(), .01)
        before = env.controller.wrist_positions[1].copy()
        action = np.zeros(24)
        action[6] = .2
        _, _, terminated, _, info = env.step(action)
        self.assertFalse(terminated)
        self.assertGreater(np.linalg.norm(env.controller.wrist_positions[1]-before), 1e-6)
        self.assertEqual(info["physics_free_hand_force_peak_n"], 0.)

    def test_leaky_hand_neutral_preserves_start_and_both_signs_execute(self):
        commands = []
        for sign in (0., -1., 1.):
            env = SupportTeacherEnv(TaskConfig(hand_action_mode="leaky"))
            self.addCleanup(env.close)
            env.reset(seed=1007, options={"free_side": 1})
            before = env.controller.target[14:].copy()
            action = np.zeros(24)
            action[12:] = .2*sign
            env.controller.update(env.data, action, .02)
            commands.append(env.controller.target[14:]-before)
        np.testing.assert_allclose(commands[0], 0., atol=1e-12)
        self.assertTrue(np.all(commands[1]<0.))
        self.assertTrue(np.all(commands[2]>0.))

    def test_reward_precision_changes_reward_without_changing_physics_or_observation(self):
        envs=[self.env(angle_reward_sigma_deg=sigma) for sigma in (4.,2.)]
        changed=False
        for _ in range(20):
            action=np.r_[np.full(12,.03),np.zeros(12)]
            left=envs[0].step(action)
            right=envs[1].step(action)
            np.testing.assert_array_equal(envs[0].data.qpos,envs[1].data.qpos)
            for key in left[0]:
                np.testing.assert_array_equal(left[0][key],right[0][key])
            changed=changed or abs(left[1]-right[1])>1e-6
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
