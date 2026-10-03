"""Checks that protect the causal path: motors -> hands -> physical wheel."""
import unittest

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from humanlike_drive.env import SteeringEnv, TaskConfig


class PhysicsTests(unittest.TestCase):
    def test_model_has_fixed_torso_and_no_wheel_motor(self):
        env=SteeringEnv()
        try:
            joints={env.model.joint(i).name for i in range(env.model.njnt)}
            self.assertNotIn("floating_base_joint",joints)
            self.assertNotIn("torso_joint",joints)
            self.assertFalse(any("hip" in j or "knee" in j or "ankle" in j for j in joints))
            wheel=env.model.joint("wheel_joint").id
            self.assertNotIn(wheel,env.model.actuator_trnid[:,0])
            self.assertEqual(env.model.neq,12)
        finally:env.close()

    def test_no_contact_wheel_stays_stationary(self):
        env=SteeringEnv(TaskConfig(target_mode="sine",episode_seconds=2,integral_gain=1))
        try:
            for g in env.wheel_geoms:
                env.model.geom_contype[g]=0;env.model.geom_conaffinity[g]=0
            env.reset(seed=3)
            for _ in range(100):
                _,_,terminated,_,info=env.step(np.zeros(26))
                self.assertFalse(terminated)
                self.assertLess(abs(env.wheel_angle),1e-8)
                self.assertEqual(info["left_force"]+info["right_force"],0)
        finally:env.close()

    def test_hands_can_move_the_passive_wheel(self):
        env=SteeringEnv(TaskConfig(episode_seconds=3))
        try:
            env.reset(seed=1)
            start=env.wheel_angle
            env.set_target(np.deg2rad(15))
            contacts=[]
            for _ in range(120):
                _,_,terminated,_,info=env.step(np.zeros(26))
                self.assertFalse(terminated)
                contacts.append(info["left_force"]+info["right_force"])
            self.assertGreater(env.wheel_angle-start,np.deg2rad(8))
            self.assertGreater(max(contacts),.2)
            self.assertEqual(info["non_hand_wheel_force"],0)
        finally:env.close()

    def test_gym_reset_and_step_contract(self):
        env=SteeringEnv(TaskConfig(settle_steps=10))
        try:check_env(env,skip_render_check=True)
        finally:env.close()

    def test_integral_feedback_tracks_a_hold_without_direct_wheel_control(self):
        env=SteeringEnv(TaskConfig(episode_seconds=12,integral_gain=1))
        try:
            env.reset(seed=1)
            env.set_target(np.deg2rad(15),velocity=.1)
            self.assertAlmostEqual(env.target_velocity,.1)
            for _ in range(500):
                _,_,terminated,_,info=env.step(np.zeros(26))
                self.assertFalse(terminated)
            self.assertLess(abs(info["error_deg"]),3)
            self.assertEqual(info["non_hand_wheel_force"],0)
        finally:env.close()


if __name__=="__main__":unittest.main()
