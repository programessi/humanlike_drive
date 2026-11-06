"""Diagnostic instrumentation must not alter unmodified execution."""
import unittest
import numpy as np

from unified_steering.structured.env import TaskConfig
from unified_steering.structured.parking import SupportTeacherEnv
from unified_steering.structured.interventions import DiagnosticConfig, DiagnosticEnv, intervention_action


class StructuredInterventionTests(unittest.TestCase):
    def test_none_instrumentation_is_physics_observation_and_reward_identical(self):
        normal=SupportTeacherEnv(TaskConfig(hand_action_mode="leaky",hand_leak_alpha=.995))
        diagnostic=DiagnosticEnv(DiagnosticConfig(hand_action_mode="leaky",hand_leak_alpha=.995))
        self.addCleanup(normal.close);self.addCleanup(diagnostic.close)
        a,_=normal.reset(seed=1007,options={"free_side":0,"direction":-1})
        b,_=diagnostic.reset(seed=1007,options={"free_side":0,"direction":-1})
        rng=np.random.default_rng(42)
        for _ in range(12):
            for key in a:
                np.testing.assert_array_equal(a[key],b[key])
            action=rng.uniform(-.05,.05,24)
            a,ra,ta,xa,_=normal.step(action)
            b,rb,tb,xb,info=diagnostic.step(action)
            np.testing.assert_array_equal(normal.data.qpos,diagnostic.data.qpos)
            self.assertEqual((ra,ta,xa),(rb,tb,xb))
        self.assertEqual(sum(info["audit_workspace_clips"]),diagnostic.controller.workspace_clips)

    def test_action_mask_changes_only_declared_channel_and_keeps_input(self):
        original=np.linspace(-.2,.2,24)
        masked=intervention_action(original,"radial_zero",1)
        expected=original.copy();expected[7]=0.
        np.testing.assert_array_equal(masked,expected)
        self.assertNotEqual(original[7],0.)

    def test_hand_anchor_is_diagnostic_target_request_not_state_assignment(self):
        env=DiagnosticEnv(DiagnosticConfig(hand_action_mode="leaky",hand_leak_alpha=.995,
                                           intervention="hand_initial"))
        self.addCleanup(env.close)
        env.reset(seed=1007,options={"free_side":0})
        env.controller.update(env.data,np.zeros(24),.02)
        env.controller.diagnostic_fixed_hand=(1,env.diagnostic_hand_anchor[1])
        before=env.data.qpos.copy()
        desired=env.controller.hand_targets(np.ones(12))
        np.testing.assert_array_equal(desired[6:],env.diagnostic_hand_anchor[1])
        np.testing.assert_array_equal(env.data.qpos,before)

    def test_hand_lead_override_restores_config_and_does_not_change_arm_bounds(self):
        env=DiagnosticEnv(DiagnosticConfig(hand_action_mode="leaky",hand_leak_alpha=.995))
        self.addCleanup(env.close)
        env.reset(seed=1007,options={"free_side":0})
        c=env.controller
        arm_args=(slice(0,14),c.config.arm_speed,c.config.arm_acceleration,c.limits,
                  env.data.qpos[c.arm_q],.02)
        before=c._bounds(*arm_args)
        c.diagnostic_hand_lead=.15
        after=c._bounds(*arm_args)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        c._bounds(slice(14,26),c.config.hand_speed,c.config.hand_acceleration,c.hand_limits,
                  env.data.qpos[c.hand_q],.02)
        self.assertAlmostEqual(c.config.joint_lead,np.deg2rad(5))

    def test_versioned_hand_lead_matches_diagnostic_without_changing_preparation(self):
        direct=SupportTeacherEnv(TaskConfig(hand_action_mode="leaky",hand_leak_alpha=.995,hand_joint_lead_rad=.15))
        diagnostic=DiagnosticEnv(DiagnosticConfig(hand_action_mode="leaky",hand_leak_alpha=.995,
                                                  intervention="hand_lead_0p15"))
        self.addCleanup(direct.close);self.addCleanup(diagnostic.close)
        a,_=direct.reset(seed=1007,options={"free_side":0,"direction":-1})
        b,_=diagnostic.reset(seed=1007,options={"free_side":0,"direction":-1})
        np.testing.assert_array_equal(direct.data.qpos,diagnostic.data.qpos)
        np.testing.assert_array_equal(direct.model.actuator_forcerange,diagnostic.model.actuator_forcerange)
        rng=np.random.default_rng(12)
        for _ in range(12):
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            action=rng.uniform(-.03,.03,24)
            a,ra,ta,xa,_=direct.step(action)
            b,rb,tb,xb,_=diagnostic.step(action)
            np.testing.assert_array_equal(direct.data.qpos,diagnostic.data.qpos)
            self.assertEqual((ra,ta,xa),(rb,tb,xb))
        self.assertAlmostEqual(direct.controller.config.joint_lead,np.deg2rad(5))


if __name__=="__main__":
    unittest.main()
