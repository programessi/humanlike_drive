import unittest
import numpy as np
from unified_steering.structured.driving_task import DrivingConfig,DrivingEnv,timed_reference

class DrivingTaskTests(unittest.TestCase):
    def test_full_both_direction_sequence_has_no_wrap_or_truncation(self):
        a,v,t=timed_reference(.02,np.deg2rad(180),-1,'sequence')
        self.assertEqual(len(t['segments']),4)
        self.assertLess(min(a),-3.14);self.assertGreater(max(a),3.14)
        self.assertGreater(t['seconds'],270.)
        self.assertAlmostEqual(a[round(t['seconds']/.05)],0.)
        self.assertLessEqual(max(abs(v)),np.deg2rad(5)+1e-10)

    def test_software_envelope_is_explicit_and_does_not_enable_540(self):
        DrivingConfig(amplitude_deg=180).validate()
        for amplitude in [186,540]:
            with self.assertRaises(ValueError):DrivingConfig(amplitude_deg=amplitude).validate()

    def test_reset_handoff_preserves_physics_and_exposes_bounded_target(self):
        states=[]
        for retain in (False,True):
            env=DrivingEnv(DrivingConfig(amplitude_deg=30,retain_preparation_hand_command=retain))
            self.addCleanup(env.close)
            obs,_=env.reset(seed=45501,options=dict(direction=1,template='short',free_side=-1))
            c=env.controller
            states.append((env.data.qpos.copy(),env.data.qvel.copy(),env.model.jnt_range.copy(),obs,c.target.copy()))
            expected=env.data.qpos[c.hand_q].copy()
            if retain:expected+=np.clip(c.closed_hand-expected,-.15,.15)
            np.testing.assert_allclose(c.target[14:],expected,atol=1e-12)
            np.testing.assert_allclose(obs['actor'][66:78],(c.target[14:]-env.data.qpos[c.hand_q])/.15,atol=1e-6)
        for k in (0,1,2):np.testing.assert_array_equal(states[0][k],states[1][k])
        self.assertGreater(np.linalg.norm(states[0][4][14:]-states[1][4][14:]),.01)

    def test_tightened_reserve_and_added_guards_do_not_relax_hard_gate(self):
        for reserve in (3.,6.):
            env=DrivingEnv(DrivingConfig(amplitude_deg=30,arm_command_margin_deg=reserve,guard_elbows_against_wheel=True))
            self.addCleanup(env.close)
            env.reset(seed=41501,options=dict(direction=1,template='short',free_side=-1))
            c=env.controller
            wheel=env.model.body('steering_wheel').id
            self.assertTrue(any(env.model.geom_bodyid[b]==wheel for a,b in c.guard_pairs))
            self.assertEqual(c.arm_command_margin_deg,reserve)
            env.data.qpos[c.arm_q[0]]=c.limits[0,0]+np.deg2rad(1.8)
            env.free_force_peak=0.
            env._advance_action=lambda action:env._physics_step()
            _,_,terminated,_,info=env.step(np.zeros(24))
            self.assertTrue(terminated);self.assertEqual(info['termination_reason'],'arm_margin')
        for reserve in (2.,9.):
            with self.assertRaises(ValueError):DrivingConfig(arm_command_margin_deg=reserve).validate()

    def test_demonstrator_prediction_does_not_write_simulator_or_executor(self):
        from unified_steering.structured import driving_roll_integral_audit as roll
        from unified_steering.structured.driving_halfturn_expert import HalfturnDemonstrator
        roll._TABLE='results/driving_r0_geometry_roll_v1/summary.json'
        env=DrivingEnv(DrivingConfig(amplitude_deg=180,retain_preparation_hand_command=True))
        self.addCleanup(env.close)
        obs,_=env.reset(seed=45501,options=dict(direction=-1,template='short',free_side=-1))
        c=env.controller
        before=[x.copy() for x in (env.data.qpos,env.data.qvel,env.data.ctrl,c.target,c.wrist_positions,c.wrist_rotations)]
        expert=HalfturnDemonstrator(env)
        action,_=expert.predict(obs)
        self.assertEqual(action.shape,(24,));self.assertTrue(np.isfinite(action).all())
        for a,b in zip(before,(env.data.qpos,env.data.qvel,env.data.ctrl,c.target,c.wrist_positions,c.wrist_rotations)):
            np.testing.assert_array_equal(a,b)

    def test_upstream_reference_rejects_unexecutable_sequence_without_mutation(self):
        env=DrivingEnv(DrivingConfig(amplitude_deg=30));self.addCleanup(env.close)
        env.reset(seed=41501,options=dict(direction=1,template='short',free_side=-1))
        before=env.command_angles.copy();q=env.data.qpos.copy()
        with self.assertRaises(ValueError):env.set_reference([0,1,2],[0,1,2])
        np.testing.assert_array_equal(before,env.command_angles)
        np.testing.assert_array_equal(q,env.data.qpos)
        env.set_reference([0,1,2,3],[0,.01,.01,0])
        np.testing.assert_array_equal(q,env.data.qpos)

if __name__=='__main__':unittest.main()
