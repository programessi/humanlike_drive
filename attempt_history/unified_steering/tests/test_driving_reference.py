"""Driving references must complete their timed outbound and zero return."""
import unittest
import numpy as np
from unified_steering.structured.driving_audit import reference

class DrivingReferenceTests(unittest.TestCase):
    def test_absolute_reference_completes_outbound_and_zero_return(self):
        for start,goal in [(1.6,90.),(-1.6,-90.),(4.,20.)]:
            with self.subTest(start=start,goal=goal):
                angles,velocities,t=reference(np.deg2rad(start),np.deg2rad(goal))
                times=np.arange(len(angles))*.05
                self.assertAlmostEqual(angles[0],np.deg2rad(start))
                hold=(times>=t['outbound_end']+.01)&(times<=t['return_start']-.01)
                self.assertTrue(hold.any())
                np.testing.assert_allclose(angles[hold],np.deg2rad(goal),atol=1e-10)
                tail=times>=t['return_end']+.01
                np.testing.assert_allclose(angles[tail],0.,atol=1e-10)
                np.testing.assert_allclose(velocities[tail],0.,atol=1e-10)
                self.assertGreaterEqual(t['seconds'],t['return_end']+3.)
                self.assertLessEqual(max(abs(velocities)),np.deg2rad(5)+1e-10)
                self.assertLessEqual(max(abs(np.diff(velocities)/.05)),np.deg2rad(40)+1e-8)

    def test_large_reference_remains_accumulated_and_is_not_a_22_second_prefix(self):
        angles,_,t=reference(0.,np.deg2rad(540))
        self.assertAlmostEqual(np.rad2deg(max(angles)),540)
        self.assertGreater(t['seconds'],400)
        self.assertAlmostEqual(angles[round(t['seconds']/.05)],0.)

    def test_reference_rejects_invalid_timing(self):
        with self.assertRaises(ValueError):reference(0.,1.,speed_deg_s=0.)

class PhysicalBaselineIsolationTests(unittest.TestCase):
    def test_tangent_baselines_do_not_modify_physics_or_executor_targets(self):
        import json,tempfile
        from pathlib import Path
        from unified_steering.structured import driving_roll_audit,driving_roll_integral_audit
        from unified_steering.structured.demonstrator import DemonstratorConfig
        from unified_steering.structured.driving_audit import DrivingAuditEnv,make_config
        protocol={'speed_deg_s':5.,'acceleration_deg_s2':40.}
        env=DrivingAuditEnv(make_config(protocol,30.));self.addCleanup(env.close)
        obs,_=env.reset(seed=40001,options={'free_side':-1,'direction':1,'mode':'step'})
        rows=[{'rotation_from_prepared_grip_deg':d,'left':{'tangent_roll_deg':d/3},
               'right':{'tangent_roll_deg':-d/3}} for d in [-90,0,90]]
        with tempfile.TemporaryDirectory() as temp:
            table=Path(temp)/'table.json';table.write_text(json.dumps({'cases':rows}))
            for module in [driving_roll_audit,driving_roll_integral_audit]:
                prior=module._TABLE
                try:
                    module._TABLE=str(table)
                    expert=module.TangentRollDemonstrator(env,DemonstratorConfig(angle_gain=6.,hand_gain=.1))
                    q=env.data.qpos.copy();v=env.data.qvel.copy();ctrl=env.data.ctrl.copy()
                    targets=env.controller.target.copy();limits=env.model.jnt_range.copy()
                    for _ in range(3):
                        action,_=expert.predict(obs)
                        self.assertEqual(action.shape,(24,));self.assertTrue(np.isfinite(action).all())
                        self.assertLessEqual(np.abs(action).max(),1.)
                    for before,after in [(q,env.data.qpos),(v,env.data.qvel),(ctrl,env.data.ctrl),
                                         (targets,env.controller.target),(limits,env.model.jnt_range)]:
                        np.testing.assert_array_equal(before,after)
                    for side in range(2):
                        pos,rotation=expert.pose(side,np.deg2rad(60))
                        self.assertTrue(np.isfinite(pos).all())
                        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=1e-10)
                finally:module._TABLE=prior

if __name__=='__main__':unittest.main()
