"""Distance constraints must differentiate the physical signed distance."""
import unittest
import numpy as np,mujoco
from unified_steering.structured.driving_task import DrivingConfig,DrivingEnv

class DistanceGradientTests(unittest.TestCase):
    def test_mixed_shape_gradient_matches_finite_difference_in_both_pair_orders(self):
        env=DrivingEnv(DrivingConfig(canonical_distance_guard_order=True))
        try:
            env.reset(seed=42001,options=dict(free_side=-1,exact_curriculum=True,direction=1,template='short'))
            c=env.controller;m=env.model;d=c.command_data
            d.qpos[:]=env.data.qpos;mujoco.mj_forward(m,d)
            palm=int(np.flatnonzero((m.geom_bodyid==m.body('left_wrist_yaw_link').id)&(m.geom_contype!=0))[0])
            elbow=m.geom('right_elbow_link_guard').id
            c.config.collision_activation_m=1.
            reference=d.qpos.copy();eps=1e-6
            for pair in [(palm,elbow),(elbow,palm)]:
                c.guard_pairs=[pair]
                rows,bounds=c._guard_constraints();self.assertEqual(len(rows),1)
                base=mujoco.mj_geomDistance(m,d,*pair,1.,np.zeros(6));finite=[]
                for index in c.arm_q:
                    d.qpos[:]=reference;d.qpos[index]+=eps;mujoco.mj_forward(m,d)
                    finite.append((mujoco.mj_geomDistance(m,d,*pair,1.,np.zeros(6))-base)/eps)
                d.qpos[:]=reference;mujoco.mj_forward(m,d)
                np.testing.assert_allclose(rows[0],finite,atol=4e-6,rtol=3e-4)
                self.assertAlmostEqual(bounds[0],4*(c.config.collision_margin_m-base),places=10)
        finally:env.close()

if __name__=='__main__':unittest.main()
