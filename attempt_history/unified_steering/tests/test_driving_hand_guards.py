"""Regression coverage for real cross-hand finger collisions and guard pruning."""
import unittest
import numpy as np
import mujoco
from unified_steering.structured.driving_task import DrivingConfig,DrivingEnv

class FullHandGuardTests(unittest.TestCase):
    def setUp(self):
        self.env=DrivingEnv(DrivingConfig(canonical_distance_guard_order=True,guard_opposite_hand_meshes=True,guard_full_physical_pairs=True))
        self.env.reset(seed=42001,options=dict(free_side=-1,exact_curriculum=True,direction=1,template='short'))
        c=self.env.controller;c.command_data.qpos[:]=self.env.data.qpos
        mujoco.mj_forward(self.env.model,c.command_data)
    def tearDown(self):self.env.close()
    def test_every_physical_cross_hand_pair_has_a_guard(self):
        e=self.env;m=e.model;pairs={frozenset(p) for p in e.controller.guard_pairs}
        groups=[[g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_contype[g]] for bodies in e.hand_body_groups]
        for a in groups[0]:
            for b in groups[1]:self.assertIn(frozenset((a,b)),pairs)
        # This exact body pair was absent from the old sparse guards.
        bodies=[m.body('L_ring_intermediate').id,m.body('R_pinky_intermediate').id]
        geoms=[next(g for g in range(m.ngeom) if m.geom_bodyid[g]==body and m.geom_contype[g]) for body in bodies]
        self.assertIn(frozenset(geoms),pairs)
    def test_column_contacts_are_guarded_and_hand_wheel_contacts_remain_allowed(self):
        e=self.env;m=e.model;pairs={frozenset(p) for p in e.controller.guard_pairs}
        column=m.geom('column').id
        for bodies in e.hand_body_groups:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] in bodies and m.geom_contype[g]:
                    self.assertIn(frozenset((g,column)),pairs)
                    for rim in e.wheel_geoms:
                        self.assertNotIn(frozenset((g,rim)),pairs)

    def test_bounding_sphere_pruning_preserves_all_active_constraints(self):
        c=self.env.controller;c.config.collision_activation_m=.5
        c.guard_opposite_hand_meshes=False;c.guard_full_physical_pairs=False;expected=c._guard_constraints()
        c.guard_opposite_hand_meshes=True;c.guard_full_physical_pairs=True;actual=c._guard_constraints()
        self.assertGreater(len(expected[0]),0)
        np.testing.assert_allclose(actual[0],expected[0],atol=1e-12)
        np.testing.assert_allclose(actual[1],expected[1],atol=1e-12)
    def test_mesh_mesh_distance_gradient_matches_finite_difference(self):
        e=self.env;c=e.controller;m=e.model;d=c.command_data
        bodies=[m.body('L_ring_intermediate').id,m.body('R_pinky_intermediate').id]
        pair=tuple(next(g for g in range(m.ngeom) if m.geom_bodyid[g]==body and m.geom_contype[g]) for body in bodies)
        c.config.collision_activation_m=1.;reference=d.qpos.copy();eps=1e-6
        for ordered in (pair,pair[::-1]):
            c.guard_pairs=[ordered];rows,_=c._guard_constraints();self.assertEqual(len(rows),1)
            base=mujoco.mj_geomDistance(m,d,*ordered,1.,np.zeros(6));finite=[]
            for q in c.arm_q:
                d.qpos[:]=reference;d.qpos[q]+=eps;mujoco.mj_forward(m,d)
                finite.append((mujoco.mj_geomDistance(m,d,*ordered,1.,np.zeros(6))-base)/eps)
            d.qpos[:]=reference;mujoco.mj_forward(m,d)
            np.testing.assert_allclose(rows[0],finite,atol=1e-5,rtol=5e-4)

if __name__=='__main__':unittest.main()
