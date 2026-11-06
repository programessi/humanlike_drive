"""Safety, time reference and preparation-assistance boundaries for S00-B."""
from pathlib import Path
import hashlib
import json
import unittest

import mujoco
import numpy as np
import yaml

from unified_steering.audit import ROOT, verified_scene
from unified_steering.clearance import Clearance
from unified_steering.support import SupportFixture, blend, motion_phase


class ReferenceTests(unittest.TestCase):
    def test_time_reference_stops_reverses_and_respects_speed(self):
        segment,pause,amplitude=5.,.5,5.
        self.assertEqual(motion_phase(0,amplitude,segment,pause),0.)
        for i,value in enumerate((5.,0.,-5.,0.)):
            self.assertAlmostEqual(motion_phase(i*(segment+pause)+segment,amplitude,segment,pause),value)
        times=np.linspace(0,4*(segment+pause),10001)
        values=np.array([motion_phase(t,amplitude,segment,pause) for t in times])
        self.assertLess(np.max(np.abs(np.diff(values)/np.diff(times))),2.)
        self.assertEqual(blend(-1),0.)
        self.assertEqual(blend(2),1.)

    def test_distance_culling_preserves_infinite_planes(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="floor" type="plane" size="0 0 .1"/>
        <geom name="sphere" type="sphere" size=".1" pos="10 0 .15"/>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        reader=object.__new__(Clearance);reader.model=model
        self.assertAlmostEqual(reader.minimum(data,np.array([[0,1]])),.05,places=6)


@unittest.skipUnless((ROOT/'assets/scene.xml').exists(),'Requires local scene snapshot and robot meshes')
class SupportPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=yaml.safe_load((ROOT/'configs/s00_support.yaml').read_text())
        cls.fixture=SupportFixture(verified_scene(cls.cfg)[0],cls.cfg)

    def test_parks_and_release_paths_have_clearance(self):
        f=self.fixture
        for side in range(2):
            self.assertTrue(f.park_check(side,f.parks[side])['passed'])
            self.assertTrue(f.release_preflight(side,f.parks[side])['passed'])
            # Original grasp is deliberately not a valid parked hand state.
            self.assertFalse(f.park_check(side,f.nominal_arm[7*side:7*side+7])['passed'])

    def test_preparation_jig_removed_but_centering_remains(self):
        f=self.fixture
        prep=f.prepare(0,30.)
        self.assertGreater(prep['peak_preparation_jig_torque_nm'],.1)
        self.assertGreater(abs(prep['jig_torque_before_removal_nm']),1e-6)
        self.assertTrue(prep['initial_metrics']['preparation_jig_active'])
        f.physics_step(f.target,0.)
        metrics=f.read(0,float(f.data.qpos[f.wheel_q]))
        self.assertEqual(metrics['disturbance_torque_applied_nm'],0.)
        self.assertGreater(abs(metrics['centering_torque_applied_nm']),.1)
        self.assertEqual(metrics['right_contact_force_norm_sum_n'],0.)
        self.assertFalse(metrics['preparation_jig_active'])
        self.assertTrue(metrics['actuator_force_limits_respected'])

    def test_curated_states_restore_with_no_preparation_assistance(self):
        folder=ROOT/'assets/initial_states/s00_support_v1'
        if not (folder/'manifest.json').exists():
            self.skipTest('Curated states not generated yet')
        manifest=json.loads((folder/'manifest.json').read_text())
        for entry in manifest['states']:
            self.assertEqual(hashlib.sha256((folder/entry['file']).read_bytes()).hexdigest(),entry['sha256'])
        f=self.fixture
        for side in ('left','right'):
            entry=next(x for x in manifest['states'] if x['support']==side)
            with np.load(folder/entry['file'],allow_pickle=False) as state:
                mujoco.mj_resetData(f.model,f.data)
                for name in ('qpos','qvel','act'):
                    getattr(f.data,name)[:]=state[name]
                f.target=state['target'].copy()
                mujoco.mj_forward(f.model,f.data)
                f.data.qacc_warmstart[:]=state['qacc_warmstart']
            initial=float(f.data.qpos[f.wheel_q])
            maximum=0.
            for step in range(1500):
                f.physics_step(f.target,0.)
                maximum=max(maximum,abs(np.rad2deg(f.data.qpos[f.wheel_q]-initial)))
            metrics=f.read(0 if side=='left' else 1,initial)
            self.assertLess(maximum,2.)
            self.assertEqual(metrics['disturbance_torque_applied_nm'],0.)
            self.assertFalse(metrics['preparation_jig_active'])
            self.assertLess(abs(maximum-entry['hold_max_error_deg']),.05)


if __name__=='__main__':
    unittest.main()
