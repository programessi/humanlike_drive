import copy
import unittest

import numpy as np
import yaml

from unified_steering.audit import ROOT
from unified_steering.disturbance import WheelDisturbance
from unified_steering.env import SteeringEnv
from unified_steering.hold_baseline import summarize_noise
from unified_steering.telemetry import PHYSICS_FIELDS


class DisturbanceTests(unittest.TestCase):
    def test_exact_half_open_pulse_and_impulse(self):
        pulse=WheelDisturbance(dict(pulses=[dict(start_s=2.,duration_s=.2,torque_nm=.05)]),.002)
        values=np.array([pulse.torque(i*.002) for i in range(1200)])
        np.testing.assert_array_equal(np.flatnonzero(values),np.arange(1000,1100))
        self.assertAlmostEqual(values.sum()*.002,.01,places=14)
        self.assertEqual(pulse.torque(2.2),0.)

    def test_invalid_or_overlapping_pulses_rejected(self):
        for config in (dict(max_abs_torque_nm=.1),dict(max_abs_torque_nm=float('nan')),
                       dict(pulses=[dict(start_s=1.001,duration_s=.2,torque_nm=.05)]),
                       dict(pulses=[dict(start_s=1.,duration_s=1e-12,torque_nm=.05)]),
                       dict(pulses=[dict(start_s=1.,duration_s=.2,torque_nm=.051)]),
                       dict(pulses=[dict(start_s=1.,duration_s=.2,torque_nm=.05)]*2)):
            with self.subTest(config=config),self.assertRaises(ValueError):WheelDisturbance(config,.002)

    def test_pulse_adds_to_centering_with_identical_pre_pulse_physics(self):
        base=yaml.safe_load((ROOT/'configs/m1.yaml').read_text())
        base.update(initial_support='both',diagnostics=True,episode_seconds=3.)
        trajectories=[]
        for torque in (0.,.05,-.05):
            config=copy.deepcopy(base)
            config['wheel_disturbance']=dict(pulses=[dict(start_s=2.,duration_s=.2,torque_nm=torque)])
            env=SteeringEnv(config);_,initial=env.reset(seed=200,options={'support':'both'})
            env.set_reference([initial['initial_angle_rad']])
            rows=[];states=[]
            for _ in range(50):
                obs,_,term,_,info=env.step(env.hold_action())
                self.assertFalse(term)
                self.assertEqual(obs['actor'].shape,(152,))
                states.append(env.sim.data.qpos.copy());rows.append(info['diagnostics']['physics'])
            physics=np.concatenate(rows);states=np.array(states)
            d=physics[:,PHYSICS_FIELDS.index('disturbance_torque_nm')]
            total=physics[:,PHYSICS_FIELDS.index('total_wheel_torque_nm')]
            centering=physics[:,PHYSICS_FIELDS.index('applied_centering_nm')]
            np.testing.assert_allclose(total,centering+d,atol=1e-14,rtol=0)
            self.assertAlmostEqual(d.sum()*env.dt,torque*.2,places=12)
            trajectories.append(states)
        for states in trajectories[1:]:np.testing.assert_array_equal(states[:40],trajectories[0][:40])
        self.assertGreater(np.max(np.abs(trajectories[1][40:]-trajectories[0][40:])),1e-8)

    def test_noise_qualification_does_not_compare_early_failure_prefixes(self):
        p=yaml.safe_load((ROOT/'configs/m31_hold_baseline.yaml').read_text())
        p['noise_durations_s']=[3.]
        rows=[dict(name=name,seed=seed,seconds=3.,complete=name=='arm_low',hold_pass=False,rmse_deg=.01)
              for name in ('default','arm_low') for seed in range(10)]
        result=summarize_noise(rows,p)['3.0']['comparison']
        self.assertEqual(result['common_complete_pairs'],0)
        self.assertIsNone(result['common_complete_rmse_ratio'])
        self.assertFalse(result['exploration_candidate_pass'])


if __name__=='__main__':unittest.main()
