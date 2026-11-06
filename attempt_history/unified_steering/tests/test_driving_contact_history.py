"""Checks teacher-only contact scope and interface consistency."""
import unittest,json
from pathlib import Path
import numpy as np,torch
from unified_steering.structured.driving_contact_history import ContactHistoryPolicy,ContactHead
from unified_steering.structured.driving_history_intent import HistoryHead
from unified_steering.structured.driving_task import DrivingEnv,DrivingConfig
from unified_steering.structured.legacy.regrasp.controller import RegraspControlConfig

class ContactTeacherChecks(unittest.TestCase):
    def test_extra_contact_does_not_change_parent_before_training(self):
        torch.manual_seed(9);old=HistoryHead(True);new=ContactHead(True,old)
        x=torch.randn(8,263);x[:,-32:]=0.;x[torch.arange(8),-32+torch.arange(8)]=1.
        expanded=torch.cat([x[:,:231],torch.randn(8,4),x[:,231:]],-1)
        torch.testing.assert_close(new(expanded),old(x),rtol=1e-6,atol=1e-7)

    def test_teacher_refuses_missing_contact(self):
        p=ContactHistoryPolicy()
        with self.assertRaisesRegex(ValueError,'Missing privileged'):p.predict({'actor':np.zeros(152,np.float32)})

    def test_optional_contact_is_current_physics_and_space_declared(self):
        env=DrivingEnv(DrivingConfig(teacher_contact_observation=True))
        try:
            obs,_=env.reset(seed=48431,options={'free_side':-1,'exact_curriculum':True,'direction':1,'template':'short'})
            self.assertEqual(obs['teacher_contact'].shape,(4,));self.assertIn('teacher_contact',env.observation_space.spaces)
            np.testing.assert_allclose(obs['teacher_contact'][:2],env.contact_forces/20.,rtol=1e-6)
            np.testing.assert_allclose(obs['teacher_contact'][2:],env._wheel_contact_torques()[:2]/.5,rtol=1e-6,atol=1e-7)
        finally:env.close()

    def test_legacy_force_reconstruction_matches_previous_physics_frame(self):
        directory=Path('tests/fixtures/legacy_invariants/data/history');path=next((directory/'labels').glob('*.npz'));z=np.load(path);trace=np.load(directory/'traces'/path.name)
        np.testing.assert_allclose(z['privileged'][1:,38:40]*100,trace['forces_n'][:-1],rtol=2e-6,atol=2e-6)

if __name__=='__main__':unittest.main()
