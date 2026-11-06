"""Independent action-scale and label-frame checks for coordinated intentions."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import tempfile,unittest
import numpy as np,torch
from unified_steering.structured.driving_coordination_policy import CoordinationPolicy,CoordinationModel
from unified_steering.structured.driving_pose_policy import PoseModel

class CoordinationPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.parent=PoseModel.load(Path('tests/fixtures/legacy_invariants/models/driving_r1_pose_v41/teacher.zip')).policy
        self.z=np.load('tests/fixtures/legacy_invariants/data/halfturn/a180_d+1_short_s48001.npz')
    def test_zero_residual_preserves_physical_wrist_velocities_and_hand_actions(self):
        p=CoordinationPolicy();p.base=deepcopy(self.parent)
        a=torch.tensor(self.z['actor'][::71]);initial=torch.tensor(np.tile(np.r_[a[0,102:121],a[0,14:26]+.15*a[0,66:78]],(len(a),1)))
        phase=torch.linspace(-.1,.1,len(a));omega=p.base.velocity_head(a[:,122:129])[:,0]
        intent=p.intention(a,initial,phase,omega,torch.zeros_like(phase));new=p.action_from_intent(a,intent,omega);old=p.base.action(a,initial,phase,omega)
        scale=torch.tensor(([.5]*3+[2/3]*3)*2+[1.]*12)
        torch.testing.assert_close(new,old*scale,atol=2e-6,rtol=2e-6)
        new.square().sum().backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))
    def test_saved_intentions_reconstruct_expert_ordinary24_actions(self):
        p=CoordinationPolicy();p.base=deepcopy(self.parent);ids=np.arange(0,len(self.z['actor']),17)
        with torch.no_grad():
            action=p.action_from_intent(torch.tensor(self.z['actor'][ids]),torch.tensor(np.c_[self.z['pose'][ids],self.z['hand_target'][ids]],dtype=torch.float32),torch.tensor(self.z['omega'][ids],dtype=torch.float32)).clamp(-1,1).numpy()
        np.testing.assert_allclose(action,self.z['action'][ids],atol=2e-6)
    def test_reload_reset_and_execution_without_expert(self):
        p=CoordinationPolicy();p.base=deepcopy(self.parent);model=CoordinationModel(p);obs={'actor':self.z['actor'][0]}
        with patch('unified_steering.structured.driving_shuffle_expert.ShuffleDemonstrator.predict',side_effect=RuntimeError('expert queried')),patch('unified_steering.structured.driving_pose_policy.geometry_labels',side_effect=RuntimeError('table queried')):
            expected=model.predict(obs)[0];model.reset();np.testing.assert_array_equal(expected,model.predict(obs)[0])
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'teacher.zip';model.save(path);loaded=CoordinationModel.load(path);np.testing.assert_array_equal(expected,loaded.predict(obs)[0])
if __name__=='__main__':unittest.main()
