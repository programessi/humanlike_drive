"""Policy-local feedback geometry and standalone weight execution."""
import tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from unified_steering.structured.driving_pose_policy import PoseModel,exp_rotation,log_rotation,rotation6

class PosePolicyTests(unittest.TestCase):
    def test_rotation_maps_match_independent_reference(self):
        vector=np.array([[.03,-.05,.07],[-.4,.2,.1],[0.,0.,0.]],np.float32)
        r=exp_rotation(torch.tensor(vector))
        np.testing.assert_allclose(r.numpy(),Rotation.from_rotvec(vector).as_matrix(),atol=2e-7)
        np.testing.assert_allclose(log_rotation(r).numpy(),vector,atol=2e-7)
        np.testing.assert_allclose(rotation6(r[:,:,:2].reshape(-1,6)).numpy(),r.numpy(),atol=2e-7)

    def test_standalone_reload_reset_and_trainable_action_graph(self):
        torch.set_num_threads(1);model=PoseModel.load(Path('tests/fixtures/legacy_invariants/models/driving_r1_pose_v41/teacher.zip'))
        z=np.load(next(Path('tests/fixtures/legacy_invariants/data/hand_actions').glob('*.npz')))
        observation=dict(actor=z['actor'][0],privileged=z['privileged'][0])
        with patch('unified_steering.structured.driving_pose_policy.geometry_labels',side_effect=RuntimeError('training source queried')):
            action=model.predict(observation)[0];self.assertEqual(action.shape,(24,));self.assertTrue(np.isfinite(action).all())
            model.reset();np.testing.assert_array_equal(model.predict(observation)[0],action)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'weights.pt';model.save(path);loaded=PoseModel.load(path)
                np.testing.assert_array_equal(loaded.predict(observation)[0],action)
        p=model.policy;a=torch.from_numpy(z['actor'][:2]);initial=torch.from_numpy(np.tile(np.r_[a[0,102:121],a[0,14:26]+.15*a[0,66:78]],(2,1)))
        omega=p.velocity_head(a[:,122:129])[:,0];out=p.action(a,initial,torch.tensor([.1,.2]),omega)
        out.square().sum().backward()
        self.assertTrue(all(v.requires_grad and v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()))

if __name__=='__main__':unittest.main()
