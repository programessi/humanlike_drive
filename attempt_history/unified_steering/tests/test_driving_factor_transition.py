import unittest
from pathlib import Path
import numpy as np,torch
from unified_steering.structured.driving_factor_transition import FactorTransition
from unified_steering.structured.driving_history_intent import HistoryIntentPolicy,actual_memory_supervision

class FactorMemoryChecks(unittest.TestCase):
    def test_destination_is_learned_and_every_transition_has_probability(self):
        p=FactorTransition();x=torch.zeros(1,267);x[0,-32+7]=1.
        with torch.no_grad():
            for module in [p.gates[7],p.destinations[7]]:
                for parameter in module.parameters():parameter.zero_()
            p.gates[7][-1].bias[1]=20.;p.destinations[7][-1].bias[31]=20.
        out=p(x);self.assertEqual(out.argmax(-1).item(),31);self.assertTrue(torch.isfinite(out).all());self.assertTrue((out.exp()>0).all())
        with torch.no_grad():p.gates[7][-1].bias[0]=50.
        self.assertEqual(p(x).argmax(-1).item(),7)

    def test_correction_uses_actual_memory_without_teaching_spurious_rollbacks(self):
        expert=np.array([0,7,7,7]);expert_previous=np.array([0,0,7,7]);network=np.array([0,7,8,8])
        state,valid=actual_memory_supervision(expert,expert_previous,network)
        np.testing.assert_array_equal(state,network)
        np.testing.assert_array_equal(valid,[1,1,1,0])

    def test_internal_intent_memory_is_bounded_with_physical_target_limits(self):
        p=HistoryIntentPolicy();p.bounded_intent.fill_(True)
        z=np.load(next(Path('tests/fixtures/legacy_invariants/data/history/labels').glob('*.npz')));a=torch.tensor(z['actor'][:2]);intent=torch.full((2,42),1e24);intent[1]*=-1
        out=p.sanitize_intent(a,intent,torch.ones(2,dtype=torch.long));pose=out[:,:30].reshape(2,2,15)
        self.assertTrue(torch.isfinite(out).all());self.assertLessEqual(out[:,30:].max().item(),1.670001);self.assertGreaterEqual(out[:,30:].min().item(),-.070001)
        self.assertLessEqual(torch.linalg.vector_norm(pose[:,:,1:3],dim=-1).max().item(),.280001)
        self.assertLessEqual(pose[:,:,:1].max().item(),.100001);self.assertGreaterEqual(pose[:,:,:1].min().item(),-.140001)
        self.assertLessEqual(abs(pose[:,:,9:12]).max().item(),.400001);self.assertLessEqual(abs(pose[:,:,12:15]).max().item(),4.000001)

    def test_teacher_labels_remain_unchanged_and_hold_branch_is_exact(self):
        p=HistoryIntentPolicy();p.bounded_intent.fill_(True)
        z=np.load(next(Path('tests/fixtures/legacy_invariants/data/history/labels').glob('*.npz')));a=torch.tensor(z['actor']);target=torch.tensor(np.c_[z['pose'],z['hand_target']]);state=torch.ones(len(a),dtype=torch.long)
        torch.testing.assert_close(p.sanitize_intent(a,target,state),target,rtol=0,atol=2e-7)
        target=target*3.;torch.testing.assert_close(p.sanitize_intent(a,target,state*0),target,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
