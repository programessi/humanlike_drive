"""Checks the learned memory is data driven and preserves the R1 anchor."""
import tempfile,unittest
from pathlib import Path
import numpy as np,torch
from unified_steering.structured.driving_conditional_memory import ConditionalTransition,ConditionalMemoryPolicy,ConditionalMemoryModel
from unified_steering.structured.evaluate import load_teacher

class ConditionalMemoryChecks(unittest.TestCase):
    def test_transition_is_learned_and_conditioned_on_previous_memory(self):
        m=ConditionalTransition()
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            bias=m.net[-1].bias.reshape(15,15)
            bias[0,14]=10.;bias[1,3]=10.
        x=torch.zeros(2,203);x[0,-15]=1.;x[1,-14]=1.
        self.assertEqual(m(x).argmax(-1).tolist(),[14,3])

    def test_initial_geometry_uses_qualified_baseline(self):
        p=ConditionalMemoryPolicy();source=load_teacher(Path('tests/fixtures/legacy_invariants/models/driving_r1_pose_v41/teacher.zip')).policy;p.base=source
        z=np.load('tests/fixtures/legacy_invariants/data/halfturn/a180_d+1_short_s48001.npz');a=torch.tensor(z['actor'][:5]);initial=torch.cat([a[:1,102:121],a[:1,14:26]+.15*a[:1,66:78]],-1).expand(5,-1)
        phase=torch.zeros(5);omega=torch.zeros(5);variation=torch.zeros(5)
        with torch.no_grad():
            for module in p.intent_net.experts:
                module[-1].bias.fill_(.8)
            intent=p.intention_for_state(a,initial,phase,omega,variation,torch.zeros(5,dtype=torch.long))
            baseline=p.base_intent(a,initial,phase)
        torch.testing.assert_close(intent[:,:30],baseline[:,:30],rtol=0,atol=0)
        self.assertGreater((intent[:,30:]-baseline[:,30:]).abs().max().item(),.1)

    def test_full_feedback_reads_hand_and_command_state(self):
        p=ConditionalMemoryPolicy(full_feedback=True);a=torch.zeros(1,152);initial=torch.zeros(1,31);phase=torch.zeros(1);state=torch.zeros(1,dtype=torch.long)
        before=p.intent_features(a,initial,phase,phase,phase,state)
        a[0,14]=.3;a[0,66]=.4;a[0,140]=.5
        after=p.intent_features(a,initial,phase,phase,phase,state)
        self.assertEqual(after.shape,(1,119))
        self.assertEqual(torch.count_nonzero(after-before).item(),3)
        self.assertAlmostEqual(after[0,69].item(),.3,places=6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.pt';ConditionalMemoryModel(p).save(path);reloaded=ConditionalMemoryModel.load(path)
        self.assertEqual(reloaded.policy.intent_mean.numel(),119)

    def test_reload_has_no_runtime_expert_dependency(self):
        m=load_teacher(Path('tests/fixtures/legacy_invariants/models/driving_r2_conditional_memory_v58/teacher.zip'));z=np.load('tests/fixtures/legacy_invariants/data/halfturn/a180_d-1_short_s48001.npz');obs={'actor':z['actor'][0]}
        m.reset();expected=m.predict(obs)[0]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt';m.save(path);other=ConditionalMemoryModel.load(path);other.reset();actual=other.predict(obs)[0]
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual.shape,(24,));self.assertTrue(np.isfinite(actual).all())
if __name__=='__main__':unittest.main()
