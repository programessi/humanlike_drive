"""Safety-relevant regression checks for action-aligned neural memory."""
import tempfile,unittest
from pathlib import Path
import json,numpy as np,torch
from unified_steering.structured.driving_history_intent import HistoryHead,HistoryIntentPolicy,HistoryIntentModel,aligned_labels
from unified_steering.structured.driving_release_memory import ReleaseTransition

class HistoryIntentChecks(unittest.TestCase):
    def test_transition_remains_learned_without_hard_mask(self):
        m=HistoryHead(True)
        with torch.no_grad():m.networks[0][-1].bias[31]=20.;m.networks[4][-1].bias[0]=20.
        x=torch.zeros(2,263);x[0,-32]=1.;x[1,-28]=1.
        self.assertEqual(m(x).argmax(-1).tolist(),[31,0])
        self.assertTrue(torch.isfinite(m(x)).all())

    def test_previous_intention_and_action_history_are_visible(self):
        p=HistoryIntentPolicy();a=torch.zeros(1,152);initial=torch.zeros(1,31);w=torch.zeros(1);prev=torch.zeros(1,42);state=torch.zeros(1,dtype=torch.long)
        before=p.features_history(a,initial,w,w,w,prev,w,state)
        a[:,78]=.3;prev[:,2]=.007;age=torch.tensor([1.25])
        after=p.features_history(a,initial,w,w,w,prev,age,state)
        self.assertEqual(after.shape,(1,263));self.assertAlmostEqual((after-before)[0,78].item(),.3,places=6)
        self.assertAlmostEqual((after-before)[0,190].item(),.14,places=5)
        self.assertEqual(after[0,230].item(),1.25)

    def test_training_label_uses_action_branch_not_post_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'traces').mkdir();(root/'labels').mkdir();(root/'protocol.json').write_text(json.dumps({'shuffle_parameters':{'release_start_deg':5}}))
            actor=np.zeros((4,152),np.float32);actor[:,128]=.1
            path=root/'labels'/'example.npz';np.savez(path,actor=actor)
            np.savez(root/'traces'/'example.npz',expert_action_stage=['clear','clear','relocate','close'],expert_action_mode=['shuffle','shuffle','shuffle','restore'],expert_action_side=[1,1,1,0],expert_stage=['clear','relocate','relocate','unload'])
            state,previous,age=aligned_labels(root,path)
            self.assertEqual(state.tolist(),[8,8,9,23]);self.assertEqual(previous.tolist(),[0,8,8,9]);np.testing.assert_allclose(age,[0,.05,.1,.05])

    def test_release_prior_does_not_prohibit_any_transition(self):
        p=ReleaseTransition();x=torch.zeros(18,206);x[:,-18:]=torch.eye(18)
        with torch.no_grad():p.log_transition_prior.copy_(torch.log(torch.full((18,18),1./18)))
        self.assertTrue(torch.isfinite(p(x)).all())

    def test_hand_anchor_is_independent_of_current_measurement(self):
        p=HistoryIntentPolicy();p.anchored_hand.fill_(True)
        a=torch.zeros(2,152);a[1,14:26]=.7;initial=torch.zeros(2,31);initial[:,-12:]=.6
        with torch.no_grad():p.base.hand_skip.weight.fill_(.1)
        out=p.reference_intent(a,initial,torch.zeros(2),torch.ones(2,dtype=torch.long))
        torch.testing.assert_close(out[:,30:],initial[:,-12:],rtol=0,atol=0)
        holding=p.reference_intent(a,initial,torch.zeros(2),torch.zeros(2,dtype=torch.long))
        self.assertGreater((holding[0,30:]-holding[1,30:]).abs().max().item(),.1)

    def test_relative_feedback_reports_command_error(self):
        p=HistoryIntentPolicy();p.relative_feedback.fill_(True)
        a=torch.zeros(1,152);initial=torch.zeros(1,31);zero=torch.zeros(1);state=torch.zeros(1,dtype=torch.long)
        identity=torch.tensor([1.,0.,0.,1.,0.,0.]);a[:,105:111]=identity;a[:,114:120]=identity
        previous=torch.zeros(1,42);previous[:,3:9]=identity;previous[:,18:24]=identity
        baseline=p.features_history(a,initial,zero,zero,zero,previous,zero,state)
        torch.testing.assert_close(baseline[:,188:230],torch.zeros(1,42),rtol=0,atol=0)
        previous[:,0]=.01
        changed=p.features_history(a,initial,zero,zero,zero,previous,zero,state)
        self.assertAlmostEqual(changed[0,188].item(),.2,places=6)

    def test_reset_and_reload_preserve_episode_boundaries(self):
        p=HistoryIntentPolicy();p.memory=23;p.memory_age=8.;p.previous_intent=torch.ones(42);p.reset()
        self.assertEqual(p.memory,0);self.assertEqual(p.memory_age,0.);self.assertIsNone(p.previous_intent)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';HistoryIntentModel(p).save(path);other=HistoryIntentModel.load(path).policy
            for key,value in p.state_dict().items():torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
