"""Behavior-preserving input migration and explicit recurrent episode boundaries."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np,torch
from unified_steering.structured.evaluate import load_teacher
from unified_steering.structured.driving_policy import normalize_from_dataset
from unified_steering.structured.driving_recurrent import RecurrentPolicy,RecurrentModel

class DrivingPolicyTests(unittest.TestCase):
    def test_normalization_preserves_parent_outputs_and_reload(self):
        torch.set_num_threads(1)
        model=load_teacher(Path('tests/fixtures/legacy_invariants/models/driving_r1_bc_v6/teacher.zip'))
        p=next(Path('tests/fixtures/legacy_invariants/data/hand_actions').glob('*.npz'));z=np.load(p)
        dataset={key:z[key][::3] for key in ['actor','privileged','action']}
        obs={key:torch.from_numpy(dataset[key]) for key in ['actor','privileged']}
        with torch.no_grad():before=model.policy.get_distribution(obs).distribution.mean.clone()
        normalize_from_dataset(model,dataset)
        with torch.no_grad():after=model.policy.get_distribution(obs).distribution.mean
        torch.testing.assert_close(before,after,atol=4e-6,rtol=4e-5)
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d)/'teacher.zip')
            Path(d,'config.json').write_text(json.dumps(dict(policy_variant='normalized-driving-v1')))
            reloaded=load_teacher(Path(d)/'teacher.zip')
            with torch.no_grad():actual=reloaded.policy.get_distribution(obs).distribution.mean
            torch.testing.assert_close(actual,after,atol=0,rtol=0)

    def test_recurrent_initialization_preserves_all_outputs_and_reset(self):
        torch.set_num_threads(1)
        model=load_teacher(Path('tests/fixtures/legacy_invariants/models/driving_r1_norm_v20/teacher.zip'))
        policy=RecurrentPolicy(model.policy).eval()
        z=np.load(next(Path('tests/fixtures/legacy_invariants/data/hand_actions').glob('*.npz')))
        obs={key:z[key][0] for key in ['actor','privileged']}
        before=model.predict(obs,deterministic=True)[0]
        actual=policy.predict(obs)[0]
        np.testing.assert_allclose(actual,before,atol=1e-6)
        self.assertIsNotNone(policy.hidden)
        policy.reset();self.assertIsNone(policy.hidden)
        np.testing.assert_allclose(policy.predict(obs)[0],actual,atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as d:
            wrapper=RecurrentModel(policy);path=Path(d)/'teacher.zip';wrapper.save(path)
            loaded=RecurrentModel.load(path);self.assertIsNone(loaded.policy.hidden)
            np.testing.assert_allclose(loaded.predict(obs)[0],actual,atol=0,rtol=0)
        self.assertEqual(policy.head.out_features,24)
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))

class MemoryPolicyTests(unittest.TestCase):
    def test_memory_stream_reset_and_portable_reload(self):
        from unified_steering.structured.driving_memory import MemoryModel,memory_for
        torch.set_num_threads(1)
        model=load_teacher(Path('tests/fixtures/legacy_invariants/models/driving_r1_memory_hand_v30/teacher.zip'))
        z=np.load(next(Path('tests/fixtures/legacy_invariants/data/hand_actions').glob('*.npz')))
        actor=z['actor'][:30];priv=z['privileged'][:30]
        with torch.no_grad():omega=model.policy.velocity_head(torch.from_numpy(actor[:,122:129])).numpy()[:,0]
        expected=memory_for(actor,omega);actions=[]
        for a,v,row in zip(actor,priv,expected):
            actions.append(model.predict(dict(actor=a,privileged=v))[0])
            self.assertAlmostEqual(model.policy.phase,float(row[0]),places=6)
        model.reset();self.assertIsNone(model.policy.initial);self.assertEqual(model.policy.phase,0.)
        np.testing.assert_array_equal(model.predict(dict(actor=actor[0],privileged=priv[0]))[0],actions[0])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'teacher.zip';model.save(path);loaded=MemoryModel.load(path)
            self.assertTrue(loaded.policy.local_hand)
            for a,v,action in zip(actor,priv,actions):
                np.testing.assert_array_equal(loaded.predict(dict(actor=a,privileged=v))[0],action)
        self.assertEqual(model.policy.head.out_features,24)
        self.assertTrue(all(p.requires_grad for p in model.policy.parameters()))

if __name__=='__main__':unittest.main()
