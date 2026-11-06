"""Check that demonstration training yields a standalone, trainable policy."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from unified_steering.structured.bootstrap import fit
from unified_steering.structured.demonstrator import Demonstrator,DemonstratorConfig
from unified_steering.structured.env import TaskConfig
from unified_steering.structured.evaluate import load_teacher
from unified_steering.structured.parking import SupportTeacherEnv
from unified_steering.structured.interventions import DiagnosticConfig,DiagnosticEnv


class BootstrapTests(unittest.TestCase):
    def test_expert_only_emits_actions_without_modifying_physics_or_limits(self):
        env=SupportTeacherEnv(TaskConfig(hand_action_mode='leaky',hand_leak_alpha=.995,hand_joint_lead_rad=.15))
        self.addCleanup(env.close)
        obs,_=env.reset(seed=20001,options={'free_side':0,'mode':'sequence','direction':-1})
        expert=Demonstrator(env,DemonstratorConfig(angle_gain=6.))
        q=env.data.qpos.copy();targets=env.controller.target.copy();limits=env.model.actuator_forcerange.copy()
        for _ in range(3):
            action,_=expert.predict(obs)
            self.assertEqual(action.shape,(24,));self.assertTrue(np.isfinite(action).all())
            self.assertLessEqual(abs(action).max(),1.)
        np.testing.assert_array_equal(q,env.data.qpos)
        np.testing.assert_array_equal(targets,env.controller.target)
        np.testing.assert_array_equal(limits,env.model.actuator_forcerange)

    def test_bc_updates_all_mean_channels_and_saves_standalone_policy(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        model=load_teacher(Path('tests/fixtures/legacy_invariants/models/r22_separate_hand_lead_v2/best.zip'))
        env=SupportTeacherEnv(TaskConfig(hand_action_mode='leaky',hand_leak_alpha=.995,hand_joint_lead_rad=.15))
        self.addCleanup(env.close)
        obs,_=env.reset(seed=20001,options={'free_side':0,'mode':'hold'})
        actor=np.repeat(obs['actor'][None],128,axis=0)
        privileged=np.repeat(obs['privileged'][None],128,axis=0)
        tensors={'actor':torch.from_numpy(actor),'privileged':torch.from_numpy(privileged)}
        with torch.no_grad():
            initial=model.policy.get_distribution(tensors).distribution.mean.clone()
        target=initial.numpy()+.05
        critic=[p.detach().clone() for p in model.policy.mlp_extractor.critic.parameters()]
        bias=model.policy.action_net.bias.detach().clone()
        fit(model,{'actor':actor,'privileged':privileged,'action':target},4,
            {'bc_learning_rate':1e-4,'bc_batch_size':128})
        with torch.no_grad():
            final=model.policy.get_distribution(tensors).distribution.mean
        self.assertLess(float((final-torch.from_numpy(target)).square().mean()),.05**2)
        self.assertTrue(torch.all((model.policy.action_net.bias-bias).abs()>1e-9))
        for before,after in zip(critic,model.policy.mlp_extractor.critic.parameters()):
            self.assertTrue(torch.equal(before,after))
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory)/'learned')
            restored=load_teacher(Path(directory)/'learned.zip')
            a=model.predict(obs,deterministic=True)[0]
            b=restored.predict(obs,deterministic=True)[0]
            np.testing.assert_array_equal(a,b)

    def test_v4_diagnostic_reports_actual_hand_lead(self):
        env=DiagnosticEnv(DiagnosticConfig(hand_action_mode='leaky',hand_leak_alpha=.995,hand_joint_lead_rad=.15))
        self.addCleanup(env.close)
        env.reset(seed=20001,options={'free_side':0})
        _,_,_,_,info=env.step(np.zeros(24))
        self.assertEqual(info['audit_hand_lead_rad'],.15)


if __name__=='__main__':unittest.main()
