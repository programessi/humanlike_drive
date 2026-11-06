"""Causal/read-only instrumentation and development selection regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from unified_steering.audit import ROOT
from unified_steering.env import SteeringEnv
from unified_steering.evaluate import evaluate
from unified_steering.models import Actor,Critic
from unified_steering.monitoring import DevelopmentMonitor,selection_score
from unified_steering.ppo import PPO
from unified_steering.snapshots import capture,restore
from unified_steering.train import resolved_config
from unified_steering.tracing import TraceRecorder,first_sustained


class MonitoringTests(unittest.TestCase):
    def test_sustained_contact_proxy_requires_full_window(self):
        times=np.arange(1,101)*.002
        mask=np.zeros(100,dtype=bool)
        mask[20:69]=True
        self.assertIsNone(first_sustained(times,mask,.1,.002))
        mask[69]=True
        self.assertAlmostEqual(first_sustained(times,mask,.1,.002),.042)

    def test_periodic_evaluation_preserves_rng_and_holding_controls_best_selection(self):
        torch.set_num_threads(1)
        config=resolved_config(ROOT/'configs/m30_monitor_smoke.yaml')
        actor=Actor(config['actor'],np.zeros(26,dtype=np.float32))
        critic=Critic(config['critic'])
        algorithm=PPO(actor,critic,config['ppo'],'cpu')
        reports=[]
        for hold,motion in ((.2,4.),(1.,1.),(.3,.8)):
            cases=[dict(name='neutral_hold',rmse_deg=hold,hold_3s_drift_deg=hold),
                   dict(name='positive',rmse_deg=motion),dict(name='negative',rmse_deg=motion+.1)]
            reports.append(dict(cases=cases,all_cases_complete=True,tracking_success_rate=.2,
                                safety_completion_rate=1.))
        def fake_evaluate(cfg,module,path,**kwargs):
            module.eval()
            torch.randn(50)
            np.random.normal(size=50)
            path.mkdir(parents=True)
            return reports.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            monitor=DevelopmentMonitor(Path(directory),config,actor,critic,algorithm,np.zeros(26),evaluator=fake_evaluate)
            actor.train()
            torch.manual_seed(91);np.random.seed(91)
            expected_torch=torch.randn(5);expected_numpy=np.random.normal(size=5)
            torch.manual_seed(91);np.random.seed(91)
            first=monitor.check(0,0,initial=True)
            torch.testing.assert_close(torch.randn(5),expected_torch,atol=0,rtol=0)
            np.testing.assert_array_equal(np.random.normal(size=5),expected_numpy)
            self.assertTrue(actor.training)
            self.assertTrue(first['selected_as_best'])
            second=monitor.check(512,1)
            self.assertFalse(second['eligible'])  # Better movement cannot hide hold regression.
            third=monitor.check(1024,2)
            self.assertTrue(third['selected_as_best'])
            self.assertEqual(monitor.best_step,1024)
            for name in ('step_000000000.pt','step_000000512.pt','step_000001024.pt','best.pt','development_index.json'):
                self.assertTrue((Path(directory)/name).exists(),name)


@unittest.skipUnless((ROOT/'assets/scene.xml').exists() and
                     (ROOT/'assets/initial_states/m1_both_v1/manifest.json').exists(),'Requires M1 assets')
class DiagnosticPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=resolved_config(ROOT/'configs/m2_smoke.yaml')
        cls.config['environment']['diagnostics']=True
        cls.env=SteeringEnv(copy.deepcopy(cls.config['environment']))

    def setUp(self):
        self.env.reset(seed=7,options={'support':'both'})

    def test_instrumentation_does_not_change_any_physics_observation_or_reward(self):
        ordinary=copy.deepcopy(self.config['environment'])
        ordinary['diagnostics']=False
        plain=SteeringEnv(ordinary)
        plain.reset(seed=7,options={'support':'both'})
        rng=np.random.default_rng(7)
        for _ in range(30):
            action=plain.hold_action().copy()
            action[:14]=rng.normal(0,.015,14)
            obs_a,reward_a,term_a,trunc_a,info_a=plain.step(action)
            obs_b,reward_b,term_b,trunc_b,info_b=self.env.step(action)
            np.testing.assert_array_equal(plain.sim.data.qpos,self.env.sim.data.qpos)
            np.testing.assert_array_equal(plain.sim.data.qvel,self.env.sim.data.qvel)
            np.testing.assert_array_equal(plain.sim.data.ctrl,self.env.sim.data.ctrl)
            for key in ('actor','critic'):
                np.testing.assert_array_equal(obs_a[key],obs_b[key])
            self.assertEqual((reward_a,term_a,trunc_a),(reward_b,term_b,trunc_b))
            self.assertEqual(info_a['execution_interventions'],info_b['execution_interventions'])
            self.assertEqual(info_b['diagnostics']['physics'].shape,(25,23))

    def test_projection_reasons_and_consecutive_runs_are_per_joint(self):
        env=self.env;e=env.executor
        q=env.sim.data.qpos[env.sim.qidx[:14]].copy()
        e.reset(np.r_[q,e.target[14:]])
        e.target[0]=q[0]+e.arm_lead-.0001
        action=e.hold_action()
        action[0]=1
        action[14:]=1
        e.begin(action)
        for _ in range(6):
            self.assertTrue(e.tick(q,env.dt))
        self.assertEqual(e.step_counts[0,0],6)
        self.assertEqual(e.longest_runs[0,0],6)
        self.assertEqual(e.step_counts[1,0],0)
        self.assertTrue(np.all(e.step_counts[3,14:]>0))
        self.assertTrue(np.all(e.step_counts[3,:14]==0))

    def test_snapshot_replays_identical_actions_bit_exactly(self):
        env=self.env
        action=env.hold_action();action[0]=.01
        for _ in range(10):
            env.step(action)
        snapshot=capture(env)
        histories=[]
        for repeat in range(2):
            restore(env,snapshot)
            rows=[]
            for _ in range(10):
                obs,reward,term,trunc,info=env.step(action)
                rows.append(np.r_[env.sim.data.qpos,env.sim.data.qvel,obs['actor'],reward])
            histories.append(rows)
        np.testing.assert_array_equal(*histories)

    def test_trace_is_loadable_without_pickle_and_matches_substep_count(self):
        env=self.env
        record=TraceRecorder(env,dict(test=True))
        old=env.last_obs['actor'].copy();action=env.hold_action()
        _,_,_,_,info=env.step(action)
        record.record(env,info,old,np.zeros(26),np.zeros(26),np.zeros(26),action)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.npz'
            result=record.save(path)
            with np.load(path,allow_pickle=False) as values:
                self.assertEqual(values['qpos'].shape,(1,39))
                self.assertEqual(values['physics'].shape,(25,23))
                self.assertEqual(json.loads(str(values['metadata_json']))['schema'],'steering-trace-v2')
            self.assertEqual(result['physics_steps'],25)

    def test_early_failure_does_not_invent_final_hold_metrics(self):
        class EarlyStopEnv(SteeringEnv):
            def step(self,action):
                obs,reward,term,trunc,info=super().step(action)
                info['termination_reason']='injected_early_failure'
                self.done=True
                return obs,reward,True,False,info
        with tempfile.TemporaryDirectory() as folder,patch('unified_steering.evaluate.SteeringEnv',EarlyStopEnv):
            result=evaluate(self.config,None,Path(folder)/'early')
            self.assertFalse(result['all_cases_complete'])
            for case in result['cases']:
                self.assertIsNone(case['hold_3s_drift_deg'])
                self.assertIsNone(case['hold_3s_peak_target_error_deg'])
                self.assertFalse(case['tracking_success'])


if __name__=='__main__':
    unittest.main()
