import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from unified_steering.audit import ROOT
from unified_steering.env import SteeringEnv
from unified_steering.evaluate_commands import definitions,evaluate_commands,command_selection_score
from unified_steering.models import Actor,Critic
from unified_steering.ppo import PPO
from unified_steering.tensorboard_logging import TrainingDashboard
from unified_steering.train import resolved_config
from unified_steering.workers import TaskSchedule,reset_task


class CommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=resolved_config(ROOT/'configs/m32_single_direction.yaml')

    def test_balanced_blocks_and_single_direction_reference_horizon(self):
        rng=np.random.default_rng(7);settings=self.config['tasks'];schedule=TaskSchedule(rng,settings)
        for _ in range(3):
            block=[schedule.next() for _ in range(10)]
            self.assertEqual({k:block.count(k) for k in settings['balanced_episode_counts']},settings['balanced_episode_counts'])
        env=SteeringEnv(copy.deepcopy(self.config['environment']))
        for task,sign in (('positive',1),('negative',-1),('hold',0)):
            obs,info=reset_task(env,rng,settings,seed=7,task_kind=task)
            self.assertEqual(info['direction'],sign)
            self.assertEqual(int(np.sign(env.reference.final-env.reference.initial)),sign)
            if sign:
                self.assertGreaterEqual(env.config['episode_seconds']+1e-8,env.reference.duration+3.)
                velocities=[env.reference.sample(t)[1] for t in np.linspace(0,env.reference.duration,100)]
                self.assertTrue(all(sign*v>=-1e-12 for v in velocities))
            else:self.assertEqual(env.config['episode_seconds'],10.)
            self.assertEqual(obs['actor'].shape,(152,))

    def test_grid_is_matched_by_sign_and_has_twenty_distinct_conditions(self):
        grid=definitions(self.config['evaluation'])
        self.assertEqual(len(grid),20)
        self.assertEqual(len({r['name'] for r in grid}),20)
        self.assertEqual([(r['amplitude'],r['speed']) for r in grid if r['sign']==1],
                         [(r['amplitude'],r['speed']) for r in grid if r['sign']==-1])

    def test_model_selection_protects_worse_direction_and_hold(self):
        def report(positive,negative,hold=.2):
            return dict(cases=[dict(name='hold_3s',sign=0,rmse_deg=hold)],all_cases_complete=True,
                        hold_worst_peak_deg=hold,directions={
                'positive':dict(tracking_success_rate=0.,baseline_improvement=positive,worst_rmse_deg=2.,mean_rmse_deg=1.),
                'negative':dict(tracking_success_rate=0.,baseline_improvement=negative,worst_rmse_deg=2.,mean_rmse_deg=1.)})
        anchor=report(0.,0.)
        eligible,base=command_selection_score(anchor,anchor)
        self.assertTrue(eligible)
        self.assertLess(command_selection_score(report(.5,-.1),anchor)[1],base)
        self.assertGreater(command_selection_score(report(.1,.1),anchor)[1],base)
        self.assertFalse(command_selection_score(report(.5,.5,hold=1.1),anchor)[0])

    def test_static_targets_cannot_qualify_as_direction_learning(self):
        cfg=copy.deepcopy(self.config)
        cfg['evaluation'].update(hold_seconds=[3.],amplitude_deg=[1.],speed_deg_s=[1.])
        with tempfile.TemporaryDirectory() as directory:
            report=evaluate_commands(cfg,None,Path(directory)/'fixed')
            self.assertTrue(report['all_cases_complete'])
            self.assertFalse(report['development_qualified'])
            self.assertEqual(report['directions']['positive']['direction_correct_rate'],0.)
            self.assertEqual(report['directions']['negative']['direction_correct_rate'],0.)

    def test_tensorboard_writes_scalars_and_parameter_histograms(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        config=copy.deepcopy(self.config);config['tensorboard']['histogram_every_updates']=1
        actor=Actor(config['actor'],np.zeros(26,dtype=np.float32));critic=Critic(config['critic'])
        algorithm=PPO(actor,critic,config['ppo'],'cpu')
        with tempfile.TemporaryDirectory() as directory:
            dashboard=TrainingDashboard(Path(directory),config)
            dashboard.update(dict(steps=512,update=1,approx_kl=.001),actor,critic,algorithm,[])
            dashboard.close()
            events=EventAccumulator(str(Path(directory)/'tensorboard')).Reload()
            self.assertIn('training/approx_kl',events.Tags()['scalars'])
            self.assertIn('gradients/reference_encoder',events.Tags()['scalars'])
            self.assertTrue(events.Tags()['histograms'])


if __name__=='__main__':unittest.main()
