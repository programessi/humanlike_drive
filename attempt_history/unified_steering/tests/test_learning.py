"""Behavioral tests for recurrent probabilities, GAE, information flow and resume."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Independent, Normal, TransformedDistribution, TanhTransform

from unified_steering.audit import ROOT
from unified_steering.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from unified_steering.distributions import TanhNormal
from unified_steering.models import Actor, Critic
from unified_steering.ppo import PPO
from unified_steering.rollout import Rollout, gae
from unified_steering.train import resolved_config
from unified_steering.workers import WorkerPool


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = resolved_config(ROOT/'configs/m2_smoke.yaml')

    def setUp(self):
        torch.manual_seed(17)
        np.random.seed(17)
        self.actor = Actor(self.config['actor'], np.zeros(26,dtype=np.float32))
        self.critic = Critic(self.config['critic'])

    def test_tanh_density_matches_torch_and_extremes_have_finite_gradients(self):
        mean = torch.randn(3,26,requires_grad=True)
        log_std = torch.full((26,),-.6,requires_grad=True)
        own = TanhNormal(mean,log_std)
        latent = mean+torch.randn_like(mean)*.2
        reference = Independent(TransformedDistribution(Normal(mean,log_std.exp()),[TanhTransform()]),1)
        torch.testing.assert_close(own.log_prob(latent),reference.log_prob(latent.tanh()),atol=2e-4,rtol=2e-5)
        extreme = torch.full_like(mean,30.)
        loss = own.log_prob(extreme).mean()+own.entropy_sample().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(log_std.grad).all())

    def test_gae_terminal_timeout_and_reset_are_different(self):
        # First step times out: bootstrap final value 10, never use reset value 100.
        reward = np.array([[1.],[2.]],dtype=np.float32)
        value = np.array([[3.],[100.]],dtype=np.float32)
        next_value = np.array([[10.],[99.]],dtype=np.float32)
        term = np.array([[False],[True]])
        trunc = np.array([[True],[False]])
        advantage, returns = gae(reward,value,next_value,term,trunc,.9,.95)
        np.testing.assert_allclose(advantage,[[7.],[-98.]],atol=1e-6)
        np.testing.assert_allclose(returns,[[10.],[2.]],atol=1e-6)
        term[0] = True
        trunc[0] = False
        advantage,_ = gae(reward,value,next_value,term,trunc,.9,.95)
        self.assertEqual(float(advantage[0,0]),-2.)

    def test_gae_continuing_rollout_bootstraps_last_state(self):
        reward = np.array([[1.],[2.]],dtype=np.float32)
        value = np.array([[3.],[4.]],dtype=np.float32)
        nxt = np.array([[4.],[5.]],dtype=np.float32)
        zeros = np.zeros((2,1),dtype=bool)
        advantage,_ = gae(reward,value,nxt,zeros,zeros,.9,.95)
        np.testing.assert_allclose(advantage,[[1+.9*4-3+.9*.95*(2+.9*5-4)],[2+.9*5-4]],rtol=1e-6)

    def test_sequence_matches_steps_resets_and_burnin(self):
        obs = torch.randn(12,2,152)*.2
        starts = torch.zeros(12,2,dtype=torch.bool)
        starts[0] = True
        starts[5,0] = True
        hidden = self.actor.initial_hidden(2)
        whole, final = self.actor(obs,hidden,starts)
        outputs = []
        for t in range(12):
            output,hidden = self.actor(obs[t:t+1],hidden,starts[t:t+1])
            outputs.append(output)
        torch.testing.assert_close(whole,torch.cat(outputs),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(final,hidden)
        reset_output,_ = self.actor(obs[5:,0:1],self.actor.initial_hidden(1),starts[5:,0:1])
        torch.testing.assert_close(reset_output,whole[5:,0:1],atol=1e-6,rtol=1e-5)
        burned = self.actor.burn_in(obs[:5,1])
        _,expected = self.actor(obs[:5,1:2],self.actor.initial_hidden(1),starts[:5,1:2])
        torch.testing.assert_close(burned,expected)
        # Recompute history under changed weights, rather than reuse old memory.
        with torch.no_grad():
            self.actor.gru.bias_ih_l0.add_(.1)
        self.assertGreater(float((self.actor.burn_in(obs[:5,1])-burned).abs().max()),.01)

    def test_actor_has_no_critic_dependency_and_rejects_privileged_input(self):
        obs = torch.randn(1,1,152)
        hidden = self.actor.initial_hidden(1)
        starts = torch.ones(1,1,dtype=torch.bool)
        before,_ = self.actor(obs,hidden,starts)
        with torch.no_grad():
            for parameter in self.critic.parameters():
                parameter.add_(100.)
        after,_ = self.actor(obs,hidden,starts)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        self.assertFalse({id(p) for p in self.actor.parameters()} & {id(p) for p in self.critic.parameters()})
        with self.assertRaises(ValueError):
            self.actor(torch.zeros(1,1,172),hidden,starts)

    def synthetic_rollout(self):
        length,count = 16,2
        obs = np.random.normal(0,.2,(length,count,152)).astype(np.float32)
        priv = np.concatenate((obs,np.zeros((length,count,20),dtype=np.float32)),axis=-1)
        starts = np.zeros((length,count),dtype=bool)
        starts[0] = True
        starts[7,0] = True
        with torch.no_grad():
            distribution,_ = self.actor.distribution(torch.tensor(obs),self.actor.initial_hidden(count),torch.tensor(starts))
            action,latent,logp = distribution.sample()
            value = self.critic(torch.tensor(priv)).numpy()
        rewards = np.random.normal(0,1,(length,count)).astype(np.float32)
        terminated = np.zeros_like(starts)
        terminated[6,0] = True
        truncated = np.zeros_like(starts)
        adv,returns = gae(rewards,value,value,terminated,truncated,.995,.97)
        return Rollout(actor=obs,critic=priv,latent=latent.numpy(),actions=action.numpy(),
                       executed_actions=np.zeros_like(action.numpy()),log_prob=logp.numpy(),values=value,rewards=rewards,
                       next_values=value.copy(),terminated=terminated,truncated=truncated,starts=starts,
                       prefixes=[np.zeros((0,152),dtype=np.float32)]*count,advantages=adv,returns=returns,episodes=[],stats={})

    def test_replay_and_ppo_update_all_branches_uses_sampled_actions(self):
        cfg = copy.deepcopy(self.config['ppo'])
        cfg.update(recurrent_sequence_steps=4,minibatch_steps=16,update_epochs=2)
        algorithm = PPO(self.actor,self.critic,cfg,'cpu')
        rollout = self.synthetic_rollout()
        self.assertLess(algorithm.replay_error(rollout),1e-5)
        # Altering execution records cannot change the action probabilities.
        rollout.executed_actions[:] = 123.
        self.assertLess(algorithm.replay_error(rollout),1e-5)
        before = {name:p.detach().clone() for name,p in self.actor.named_parameters()}
        stats = algorithm.update(rollout)
        self.assertGreater(stats['optimizer_steps'],0)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        for module in ('state_encoder','reference_encoder','gru','action_head','log_std'):
            change = sum(float((p-before[name]).square().sum()) for name,p in self.actor.named_parameters() if name.startswith(module))
            self.assertGreater(change,0.,module)
        expected_start = rollout.actor[7:8,0]
        np.testing.assert_array_equal(rollout.context(0,8),expected_start)

    def test_checkpoint_restores_optimizers_rng_scaling_and_inference(self):
        cfg = copy.deepcopy(self.config['ppo'])
        cfg.update(recurrent_sequence_steps=4,minibatch_steps=16,update_epochs=1)
        algorithm = PPO(self.actor,self.critic,cfg,'cpu')
        algorithm.update(self.synthetic_rollout())
        obs = torch.randn(1,1,152)
        starts = torch.ones(1,1,dtype=torch.bool)
        expected,_ = self.actor(obs,self.actor.initial_hidden(1),starts)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.pt'
            save_checkpoint(path,self.actor,self.critic,algorithm,self.config,np.zeros(26),512,1)
            expected_rng = torch.randn(10)
            expected_numpy = np.random.normal(size=10)
            actor = Actor(self.config['actor'],np.zeros(26))
            critic = Critic(self.config['critic'])
            restored = PPO(actor,critic,cfg,'cpu')
            payload = read_checkpoint(path)
            restore_checkpoint(payload,actor,critic,restored,restore_rng=True)
            actual,_ = actor(obs,actor.initial_hidden(1),starts)
            torch.testing.assert_close(expected,actual,atol=0,rtol=0)
            torch.testing.assert_close(torch.randn(10),expected_rng,atol=0,rtol=0)
            np.testing.assert_array_equal(np.random.normal(size=10),expected_numpy)
            self.assertEqual(payload['steps'],512)
            for saved,loaded in ((algorithm.actor_optimizer,restored.actor_optimizer),
                                 (algorithm.critic_optimizer,restored.critic_optimizer)):
                self.assertEqual(saved.state_dict()['param_groups'],loaded.state_dict()['param_groups'])
                for key,state in saved.state_dict()['state'].items():
                    for field,value in state.items():
                        torch.testing.assert_close(value,loaded.state_dict()['state'][key][field],atol=0,rtol=0)


@unittest.skipUnless((ROOT/'assets/scene.xml').exists() and
                     (ROOT/'assets/initial_states/m1_both_v1/manifest.json').exists(), 'Requires M1 physics assets')
class WorkerIntegrationTests(unittest.TestCase):
    def test_timeout_returns_final_observation_separate_from_reset(self):
        config = resolved_config(ROOT/'configs/m2_smoke.yaml')
        environment = copy.deepcopy(config['environment'])
        environment['episode_seconds'] = .05
        pool = WorkerPool(environment,dict(hold_fraction=1.,amplitude_deg=5.),2,17)
        try:
            initial = pool.observations['actor'].copy()
            actions = np.tile(pool.initial_action,(2,1))
            obs, final, reward, term, trunc, episodes, records = pool.step(actions)
            self.assertTrue(np.all(trunc))
            self.assertFalse(np.any(term))
            np.testing.assert_array_equal(obs['actor'],initial)
            self.assertTrue(np.all(np.abs(final[:,134]-obs['actor'][:,134]) > 1e-8))
            self.assertTrue(np.isfinite(reward).all())
            for episode,record in zip(episodes,records):
                self.assertEqual(episode['length'],1)
                self.assertFalse(episode['terminated'])
                self.assertEqual(record['physics_steps'],25)
                json.dumps(episode,allow_nan=False)
        finally:
            pool.close()


if __name__ == '__main__':
    unittest.main()
