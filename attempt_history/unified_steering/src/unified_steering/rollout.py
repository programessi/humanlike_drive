"""Time-major on-policy rollouts with explicit timeout bootstrap and GRU context."""
from dataclasses import dataclass
import time

import numpy as np
import torch


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Bootstrap timeouts at final state, but never carry GAE through any reset."""
    advantages = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        carry = delta+gamma*lam*(~(terminated[t] | truncated[t]))*carry
        advantages[t] = carry
    return advantages, advantages+values


@dataclass
class Rollout:
    actor: np.ndarray
    critic: np.ndarray
    latent: np.ndarray
    actions: np.ndarray
    executed_actions: np.ndarray
    log_prob: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    starts: np.ndarray
    prefixes: list
    advantages: np.ndarray
    returns: np.ndarray
    episodes: list
    stats: dict

    def context(self, env, start):
        """Frames before a training chunk, starting at the most recent episode reset."""
        resets = np.flatnonzero(self.starts[:start+1, env])
        if len(resets):
            return self.actor[resets[-1]:start, env]
        return np.concatenate((self.prefixes[env], self.actor[:start, env]), axis=0)

    def chunks(self, length):
        if len(self.actor) % length:
            raise ValueError('Rollout steps must be divisible by sequence length')
        return [(env, start) for env in range(self.actor.shape[1]) for start in range(0, len(self.actor), length)]


class Collector:
    def __init__(self, pool, device):
        self.pool = pool
        self.device = device
        self.count = len(pool.connections)
        self.histories = [[] for _ in range(self.count)]
        self.starts = np.ones(self.count, dtype=bool)

    @torch.no_grad()
    def collect(self, actor, critic, steps, gamma, lam):
        actor.eval()
        critic.eval()
        prefixes = [np.asarray(history, dtype=np.float32).reshape(-1, 152).copy() for history in self.histories]
        hidden = torch.cat([actor.burn_in(torch.as_tensor(context, device=self.device)) for context in prefixes])
        rows = {name: [] for name in ('actor','critic','latent','actions','executed_actions','log_prob','values',
                                     'rewards','next_values','terminated','truncated','starts')}
        episodes = []
        term_sums = {}
        task_steps={name:0 for name in ('hold','positive','negative','roundtrip','unknown')}
        physics_steps, interventions = 0, 0
        begin = time.perf_counter()
        for _ in range(steps):
            obs = self.pool.observations
            actor_obs = torch.as_tensor(obs['actor'], device=self.device)
            starts = torch.as_tensor(self.starts, device=self.device)
            distribution, hidden = actor.distribution(actor_obs.unsqueeze(0), hidden, starts.unsqueeze(0))
            actions, latent, log_prob = distribution.sample()
            values = critic(torch.as_tensor(obs['critic'], device=self.device))
            sampled = actions[0].cpu().numpy()
            _, final_critic, rewards, terminated, truncated, completed, records = self.pool.step(sampled)
            next_values = critic(torch.as_tensor(final_critic, device=self.device)).cpu().numpy()
            payload = dict(actor=obs['actor'], critic=obs['critic'], latent=latent[0].cpu().numpy(),
                           actions=sampled, executed_actions=np.stack([r['executed_action'] for r in records]),
                           log_prob=log_prob[0].cpu().numpy(), values=values.cpu().numpy(), rewards=rewards,
                           next_values=next_values, terminated=terminated, truncated=truncated, starts=self.starts.copy())
            for key, value in payload.items():
                rows[key].append(value.copy())
            for index, record in enumerate(records):
                task=record.get('task','unknown')
                task_steps[task]=task_steps.get(task,0)+1
                if completed[index] is not None:
                    episodes.append(completed[index])
                for key, value in record['reward_terms'].items():
                    term_sums[key] = term_sums.get(key, 0.)+float(value)
                physics_steps += record['physics_steps']
                interventions += record['interventions']
                if terminated[index] or truncated[index]:
                    self.histories[index] = []
                else:
                    self.histories[index].append(obs['actor'][index].copy())
            self.starts = terminated | truncated
        rows = {key: np.stack(value) for key, value in rows.items()}
        for key, value in rows.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f'Invalid rollout field {key}')
        advantages, returns = gae(rows['rewards'], rows['values'], rows['next_values'],
                                   rows['terminated'], rows['truncated'], gamma, lam)
        stats = dict(collection_seconds=time.perf_counter()-begin, physics_steps=physics_steps,
                     intervention_fraction=interventions/max(1, physics_steps),
                     mean_sample_execution_difference=float(np.mean(np.abs(rows['actions']-rows['executed_actions']))),
                     **{f'task_steps_{key}':value for key,value in task_steps.items()},
                     **{f'reward_term_{key}': value/(steps*self.count) for key, value in term_sums.items()})
        return Rollout(**rows, prefixes=prefixes, advantages=advantages, returns=returns, episodes=episodes, stats=stats)
