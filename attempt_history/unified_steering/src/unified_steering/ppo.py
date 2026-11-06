"""Recurrent clipped PPO; equations checked against SB3 2.6.0 (see docs)."""
import time

import numpy as np
import torch


class PPO:
    def __init__(self, actor, critic, config, device):
        self.actor, self.critic = actor, critic
        self.config, self.device = config, device
        if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
            raise ValueError('Actor and critic must not share parameters')
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config['actor_learning_rate'], eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['critic_learning_rate'], eps=1e-5)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device)

    def batch(self, rollout, chunks):
        length = self.config['recurrent_sequence_steps']
        fields = {}
        for key in ('actor','critic','latent','log_prob','starts','advantages','returns'):
            source = getattr(rollout, key)
            fields[key] = self.tensor(np.stack([source[start:start+length, env] for env, start in chunks], axis=1))
        with torch.no_grad():
            fields['hidden'] = torch.cat([self.actor.burn_in(self.tensor(rollout.context(env, start))) for env, start in chunks])
        return fields

    @torch.no_grad()
    def replay_error(self, rollout):
        """Before any update, probabilities must agree with those used in sampling."""
        largest = 0.
        for chunk in rollout.chunks(self.config['recurrent_sequence_steps']):
            batch = self.batch(rollout, [chunk])
            distribution, _ = self.actor.distribution(batch['actor'], batch['hidden'], batch['starts'])
            ratio = (distribution.log_prob(batch['latent'])-batch['log_prob']).exp()
            largest = max(largest, float((ratio-1).abs().max()))
        return largest

    def update(self, rollout):
        begin = time.perf_counter()
        replay_error = self.replay_error(rollout)
        if replay_error > 1e-4:
            raise RuntimeError(f'Old policy probability replay failed: {replay_error}')
        self.actor.train()
        self.critic.train()
        cfg = self.config
        chunks = rollout.chunks(cfg['recurrent_sequence_steps'])
        chunks_per_batch = max(1, cfg['minibatch_steps']//cfg['recurrent_sequence_steps'])
        normalized = (rollout.advantages-rollout.advantages.mean())/(rollout.advantages.std()+1e-8)
        raw_advantages = rollout.advantages
        rollout.advantages = normalized
        log = []
        early_stop = False
        optimizer_steps = 0
        try:
            for epoch in range(cfg['update_epochs']):
                order = np.random.permutation(len(chunks))
                for offset in range(0, len(chunks), chunks_per_batch):
                    selected = [chunks[i] for i in order[offset:offset+chunks_per_batch]]
                    batch = self.batch(rollout, selected)
                    distribution, _ = self.actor.distribution(batch['actor'], batch['hidden'], batch['starts'])
                    log_prob = distribution.log_prob(batch['latent'])
                    log_ratio = log_prob-batch['log_prob']
                    ratio = log_ratio.exp()
                    kl = ((ratio-1)-log_ratio).mean()
                    value = self.critic(batch['critic'])
                    policy_loss = -torch.minimum(ratio*batch['advantages'],
                                                ratio.clamp(1-cfg['clip_range'],1+cfg['clip_range'])*batch['advantages']).mean()
                    entropy = distribution.entropy_sample().mean()
                    actor_loss = policy_loss-cfg['entropy_coefficient']*entropy
                    critic_loss = .5*(value-batch['returns']).square().mean()
                    if not all(torch.isfinite(x).all() for x in (ratio, actor_loss, critic_loss, entropy, kl)):
                        raise FloatingPointError('Nonfinite PPO objective')
                    if float(kl.detach()) > 1.5*cfg['target_kl']:
                        early_stop = True
                        break
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    critic_loss.backward()
                    actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
                    critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    optimizer_steps += 1
                    log.append(dict(policy_loss=float(policy_loss.detach()), value_loss=float(critic_loss.detach()),
                                    entropy=float(entropy.detach()), approx_kl=float(kl.detach()),
                                    clip_fraction=float(((ratio-1).abs()>cfg['clip_range']).float().mean().detach()),
                                    actor_gradient_norm=float(actor_grad), critic_gradient_norm=float(critic_grad)))
                if early_stop:
                    break
        finally:
            rollout.advantages = raw_advantages
        if not log:
            raise RuntimeError('PPO did not take an optimizer step')
        if not all(torch.isfinite(p).all() for module in (self.actor,self.critic) for p in module.parameters()):
            raise FloatingPointError('Nonfinite parameters after optimizer step')
        with torch.no_grad():
            values = self.critic(self.tensor(rollout.critic)).cpu().numpy()
        variance = float(np.var(rollout.returns))
        explained = 1-float(np.var(rollout.returns-values))/variance if variance > 1e-10 else 0.
        return dict(**{key: float(np.mean([row[key] for row in log])) for key in log[0]},
                    optimizer_steps=optimizer_steps, kl_early_stop=early_stop, probability_replay_error=replay_error,
                    explained_variance=explained, update_seconds=time.perf_counter()-begin,
                    arm_std=float(self.actor.log_std[:14].exp().mean().detach()),
                    hand_std=float(self.actor.log_std[14:].exp().mean().detach()))
