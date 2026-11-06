"""Small recurrent actor and asymmetric value network, with explicit GRU state."""
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from . import ACTOR_DIM, ACTION_DIM, HIDDEN_DIM


def mlp(sizes):
    layers = []
    for left, right in zip(sizes[:-1], sizes[1:]):
        layer = nn.Linear(left, right)
        nn.init.orthogonal_(layer.weight, 2 ** .5)
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.ELU()])
    return nn.Sequential(*layers)


class RecurrentActor(nn.Module):
    def __init__(self, log_std_init=-2.5):
        super().__init__()
        self.encoder = mlp([ACTOR_DIM, 256, 128])
        self.gru = nn.GRUCell(128, HIDDEN_DIM)
        self.mean = nn.Linear(HIDDEN_DIM, ACTION_DIM)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(log_std_init)))

    def forward(self, observation, hidden, episode_start):
        hidden = hidden * (1. - episode_start.reshape(-1, 1))
        hidden = self.gru(self.encoder(observation), hidden)
        return self.mean(hidden), hidden

    def distribution(self, observation, hidden, episode_start):
        mean, hidden = self(observation, hidden, episode_start)
        return Normal(mean, self.log_std.clamp(-5., 0.).exp().expand_as(mean)), hidden


def squashed_log_prob(distribution, latent):
    action = torch.tanh(latent)
    return (distribution.log_prob(latent) - torch.log(1 - action.square() + 1e-6)).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, log_std_init=-2.5):
        super().__init__()
        self.actor = RecurrentActor(log_std_init)
        self.critic = nn.Sequential(mlp([ACTOR_DIM + 54, 512, 256, 128]), nn.Linear(128, 1))

    def value(self, actor_observation, privileged):
        return self.critic(torch.cat([actor_observation, privileged], -1)).squeeze(-1)


class DeploymentActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.encoder = deepcopy(actor.encoder)
        self.gru = deepcopy(actor.gru)
        self.mean = deepcopy(actor.mean)

    def forward(self, observation, hidden, episode_start):
        hidden = hidden * (1. - episode_start.reshape(-1, 1))
        hidden = self.gru(self.encoder(observation), hidden)
        return torch.tanh(self.mean(hidden)), hidden


def export_actor(network, path):
    actor = DeploymentActor(network.actor).cpu().eval()
    torch.jit.script(actor).save(str(path))


def generalized_advantages(rewards, values, next_values, dones, gamma=.995, lam=.95):
    """Timeouts bootstrap value but stop GAE propagation across episode resets."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_values[index] - values[index]
        carry = delta + gamma * lam * (1 - dones[index]) * carry
        advantages[index] = carry
    return advantages, advantages + values
