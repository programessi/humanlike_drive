"""Privileged teacher and history-only student share the action decoder."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import ACTOR_DIM, PRIVILEGED_DIM, ACTION_DIM


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers.extend([nn.Linear(a, b), nn.ELU()])
    return nn.Sequential(*layers)


class TeacherFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, ACTOR_DIM + 16 + PRIVILEGED_DIM)
        self.encoder = mlp([PRIVILEGED_DIM, 128, 64, 16])

    def forward(self, observations):
        o, p = observations["actor"], observations["privileged"]
        return torch.cat([o, self.encoder(p), p], -1)


class AsymmetricNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim_pi = self.latent_dim_vf = 128
        self.actor = mlp([ACTOR_DIM + 16, 256, 256, 128])
        self.critic = mlp([ACTOR_DIM + 16 + PRIVILEGED_DIM, 512, 256, 128])

    def forward_actor(self, features):
        return self.actor(features[:, :ACTOR_DIM + 16])

    def forward_critic(self, features):
        return self.critic(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class TeacherPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = TeacherFeatures
        kwargs["net_arch"] = []
        kwargs["activation_fn"] = nn.ELU
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.log_std[:12] = float(np.log(.25))
            self.log_std[12:] = float(np.log(.045))

    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricNetworks()


class TeacherActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder = deepcopy(policy.features_extractor.encoder)
        self.decoder = deepcopy(policy.mlp_extractor.actor)
        self.head = deepcopy(policy.action_net)

    def latent(self, privileged):
        return self.encoder(privileged)

    def forward(self, observation, privileged):
        z = self.encoder(privileged)
        return self.head(self.decoder(torch.cat([observation, z], -1))).clamp(-1., 1.)


class HistoryStudent(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.history_encoder = mlp([ACTOR_DIM, 256, 128])
        self.gru = nn.GRUCell(128, 128)
        self.latent_head = nn.Linear(128, 16)
        self.decoder = deepcopy(teacher.decoder)
        self.head = deepcopy(teacher.head)

    @torch.jit.export
    def with_latent(self, observation, hidden, episode_start):
        hidden = hidden * (1 - episode_start.reshape(-1, 1))
        hidden = self.gru(self.history_encoder(observation), hidden)
        z = self.latent_head(hidden)
        action = self.head(self.decoder(torch.cat([observation, z], -1))).clamp(-1., 1.)
        return action, hidden, z

    def forward(self, observation, hidden, episode_start):
        action, hidden, _ = self.with_latent(observation, hidden, episode_start)
        return action, hidden
