"""SB3 PPO policy with structurally separate actor and privileged critic inputs."""
from __future__ import annotations

import numpy as np
from copy import deepcopy
import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActorFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, int(np.prod(observation_space["actor"].shape)))

    def forward(self, observations):
        return observations["actor"].flatten(start_dim=1)


class CriticFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        width = int(np.prod(observation_space["actor"].shape) + np.prod(observation_space["privileged"].shape))
        super().__init__(observation_space, width)

    def forward(self, observations):
        return torch.cat([observations["actor"].flatten(start_dim=1), observations["privileged"]], dim=1)


class AsymmetricMLP(nn.Module):
    def __init__(self, actor_dim, critic_dim, actor_layers, critic_layers, activation, device):
        super().__init__()
        def network(width, layers):
            modules = []
            for next_width in layers:
                modules.extend([nn.Linear(width, next_width), activation()])
                width = next_width
            return nn.Sequential(*modules).to(device)
        self.policy_net = network(actor_dim, actor_layers)
        self.value_net = network(critic_dim, critic_layers)
        self.latent_dim_pi, self.latent_dim_vf = actor_layers[-1], critic_layers[-1]

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        actor, critic = features
        return self.forward_actor(actor), self.forward_critic(critic)


class AsymmetricPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = ActorFeatures
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.pi_features_extractor = ActorFeatures(self.observation_space)
        self.vf_features_extractor = CriticFeatures(self.observation_space)
        self.features_extractor = self.pi_features_extractor
        architecture = self.net_arch if isinstance(self.net_arch, dict) else {}
        self.mlp_extractor = AsymmetricMLP(
            self.pi_features_extractor.features_dim, self.vf_features_extractor.features_dim,
            architecture.get("pi", [256, 256, 128]), architecture.get("vf", [512, 512, 256]),
            self.activation_fn, self.device)


class DeploymentActor(nn.Module):
    """Only actor history enters this graph. No critic or contact input exists."""
    def __init__(self, policy):
        super().__init__()
        self.network = policy.mlp_extractor.policy_net
        self.action = policy.action_net

    def forward(self, history):
        return self.action(self.network(history.flatten(start_dim=1))).clamp(-1, 1)


def export_actor(model, path):
    actor = deepcopy(DeploymentActor(model.policy)).eval().to("cpu")
    shape = model.observation_space["actor"].shape
    example = torch.zeros((1, *shape), dtype=torch.float32)
    torch.jit.trace(actor, example).save(str(path))


def load_policy(checkpoint, **kwargs):
    """Load preserved checkpoints after package reorganization without rewriting weights.

    Historical SB3 archives name the old policy module. Override only the class
    location; architecture, tensors and optimizer state still come from the file.
    Callers must validate task metadata and observation/action spaces.
    """
    from stable_baselines3 import PPO
    return PPO.load(checkpoint, custom_objects={"policy_class": AsymmetricPolicy}, **kwargs)
