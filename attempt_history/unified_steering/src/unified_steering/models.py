"""Asymmetric actor/critic with no shared features or privileged actor input."""
import numpy as np
import torch
from torch import nn

from .contract import ACTION_DIM, OBSERVATION_DIM, CRITIC_DIM, PREVIEW_INDICES
from .distributions import TanhNormal


def mlp(sizes, output_gain=None):
    layers = []
    for index, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = nn.Linear(a, b)
        last = index == len(sizes)-2
        nn.init.orthogonal_(layer.weight, gain=output_gain if last and output_gain is not None else np.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        if not last or output_gain is None:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, config, initial_action):
        super().__init__()
        self.hidden_size = config['gru_hidden']
        self.state_encoder = mlp(config['state_encoder'])
        self.reference_encoder = mlp(config['reference_encoder'])
        self.gru = nn.GRU(config['state_encoder'][-1], self.hidden_size)
        for name, parameter in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(parameter)
            else:
                nn.init.orthogonal_(parameter)
        self.action_head = mlp(config['action_head'], output_gain=.01)
        action = torch.as_tensor(initial_action, dtype=torch.float32)
        if action.shape != (ACTION_DIM,) or torch.any(action.abs() >= 1):
            raise ValueError('Initial deterministic action must be strictly inside (-1,1)')
        with torch.no_grad():
            self.action_head[-1].bias.copy_(torch.atanh(action))
        std = [config['initial_arm_std']]*14+[config['initial_hand_std']]*12
        self.log_std = nn.Parameter(torch.tensor(std, dtype=torch.float32).log())
        if config['state_encoder'][0] != 140 or config['reference_encoder'][0] != 12:
            raise ValueError('Encoder inputs must match interface v1')
        if config['action_head'][0] != self.hidden_size+config['reference_encoder'][-1] or config['action_head'][-1] != 26:
            raise ValueError('Invalid actor head dimensions')

    def initial_hidden(self, batch):
        return self.log_std.new_zeros(batch, self.hidden_size)

    @staticmethod
    def split(observation):
        if observation.shape[-1] != OBSERVATION_DIM:
            raise ValueError('Actor accepts only 152 deployable inputs')
        reference = observation[..., PREVIEW_INDICES]
        state = torch.cat((observation[..., :PREVIEW_INDICES.start], observation[..., PREVIEW_INDICES.stop:]), dim=-1)
        return state, reference

    def forward(self, observation, hidden, episode_start):
        """Time-major [T,B,152]; reset memory before processing each start frame."""
        if observation.ndim != 3 or hidden.shape != (observation.shape[1], self.hidden_size):
            raise ValueError('Expected time-major observations and [B,H] hidden state')
        state, reference = self.split(observation)
        encoded = self.state_encoder(state)
        outputs = []
        for t in range(len(observation)):
            hidden = hidden*(~episode_start[t].bool()).to(hidden.dtype).unsqueeze(-1)
            output, next_hidden = self.gru(encoded[t:t+1], hidden.unsqueeze(0))
            hidden = next_hidden.squeeze(0)
            outputs.append(output.squeeze(0))
        features = torch.cat((torch.stack(outputs), self.reference_encoder(reference)), dim=-1)
        return self.action_head(features), hidden

    def distribution(self, observation, hidden, episode_start):
        mean, hidden = self(observation, hidden, episode_start)
        return TanhNormal(mean, self.log_std), hidden

    @torch.no_grad()
    def burn_in(self, context):
        """Recompute memory under CURRENT weights from this episode's past frames."""
        if not len(context):
            return self.initial_hidden(1)
        state, _ = self.split(context)
        _, hidden = self.gru(self.state_encoder(state).unsqueeze(1))
        return hidden.squeeze(0)


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['layers'][0] != CRITIC_DIM or config['layers'][-1] != 1:
            raise ValueError('Critic must map 172 inputs to one value')
        self.network = mlp(config['layers'], output_gain=1.)

    def forward(self, observation):
        if observation.shape[-1] != CRITIC_DIM:
            raise ValueError('Critic accepts 172 inputs')
        return self.network(observation).squeeze(-1)
