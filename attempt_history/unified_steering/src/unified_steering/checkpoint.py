"""Atomic checkpoints with architecture, optimizer, scaling contract and RNG state."""
import os
import random
from pathlib import Path

import numpy as np
import torch

from .contract import INTERFACE_VERSION

NORMALIZATION = 'fixed-v1: observations.py; no running moments or reward normalization'


def save_checkpoint(path, actor, critic, ppo, config, initial_action, steps, updates):
    path = Path(path)
    payload = dict(interface_version=INTERFACE_VERSION, normalization=NORMALIZATION,
                   actor=actor.state_dict(), critic=critic.state_dict(),
                   actor_optimizer=ppo.actor_optimizer.state_dict(), critic_optimizer=ppo.critic_optimizer.state_dict(),
                   config=config, initial_action=np.asarray(initial_action).tolist(), steps=steps, updates=updates,
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   resume_semantics='Restore training state; simulation workers start new episodes with cleared GRU')
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path, device='cpu'):
    # Checkpoints are generated locally by this project; torch optimizer/RNG
    # payloads require weights_only=False. Do not load untrusted pickle files.
    result = torch.load(path, map_location=device, weights_only=False)
    if result['interface_version'] != INTERFACE_VERSION or result['normalization'] != NORMALIZATION:
        raise ValueError('Checkpoint interface/scaling mismatch')
    return result


def restore_checkpoint(payload, actor, critic, ppo=None, restore_rng=False):
    actor.load_state_dict(payload['actor'])
    critic.load_state_dict(payload['critic'])
    if ppo is not None:
        ppo.actor_optimizer.load_state_dict(payload['actor_optimizer'])
        ppo.critic_optimizer.load_state_dict(payload['critic_optimizer'])
    if restore_rng:
        torch.set_rng_state(payload['torch_rng'].cpu())
        np.random.set_state(payload['numpy_rng'])
        random.setstate(payload['python_rng'])
        if payload['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
