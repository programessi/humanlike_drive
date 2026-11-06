"""Measure CPU/CUDA learner with the current CPU MuJoCo workers, not GPU physics."""
import argparse
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch

from .audit import ROOT, write_json
from .models import Actor, Critic
from .ppo import PPO
from .rollout import Collector
from .train import resolved_config
from .workers import WorkerPool


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(config, device_name, repeats):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device(device_name)
    cfg = config['ppo']
    begin = time.perf_counter()
    pool = WorkerPool(config['environment'],config['tasks'],cfg['num_envs'],config['seed'])
    startup = time.perf_counter()-begin
    try:
        actor = Actor(config['actor'],pool.initial_action).to(device)
        critic = Critic(config['critic']).to(device)
        ppo = PPO(actor,critic,cfg,device)
        collector = Collector(pool,device)
        measurements = []
        for index in range(repeats+1):
            synchronize(device)
            begin = time.perf_counter()
            rollout = collector.collect(actor,critic,cfg['rollout_steps'],cfg['gamma'],cfg['gae_lambda'])
            synchronize(device)
            collection = time.perf_counter()-begin
            begin = time.perf_counter()
            stats = ppo.update(rollout)
            synchronize(device)
            update = time.perf_counter()-begin
            row = dict(collection_seconds=collection,update_seconds=update,
                       optimizer_steps=stats['optimizer_steps'],probability_replay_error=stats['probability_replay_error'])
            if index:
                measurements.append(row)
        duration = sum(row['collection_seconds']+row['update_seconds'] for row in measurements)
        calls = cfg['num_envs']*cfg['rollout_steps']*repeats
        return dict(device=device_name,worker_startup_seconds=startup,warmup_batches=1,
                    measured_batches=repeats,control_calls=calls,collection_and_update_seconds=duration,
                    end_to_end_calls_per_second=calls/duration,measurements=measurements)
    finally:
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/m2_smoke.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = resolved_config(args.config)
    report = dict(kind='CPU_MuJoCo_with_CPU_or_CUDA_learner_benchmark',
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  environment_count=config['ppo']['num_envs'],rollout_steps=config['ppo']['rollout_steps'],
                  sequence_steps=config['ppo']['recurrent_sequence_steps'],minibatch_steps=config['ppo']['minibatch_steps'],
                  torch_threads=1,physics_backend='ordinary CPU MuJoCo, unchanged',results=[],
                  limits='Short sequential benchmark; same initial weights/seeds/config, CPU/CUDA random streams and trajectories may differ; not a large-batch GPU physics test.')
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        result = measure(copy.deepcopy(config),device,args.repeats)
        report['results'].append(result)
        print(result,flush=True)
        write_json(args.output/'summary.json',report)


if __name__ == '__main__':
    main()
