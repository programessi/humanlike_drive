"""Train and archive the M2 recurrent PPO engineering smoke."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import yaml

from .audit import ROOT, digest, write_json
from .checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint, NORMALIZATION
from .contract import INTERFACE_VERSION
from .evaluate import evaluate
from .models import Actor, Critic
from .ppo import PPO
from .rollout import Collector
from .workers import WorkerPool
from .monitoring import DevelopmentMonitor
from .tensorboard_logging import TrainingDashboard


def resolved_config(path):
    overrides = yaml.safe_load(Path(path).read_text())
    design = yaml.safe_load((ROOT/overrides['design']).read_text())
    result = copy.deepcopy(design)
    result.update({key:value for key,value in overrides.items() if key not in ('ppo','environment_overrides')})
    result['ppo'].update(overrides.get('ppo',{}))
    environment = yaml.safe_load((ROOT/overrides['environment_config']).read_text())
    for key, value in overrides.get('environment_overrides',{}).items():
        if isinstance(value,dict):
            environment[key].update(value)
        else:
            environment[key] = value
    result['environment'] = environment
    ppo = result['ppo']
    if ppo['rollout_steps'] % ppo['recurrent_sequence_steps'] or ppo['minibatch_steps'] % ppo['recurrent_sequence_steps']:
        raise ValueError('Rollout/minibatch steps must be multiples of sequence length')
    if result['interface_version'] != INTERFACE_VERSION:
        raise ValueError('Wrong interface version')
    return result


def parameter_change(initial, module):
    return {name: float(torch.linalg.vector_norm(parameter.detach().cpu()-initial[name]))
            for name,parameter in module.named_parameters()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/m2_smoke.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--total-steps', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip final comparison suites; configured periodic evaluation still runs')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = resolved_config(args.config)
    if args.total_steps is not None:
        config['total_steps'] = args.total_steps
    if config['total_steps'] < 1:
        raise ValueError('Positive total steps required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    torch.set_num_threads(config['torch_threads'])
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(config['device'])
    ppo_config = config['ppo']
    write_json(output/'resolved_config.json', config)
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), interface_version=INTERFACE_VERSION,
                      python=platform.python_version(), torch=torch.__version__, device=str(device),
                      normalization=NORMALIZATION, config_sha256=digest(args.config),
                      scene_sha256=digest(ROOT/config['environment']['scene']),
                      source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/unified_steering').glob('*.py'))},
                      algorithm_reference='SB3 2.6.0 PPO clipped objective/GAE/KL; custom GRU, separate critic, exact tanh density',
                      resumed_from=str(args.resume.resolve()) if args.resume else None)
    write_json(output/'provenance.json', provenance)
    pool = None
    dashboard = None
    begin = time.perf_counter()
    try:
        dashboard=TrainingDashboard(output,config)
        pool = WorkerPool(config['environment'], config['tasks'], ppo_config['num_envs'], seed)
        initial_action = pool.initial_action
        actor, critic = Actor(config['actor'],initial_action).to(device), Critic(config['critic']).to(device)
        algorithm = PPO(actor,critic,ppo_config,device)
        steps, updates = 0, 0
        if args.resume:
            payload = read_checkpoint(args.resume,device)
            for key in ('actor','critic','ppo','environment','tasks','seed'):
                if payload['config'][key] != config[key]:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            restore_checkpoint(payload,actor,critic,algorithm,restore_rng=True)
            steps, updates = payload['steps'], payload['updates']
        elif config.get('initialization'):
            settings=config['initialization']
            path=ROOT/settings['checkpoint']
            payload=read_checkpoint(path,device)
            if not np.allclose(initial_action,payload['initial_action'],atol=1e-7):
                raise ValueError('Initialization grasp action differs from source checkpoint')
            restore_checkpoint(payload,actor,critic)  # New optimizers and step counts intentionally retained.
            with torch.no_grad():
                actor.log_std[:14].fill_(np.log(settings['arm_std']))
                actor.log_std[14:].fill_(np.log(settings['hand_std']))
            write_json(output/'initialization.json',dict(source=str(path),source_sha256=digest(path),
                      optimizer_reset=True,counters_reset=True,arm_std=settings['arm_std'],hand_std=settings['hand_std'],
                      observation_and_reward_unchanged=True))
        starting_steps = steps
        initial_actor = {name:p.detach().cpu().clone() for name,p in actor.named_parameters()}
        initial_critic = {name:p.detach().cpu().clone() for name,p in critic.named_parameters()}
        save_checkpoint(output/'initial.pt',actor,critic,algorithm,config,initial_action,steps,updates)
        write_json(output/'model.json', dict(actor_parameters=sum(p.numel() for p in actor.parameters()),
                                           critic_parameters=sum(p.numel() for p in critic.parameters()),
                                           initial_deterministic_action=initial_action.tolist()))
        collector = Collector(pool,device)
        monitor = DevelopmentMonitor(output,config,actor,critic,algorithm,initial_action)
        print('Initial fixed development evaluation starting',flush=True)
        monitor.check(steps,updates,initial=True)
        dashboard.evaluation(monitor.last_report,monitor.last_directory,steps)
        failures = Counter()
        episodes_completed, timeouts, physics_steps = 0, 0, 0
        history = []
        cumulative_task_steps={name:0 for name in ('hold','positive','negative','roundtrip','unknown')}
        with (output/'updates.csv').open('w') as update_stream, (output/'episodes.jsonl').open('w') as episode_stream:
            writer = None
            while steps < config['total_steps']:
                rollout = collector.collect(actor,critic,ppo_config['rollout_steps'],ppo_config['gamma'],ppo_config['gae_lambda'])
                stats = algorithm.update(rollout)
                steps += ppo_config['num_envs']*ppo_config['rollout_steps']
                updates += 1
                physics_steps += rollout.stats['physics_steps']
                for episode in rollout.episodes:
                    episodes_completed += 1
                    if episode['terminated']:
                        failures[episode['reason']] += 1
                    timeouts += int(episode['truncated'])
                    episode_stream.write(json.dumps(dict(step=steps,**episode),allow_nan=False)+'\n')
                episode_stream.flush()
                row = dict(update=updates, steps=steps, wall_seconds=time.perf_counter()-begin,
                           mean_reward=float(rollout.rewards.mean()),
                           finished_episode_count=episodes_completed, failure_count=sum(failures.values()),
                           **rollout.stats, **stats)
                for task in cumulative_task_steps:
                    cumulative_task_steps[task]+=rollout.stats.get('task_steps_'+task,0)
                    row['cumulative_task_steps_'+task]=cumulative_task_steps[task]
                if writer is None:
                    writer = csv.DictWriter(update_stream,fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                update_stream.flush()
                history.append(row)
                dashboard.update(row,actor,critic,algorithm,rollout.episodes)
                print(json.dumps({key:row[key] for key in ('update','steps','mean_reward','approx_kl','value_loss',
                                                           'probability_replay_error','failure_count','wall_seconds')}),flush=True)
                if updates % config['checkpoint_every_updates'] == 0:
                    save_checkpoint(output/'latest.pt',actor,critic,algorithm,config,initial_action,steps,updates)
                    if monitor.retain:
                        monitor.save_named(steps,updates)
                evaluation = monitor.check(steps,updates)
                if evaluation is not None:
                    dashboard.evaluation(monitor.last_report,monitor.last_directory,steps)
                    print(json.dumps(dict(development=evaluation),allow_nan=False),flush=True)
            save_checkpoint(output/'final.pt',actor,critic,algorithm,config,initial_action,steps,updates)
            save_checkpoint(output/'latest.pt',actor,critic,algorithm,config,initial_action,steps,updates)
        pool.close()
        pool = None
        training_wall = time.perf_counter()-begin
        actor_changes = parameter_change(initial_actor,actor)
        critic_changes = parameter_change(initial_critic,critic)
        summary = dict(kind=config.get('run_kind','M2_recurrent_PPO_engineering_smoke_not_task_success'),
                       steps=steps, interaction_steps_this_run=steps-starting_steps, updates=updates,
                       training_wall_seconds=training_wall, control_steps_per_second=(steps-starting_steps)/training_wall,
                       physics_steps=physics_steps, episodes_completed=episodes_completed, timeouts=timeouts,
                       failure_reasons=dict(failures), finite_training=True,
                       max_probability_replay_error=max(row['probability_replay_error'] for row in history) if history else 0.,
                       total_optimizer_steps=sum(row['optimizer_steps'] for row in history),
                       actor_parameter_changes=actor_changes, critic_parameter_changes=critic_changes,
                       normalization=NORMALIZATION, task_success_claimed=False)
        summary['task_steps_this_run']=cumulative_task_steps
        if monitor.last_report is not None:
            summary['last_development_evaluation']=monitor.last_report
            summary['best_development_step']=monitor.best_step
        summary['engineering_budget_passed'] = (steps-starting_steps >= 50000 and steps-starting_steps <= 100000
                                                and bool(history) and max(actor_changes.values()) > 0 and max(critic_changes.values()) > 0)
        write_json(output/'summary.json',summary)
        if not args.skip_evaluation:
            summary['final_evaluation'] = evaluate(config,actor,output/'evaluation_final')
            payload = read_checkpoint(output/'initial.pt',device)
            restore_checkpoint(payload,actor,critic)
            summary['initial_evaluation'] = evaluate(config,actor,output/'evaluation_initial')
            summary['fixed_target_evaluation'] = evaluate(config,None,output/'evaluation_fixed_targets')
            write_json(output/'summary.json',summary)
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)
    except BaseException as error:
        write_json(output/'failure.json',dict(error=repr(error),wall_seconds=time.perf_counter()-begin))
        raise
    finally:
        if dashboard is not None:
            dashboard.close()
        if pool is not None:
            pool.close()


if __name__ == '__main__':
    main()
