"""Fixed development suite: deterministic learned actor or fixed joint targets."""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from .audit import ROOT, write_json
from .checkpoint import read_checkpoint, restore_checkpoint
from .env import SteeringEnv
from .models import Actor, Critic
from .tracing import TraceRecorder


@torch.no_grad()
def evaluate(config, actor, output, *, diagnostics=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env_config = copy.deepcopy(config['environment'])
    env_config['initial_support'] = 'both'
    env_config['episode_seconds'] = 30.
    env_config['reference']['speed_deg_s'] = 5.
    env_config['reference']['hold_seconds'] = 1.
    env_config['diagnostics'] = diagnostics
    env = SteeringEnv(env_config)
    if actor is not None:
        actor.eval()
        device = next(actor.parameters()).device
    cases = []
    definitions = [('neutral_hold',[0]), ('offset_plus5',[0,5]), ('offset_minus5',[0,-5]),
                   ('roundtrip_plus_first',[0,5,-5,0]), ('roundtrip_minus_first',[0,-5,5,0])]
    for index, (name, offsets) in enumerate(definitions):
        obs, initial = env.reset(seed=7000+index, options={'support':'both'})
        angles = initial['initial_angle_rad']+np.deg2rad(offsets)
        env.set_reference(angles)
        obs = env.encoder.build(env.reference, 0.)
        env.last_obs = obs
        duration = 3. if len(offsets) == 1 else env.reference.duration+3.
        steps = int(np.ceil(duration/env.period-1e-8))
        env.config['episode_seconds'] = steps*env.period
        hidden = actor.initial_hidden(1) if actor is not None else None
        recorder = TraceRecorder(env,dict(case=name,controller='actor' if actor is not None else 'fixed_targets')) if diagnostics else None
        rows = []
        interventions, ticks = 0, 0
        reason, terminated = None, False
        for step in range(steps):
            actor_input = obs['actor'].copy()
            if actor is None:
                action = env.hold_action()
                if diagnostics:
                    mean = np.arctanh(np.clip(action,-.999999,.999999))
                    std = np.zeros(26)
            else:
                distribution, hidden = actor.distribution(torch.as_tensor(obs['actor'], device=device).view(1,1,-1),
                                                           hidden, torch.tensor([[step == 0]], device=device))
                action = distribution.deterministic()[0,0].cpu().numpy()
                if diagnostics:
                    mean = distribution.mean[0,0].cpu().numpy()
                    std = distribution.log_std.exp().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if not info['observation_valid']:
                raise FloatingPointError('Evaluation produced invalid physics observation')
            if recorder is not None:
                recorder.record(env,info,actor_input,mean,std,mean,action)
            rows.append(dict(time_s=(step+1)*env.period, reference_deg=float(np.rad2deg(info['reference_angle_rad'])),
                             angle_deg=float(np.rad2deg(info['wheel_angle_rad'])),
                             error_deg=float(np.rad2deg(info['error_rad'])), reward=reward,
                             collision_peak_n=info['collision_peak_n'], interventions=info['execution_interventions']))
            interventions += info['execution_interventions']
            ticks += info['physics_steps']
            reason = info['termination_reason']
            if terminated or truncated:
                break
        with (output/f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        errors = np.array([row['error_deg'] for row in rows])
        rmse = float(np.sqrt(np.mean(errors**2)))
        complete = not terminated and len(rows) == steps
        tail = [row for row in rows if row['time_s'] >= steps*env.period-3.-1e-8] if complete else []
        anchor = float(np.rad2deg(initial['initial_angle_rad'])) if steps*env.period <= 3.+1e-8 else (tail[0]['angle_deg'] if tail else 0.)
        tail_drift = max(abs(row['angle_deg']-anchor) for row in tail) if tail else None
        cases.append(dict(name=name, rmse_deg=rmse, peak_error_deg=float(np.max(np.abs(errors))),
                          hold_3s_drift_deg=tail_drift,
                          hold_3s_peak_target_error_deg=max(abs(row['error_deg']) for row in tail) if tail else None,
                          safety_completion=complete, tracking_success=complete and rmse <= 2. and tail_drift <= 2.,
                          termination_reason=reason, steps=len(rows),
                          intervention_fraction=interventions/max(ticks,1), trace=f'{name}.csv'))
        if recorder is not None:
            cases[-1]['diagnostics'] = recorder.save(output/f'{name}.npz')
    report = dict(kind='fixed_development_evaluation_not_generalization_test',
                  controller='deterministic_actor' if actor is not None else 'fixed_joint_targets', cases=cases,
                  mean_case_rmse_deg=float(np.mean([case['rmse_deg'] for case in cases])),
                  safety_completion_rate=float(np.mean([case['safety_completion'] for case in cases])),
                  tracking_success_rate=float(np.mean([case['tracking_success'] for case in cases])),
                  all_cases_complete=all(case['safety_completion'] for case in cases))
    moving = [case for case in cases if case['name'] != 'neutral_hold']
    report['mean_moving_case_rmse_deg'] = float(np.mean([case['rmse_deg'] for case in moving]))
    report['worst_moving_case_rmse_deg'] = float(max(case['rmse_deg'] for case in moving))
    report['time_pooled_rmse_deg'] = float(np.sqrt(sum(case['rmse_deg']**2*case['steps'] for case in cases)/sum(case['steps'] for case in cases)))
    report['pooled_metric_scope'] = 'Observed samples only; consult completion flags for early failures'
    write_json(output/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-targets', action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = read_checkpoint(args.checkpoint)
    config = payload['config']
    actor, critic = Actor(config['actor'], payload['initial_action']), Critic(config['critic'])
    restore_checkpoint(payload, actor, critic)
    result = evaluate(config, None if args.fixed_targets else actor, args.output,diagnostics=args.diagnostics)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == '__main__':
    main()
