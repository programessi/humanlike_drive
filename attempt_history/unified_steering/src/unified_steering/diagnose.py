"""M3-0 frozen-policy replay, noise ablations and cloned-state direction probes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from .audit import ROOT, digest, write_json
from .checkpoint import read_checkpoint, restore_checkpoint
from .env import SteeringEnv
from .evaluate import evaluate
from .models import Actor, Critic
from .snapshots import capture, restore
from .tracing import TraceRecorder

_CACHE = {}


def actor_hash(actor):
    value = hashlib.sha256()
    for name,tensor in actor.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def initialize_worker(protocol):
    torch.set_num_threads(1)
    _CACHE['protocol'] = protocol
    _CACHE['actors'] = {}
    for name,key in (('initial','initial_checkpoint'),('final','final_checkpoint')):
        payload = read_checkpoint(ROOT/protocol[key])
        actor,critic = Actor(payload['config']['actor'],payload['initial_action']),Critic(payload['config']['critic'])
        restore_checkpoint(payload,actor,critic)
        actor.eval()
        _CACHE['actors'][name] = actor
        _CACHE[name+'_hash'] = actor_hash(actor)
        _CACHE['config'] = payload['config']
    expected=np.r_[np.full(14,protocol['baseline_std'][0]),np.full(12,protocol['baseline_std'][1])]
    if not np.allclose(_CACHE['actors']['initial'].log_std.detach().exp().numpy(),expected,atol=1e-7):
        raise ValueError('Declared initial exploration standard deviation differs from checkpoint')
    env_config = copy.deepcopy(_CACHE['config']['environment'])
    env_config.update(initial_support='both',episode_seconds=30.,diagnostics=True)
    env_config['reference']['speed_deg_s'] = 5.
    env_config['reference']['hold_seconds'] = 1.
    _CACHE['env'] = SteeringEnv(env_config)


def assert_frozen(name):
    if actor_hash(_CACHE['actors'][name]) != _CACHE[name+'_hash']:
        raise RuntimeError('Frozen actor parameters changed')


@torch.no_grad()
def frozen_suite(job):
    name,directory = job
    actor = None if name=='fixed_targets' else _CACHE['actors'][name]
    report = evaluate(_CACHE['config'],actor,Path(directory),diagnostics=True)
    if actor is not None:
        assert_frozen(name)
    return name,report


def policy_step(actor, obs, hidden, first, epsilon=None, std_override=None):
    distribution,next_hidden = actor.distribution(torch.as_tensor(obs['actor']).view(1,1,-1),hidden,
                                                  torch.tensor([[first]],dtype=torch.bool))
    mean = distribution.mean[0,0].detach().numpy()
    std = distribution.log_std.exp().detach().numpy()
    if std_override is not None:
        std = np.r_[np.full(14,std_override[0]),np.full(12,std_override[1])].astype(np.float32)
    latent = mean.copy() if epsilon is None else mean+std*epsilon
    action = torch.from_numpy(latent).tanh().numpy()
    return action,next_hidden,mean,std,latent


@torch.no_grad()
def hold_trial(job):
    group,controller,seed,std_override,directory = job
    protocol = _CACHE['protocol']
    env,actor = _CACHE['env'],_CACHE['actors'][controller]
    env.config['reference']['hold_seconds'] = 1.
    env.config['episode_seconds'] = protocol['hold_seconds']
    obs,initial = env.reset(seed=seed,options={'support':'both'})
    env.set_reference([initial['initial_angle_rad']])
    obs = env.encoder.build(env.reference,0.)
    env.last_obs = obs
    hidden = actor.initial_hidden(1)
    steps = round(protocol['hold_seconds']/env.period)
    # Common random numbers: same 26D epsilon sequence for all groups and actors.
    epsilon = np.random.default_rng(seed).standard_normal((steps,26)).astype(np.float32)
    recorder = TraceRecorder(env,dict(group=group,controller=controller,seed=seed,std_override=std_override,
                                     checkpoint_actor_hash=_CACHE[controller+'_hash'],sampling='common_numpy_epsilon'))
    errors=[]
    terminated=False
    for step in range(steps):
        actor_input = obs['actor'].copy()
        action,hidden,mean,std,latent = policy_step(actor,obs,hidden,step==0,epsilon[step],std_override)
        obs,_,terminated,truncated,info = env.step(action)
        recorder.record(env,info,actor_input,mean,std,latent,action)
        errors.append(float(info['error_rad']))
        if terminated or truncated:
            break
    complete = not terminated and len(errors)==steps
    trace = recorder.save(Path(directory)/f'{group}_{seed}.npz')
    rmse = float(np.rad2deg(np.sqrt(np.mean(np.square(errors)))))
    peak = float(np.rad2deg(np.max(np.abs(errors))))
    result = dict(group=group,controller=controller,seed=seed,std_override=std_override,
                  rmse_deg=rmse,peak_error_deg=peak,safety_completion=complete,
                  hold_success=bool(complete and peak<=2.),steps=len(errors),
                  trace=trace,trace_file=f'{group}_{seed}.npz')
    assert_frozen(controller)
    return result


@torch.no_grad()
def direction_probe(job):
    controller,anchor,directory = job
    protocol = _CACHE['protocol']
    env,actor = _CACHE['env'],_CACHE['actors'][controller]
    env.config['episode_seconds']=30.
    env.config['reference']['hold_seconds']=1.
    obs,initial = env.reset(seed=909,options={'support':'both'})
    theta = initial['initial_angle_rad']
    env.set_reference([theta,theta+np.deg2rad(5)] if anchor=='positive_approach_2s' else [theta])
    obs=env.encoder.build(env.reference,0.)
    env.last_obs=obs
    hidden=actor.initial_hidden(1)
    warmup = {'initial':0,'neutral_1s':20,'positive_approach_2s':40}[anchor]
    history=[]
    for step in range(warmup):
        history.append(obs['actor'].copy())
        action,hidden,_,_,_ = policy_step(actor,obs,hidden,step==0)
        obs,_,term,trunc,info=env.step(action)
        if term or trunc:
            return dict(controller=controller,anchor=anchor,valid=False,reason=info['termination_reason'])
    snapshot=capture(env)
    anchor_hidden=hidden.clone()
    anchor_angle=float(env.sim.data.qpos[env.sim.wheel_q])
    physical_hash=hashlib.sha256(snapshot['integration'].tobytes()).hexdigest()
    hidden_hash=hashlib.sha256(anchor_hidden.numpy().tobytes()).hexdigest()
    np.savez_compressed(Path(directory)/f'{controller}_{anchor}.anchor.npz',
                        integration=snapshot['integration'],warmstart=snapshot['warmstart'],hidden=anchor_hidden.numpy(),
                        actor_input=obs['actor'],history=np.asarray(history,dtype=np.float32).reshape(-1,152))
    # Static signed error is an input-sensitivity probe only, never stepped in physics.
    instantaneous=[]
    for sign in (1,-1):
        alternate=obs['actor'].copy()
        alternate[138:149]=sign*protocol['probe_amplitude_deg']/180.
        alternate[149]=0.
        distribution,_=actor.distribution(torch.tensor(alternate).view(1,1,-1),anchor_hidden,
                                          torch.tensor([[warmup==0]]))
        instantaneous.append(distribution.deterministic()[0,0].numpy())
    branches=[]
    first_actions=[]
    for sign in (1,-1):
        restore(env,snapshot)
        hidden=anchor_hidden.clone()
        env.config['reference']['hold_seconds']=0.
        env.set_reference([anchor_angle,anchor_angle+sign*np.deg2rad(protocol['probe_amplitude_deg'])])
        obs=env.encoder.build(env.reference,0.)
        env.last_obs=obs
        recorder=TraceRecorder(env,dict(controller=controller,anchor=anchor,sign=sign,
                                        anchor_physics_hash=physical_hash,anchor_hidden_hash=hidden_hash,
                                        checkpoint_actor_hash=_CACHE[controller+'_hash'],reference='smooth_no_dwell'))
        terminated=False
        count=round(protocol['probe_seconds']/env.period)
        for step in range(count):
            actor_input=obs['actor'].copy()
            action,hidden,mean,std,latent=policy_step(actor,obs,hidden,warmup==0 and step==0)
            if step==0:
                first_actions.append(action.copy())
            obs,_,terminated,truncated,info=env.step(action)
            recorder.record(env,info,actor_input,mean,std,latent,action)
            if terminated or truncated:
                break
        filename=f'{controller}_{anchor}_{sign:+d}.npz'
        trace=recorder.save(Path(directory)/filename)
        delta=float(np.rad2deg(env.sim.data.qpos[env.sim.wheel_q]-anchor_angle))
        branches.append(dict(sign=sign,wheel_change_deg=delta,signed_progress_deg=sign*delta,
                             safety_completion=bool(not terminated and step+1==count),
                             trace=trace,trace_file=filename))
    sensitivity=instantaneous[0]-instantaneous[1]
    smooth=first_actions[0]-first_actions[1]
    assert_frozen(controller)
    return dict(controller=controller,anchor=anchor,valid=True,
                anchor_angle_deg=float(np.rad2deg(anchor_angle)),anchor_physics_hash=physical_hash,
                anchor_hidden_hash=hidden_hash,instant_action_rms_difference=float(np.sqrt(np.mean(sensitivity**2))),
                instant_arm_rms_difference=float(np.sqrt(np.mean(sensitivity[:14]**2))),
                instant_hand_rms_difference=float(np.sqrt(np.mean(sensitivity[14:]**2))),
                smooth_first_action_rms_difference=float(np.sqrt(np.mean(smooth**2))),
                static_plus_action=instantaneous[0].tolist(),static_minus_action=instantaneous[1].tolist(),branches=branches)


def aggregate(trials):
    groups={}
    for trial in trials:
        groups.setdefault(trial['group'],[]).append(trial)
    result={}
    for name,rows in groups.items():
        completed=[r['rmse_deg'] for r in rows if r['safety_completion']]
        result[name]=dict(trials=len(rows),safety_completion_rate=float(np.mean([r['safety_completion'] for r in rows])),
                          hold_success_rate=float(np.mean([r['hold_success'] for r in rows])),
                          mean_observed_prefix_rmse_deg=float(np.mean([r['rmse_deg'] for r in rows])),
                          mean_completed_rmse_deg=float(np.mean(completed)) if completed else None,
                          mean_peak_error_deg=float(np.mean([r['peak_error_deg'] for r in rows])))
    return result


def improves(candidate, baseline):
    for field in ('hold_success_rate','safety_completion_rate'):
        if candidate[field] != baseline[field]:
            return candidate[field]>baseline[field]
    a,b=candidate['mean_completed_rmse_deg'],baseline['mean_completed_rmse_deg']
    return a is not None and b is not None and a < .9*b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/m30_diagnosis.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=yaml.safe_load(args.config.read_text())
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    for name in ('trials','probes'):
        (output/name).mkdir()
    (output/'config.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    provenance=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),kind='M3_0_frozen_diagnosis_no_training',
                    protocol_sha256=digest(args.config),checkpoint_sha256={key:digest(ROOT/protocol[key]) for key in ('initial_checkpoint','final_checkpoint')},
                    source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/unified_steering').glob('*.py'))})
    write_json(output/'provenance.json',provenance)
    report=dict(kind='M3_0_frozen_policy_diagnosis',training_updates=0,noise_seed_count=len(protocol['noise_seeds']))
    with ProcessPoolExecutor(max_workers=protocol['workers'],mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker,initargs=(protocol,)) as pool:
        suites=dict(pool.map(frozen_suite,[(name,str(output/'frozen'/name)) for name in ('initial','final','fixed_targets')]))
        report['frozen_suites']=suites
        checks=[]
        for name,directory in (('initial','evaluation_initial'),('final','evaluation_final'),('fixed_targets','evaluation_fixed_targets')):
            archived=json.loads((ROOT/protocol['archived_result']/directory/'summary.json').read_text())
            for old,new in zip(archived['cases'],suites[name]['cases'],strict=True):
                difference=abs(old['rmse_deg']-new['rmse_deg'])
                checks.append(dict(controller=name,case=old['name'],rmse_difference_deg=difference,passed=difference<=1e-6))
        report['frozen_regression']=checks
        if not all(check['passed'] for check in checks):
            write_json(output/'summary.json',report)
            raise RuntimeError('Instrumentation changed frozen evaluation results')
        print('Frozen regression passed: 15/15 cases',flush=True)
        jobs=[]
        for group,controller,std in (('initial_default','initial',None),('final_default','final',None),
                                     ('initial_arm_low','initial',protocol['arm_low_std']),
                                     ('initial_hand_low','initial',protocol['hand_low_std'])):
            jobs.extend((group,controller,seed,std,str(output/'trials')) for seed in protocol['noise_seeds'])
        trials=list(pool.map(hold_trial,jobs))
        groups=aggregate(trials)
        combine=bool(improves(groups['initial_arm_low'],groups['initial_default']) and
                     improves(groups['initial_hand_low'],groups['initial_default']))
        report['combined_trial_triggered']=combine
        if combine:
            extra=[('initial_combined_low','initial',seed,protocol['combined_std'],str(output/'trials')) for seed in protocol['noise_seeds']]
            trials.extend(pool.map(hold_trial,extra))
            groups=aggregate(trials)
        report['noise_groups']=groups
        report['noise_trials']=trials
        print('Noise comparisons:',json.dumps(groups,allow_nan=False),flush=True)
        report['direction_probes']=list(pool.map(direction_probe,[(name,anchor,str(output/'probes'))
                                          for name in ('initial','final') for anchor in protocol['probe_anchors']]))
    for key,expected in provenance['checkpoint_sha256'].items():
        if digest(ROOT/protocol[key]) != expected:
            raise RuntimeError('Source checkpoint file changed')
    report['checkpoint_files_unchanged']=True
    report['regression_passed']=all(check['passed'] for check in checks)
    report['limits']='Single nominal both-hand snapshot; ten paired noise seeds; frozen inference and short branches, not a convergence study or learned recovery proof'
    write_json(output/'summary.json',report)
    print('Diagnostic complete:',output,flush=True)


if __name__=='__main__':
    main()
