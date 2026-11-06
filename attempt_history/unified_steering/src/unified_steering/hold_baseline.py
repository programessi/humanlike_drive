"""Step 4.1 frozen holding, bounded pulse validation and paired exploration checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from .audit import ROOT, digest, write_json
from .checkpoint import read_checkpoint
from .diagnose import actor_hash, policy_step
from .env import SteeringEnv
from .models import Actor
from .tracing import TraceRecorder, first_sustained

_CACHE = {}


def initialize(protocol):
    torch.set_num_threads(1)
    p = read_checkpoint(ROOT/protocol['initial_checkpoint'])
    actor = Actor(p['config']['actor'],p['initial_action'])
    actor.load_state_dict(p['actor'])
    actor.eval()
    _CACHE.update(protocol=protocol,actor=actor,actor_hash=actor_hash(actor),config=p['config']['environment'])


@torch.no_grad()
def trial(job):
    name,controller,seconds,torque,seed,std,directory = job
    p = _CACHE['protocol']
    config = copy.deepcopy(_CACHE['config'])
    config.update(initial_support='both',episode_seconds=seconds,diagnostics=True,
                  wheel_disturbance=dict(max_abs_torque_nm=.05,pulses=[dict(start_s=p['pulse_start_s'],
                     duration_s=p['pulse_duration_s'],torque_nm=torque)] if torque else []))
    env = SteeringEnv(config)
    obs,initial = env.reset(seed=seed,options={'support':'both'})
    env.set_reference([initial['initial_angle_rad']])
    obs = env.encoder.build(env.reference,0.)
    env.last_obs = obs
    actor = _CACHE['actor']
    hidden = actor.initial_hidden(1)
    count = round(seconds/env.period)
    epsilon = np.random.default_rng(seed).standard_normal((count,26)).astype(np.float32) if std else None
    recorder = TraceRecorder(env,dict(case=name,controller=controller,seconds=seconds,torque_nm=torque,
                                     seed=seed,std_override=std,checkpoint_actor_hash=_CACHE['actor_hash'],
                                     sampling='common_numpy_epsilon' if std else 'deterministic'))
    errors=[]
    for step in range(count):
        previous = obs['actor'].copy()
        if controller=='fixed_targets':
            action=env.hold_action()
            mean=np.arctanh(np.clip(action,-.999999,.999999));scale=np.zeros(26);latent=mean
        else:
            action,hidden,mean,scale,latent=policy_step(actor,obs,hidden,step==0,
                                                       epsilon[step] if std else None,std)
        obs,_,terminated,truncated,info=env.step(action)
        recorder.record(env,info,previous,mean,scale,latent,action)
        errors.append(info['error_rad'])
        if terminated or truncated:
            break
    complete=bool(not terminated and step+1==count)
    filename=f'{name}_{seconds:g}s_{controller}_{seed}.npz'
    trace=recorder.save(Path(directory)/filename)
    physics=np.concatenate(recorder.physics)
    fields=info['diagnostics']['physics_fields']
    col=lambda name:physics[:,fields.index(name)]
    error=np.rad2deg(initial['initial_angle_rad']-col('wheel_angle_rad'))
    peak=float(np.max(np.abs(error)))
    end=p['pulse_start_s']+p['pulse_duration_s']
    recovery_start=first_sustained(col('time_s'),(col('time_s')>=end-1e-9)&(np.abs(error)<=p['gates']['recovery_band_deg']),
                                   p['gates']['recovery_dwell_s'],env.dt) if torque else None
    final=error[col('time_s')>=seconds-1.-1e-9] if complete else []
    final_peak=float(np.max(np.abs(final))) if len(final) else None
    recovery_delay=max(0.,recovery_start-end) if recovery_start is not None else None
    pulse=col('disturbance_torque_nm')
    expected_impulse=torque*p['pulse_duration_s']
    verified=bool(abs(float(pulse.sum()*env.dt)-expected_impulse)<1e-10 and
                  np.max(np.abs(col('total_wheel_torque_nm')-col('applied_centering_nm')-pulse))<1e-12)
    if actor_hash(actor)!=_CACHE['actor_hash']:
        raise RuntimeError('Frozen actor changed')
    return dict(name=name,controller=controller,seconds=seconds,torque_nm=torque,seed=seed,std=std,
                complete=complete,rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(errors))))),
                physics_peak_error_deg=peak,final_1s_peak_error_deg=final_peak,
                recovery_delay_s=recovery_delay,
                hold_pass=bool(complete and peak<=p['gates']['hold_peak_error_deg']),
                pulse_delivery_verified=verified,applied_impulse_nm_s=float(pulse.sum()*env.dt),
                trace_file=filename,trace=trace)


def summarize_noise(rows, protocol):
    report={}
    g=protocol['gates']
    for seconds in protocol['noise_durations_s']:
        groups={name:[r for r in rows if r['name']==name and r['seconds']==seconds] for name in protocol['noise_groups']}
        result={name:dict(trials=len(group),safety_completion_rate=float(np.mean([r['complete'] for r in group])),
                          hold_pass_rate=float(np.mean([r['hold_pass'] for r in group])),
                          complete_rmse_deg=float(np.mean([r['rmse_deg'] for r in group if r['complete']])) if any(r['complete'] for r in group) else None)
                for name,group in groups.items()}
        base={r['seed']:r for r in groups['default']}
        paired=[r for r in groups['arm_low'] if r['complete'] and base[r['seed']]['complete']]
        ratio=float(np.mean([r['rmse_deg'] for r in paired])/np.mean([base[r['seed']]['rmse_deg'] for r in paired])) if paired else None
        result['comparison']=dict(common_complete_pairs=len(paired),common_complete_rmse_ratio=ratio,
                                  exploration_candidate_pass=bool(result['arm_low']['safety_completion_rate']>=g['noise_min_safety_completion_rate']
                                    and len(paired)>=g['noise_min_common_complete_pairs'] and ratio<=g['noise_max_common_complete_rmse_ratio']))
        report[str(seconds)]=result
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/m31_hold_baseline.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=yaml.safe_load(args.config.read_text())
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'traces').mkdir()
    (output/'config.yaml').write_text(yaml.safe_dump(p,sort_keys=False))
    provenance=dict(checkpoint=str(ROOT/p['initial_checkpoint']),checkpoint_sha256=digest(ROOT/p['initial_checkpoint']),
                    config_sha256=digest(args.config),source_sha256={str(x.relative_to(ROOT)):digest(x) for x in sorted((ROOT/'src/unified_steering').glob('*.py'))})
    write_json(output/'provenance.json',provenance)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    jobs=[(case['name'],controller,case['seconds'],case['torque_nm'],200,None,str(output/'traces'))
          for case in p['cases'] for controller in ('initial','fixed_targets')]
    with ProcessPoolExecutor(max_workers=p['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(p,)) as pool:
        nominal=list(pool.map(trial,jobs))
        print('Deterministic cases complete',flush=True)
        noise=list(pool.map(trial,[(group,'initial',seconds,0.,seed,std,str(output/'traces'))
                                  for seconds in p['noise_durations_s'] for group,std in p['noise_groups'].items() for seed in p['noise_seeds']]))
    pulse_checks=[]
    recovery=[]
    for row in nominal:
        if not row['torque_nm']:continue
        baseline=next(r for r in nominal if r['controller']==row['controller'] and r['name']=='hold_10s')
        with np.load(output/'traces'/row['trace_file'],allow_pickle=False) as a,np.load(output/'traces'/baseline['trace_file'],allow_pickle=False) as b:
            col=list(a['physics_fields']).index('wheel_angle_rad')
            full=row['complete'] and baseline['complete']
            delta=float(np.rad2deg(np.max(np.abs(a['physics'][:,col]-b['physics'][:,col])))) if full else None
        informative=bool(full and delta>=p['gates']['disturbance_min_paired_deviation_deg'])
        pulse_checks.append(dict(controller=row['controller'],case=row['name'],paired_peak_deviation_deg=delta,
                                 informative=informative,delivery_verified=row['pulse_delivery_verified']))
        if row['controller']=='initial':
            recovery.append(dict(case=row['name'],informative=informative,
                    passed=bool(informative and row['hold_pass'] and row['recovery_delay_s'] is not None
                                and row['recovery_delay_s']<=p['gates']['recovery_deadline_s']
                                and row['final_1s_peak_error_deg']<=p['gates']['recovery_band_deg'])))
    old=json.loads((ROOT/p['fixed_reference_archive']).read_text())
    anchor=next(r for r in nominal if r['controller']=='initial' and r['name']=='hold_3s')
    old_hold=next(r for r in old['cases'] if r['name']=='neutral_hold')
    difference=abs(anchor['rmse_deg']-old_hold['rmse_deg'])
    if difference>1e-6:raise RuntimeError('No-pulse initial baseline changed')
    if digest(ROOT/p['initial_checkpoint'])!=provenance['checkpoint_sha256']:raise RuntimeError('Source checkpoint changed')
    noise_report=summarize_noise(noise,p)
    report=dict(kind='step_4_1_frozen_hold_baseline',training_updates=0,
                evaluation_complete=True,deterministic_cases=nominal,noise_trials=noise,noise_summary=noise_report,
                pulse_checks=pulse_checks,recovery_checks=recovery,baseline_rmse_difference_deg=difference,
                all_pulse_deliveries_verified=all(r['pulse_delivery_verified'] for r in nominal+noise),
                nominal_initial_hold_pass=all(r['hold_pass'] for r in nominal if r['controller']=='initial' and not r['torque_nm']),
                disturbance_recovery_qualified=all(r['passed'] for r in recovery),
                exploration_candidate_pass=all(r['comparison']['exploration_candidate_pass'] for r in noise_report.values()),
                full_M3_A_stage_qualified=False,
                limits='One nominal grasp snapshot, 10 paired exploration seeds, frozen inference; no trained recovery or independent stage acceptance claim')
    write_json(output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('nominal_initial_hold_pass','disturbance_recovery_qualified','exploration_candidate_pass','noise_summary','pulse_checks')},allow_nan=False),flush=True)


if __name__=='__main__':main()
