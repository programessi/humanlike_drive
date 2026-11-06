"""Fixed 4.2 development grid: signed single moves and nominal hold regression."""
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from .audit import write_json
from .env import SteeringEnv


def definitions(settings):
    cases=[dict(name=f'hold_{seconds:g}s',sign=0,seconds=float(seconds),amplitude=0.,speed=1.)
           for seconds in settings['hold_seconds']]
    for sign,label in ((1,'positive'),(-1,'negative')):
        cases.extend(dict(name=f'{label}_{amplitude:g}deg_{speed:g}dps',sign=sign,seconds=None,
                          amplitude=float(amplitude),speed=float(speed))
                     for amplitude in settings['amplitude_deg'] for speed in settings['speed_deg_s'])
    return cases


@torch.no_grad()
def evaluate_commands(config,actor,output,*,diagnostics=False,fixed_report=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    settings=config['evaluation']
    cfg=copy.deepcopy(config['environment'])
    cfg.update(initial_support='both',episode_seconds=30.,diagnostics=False,wheel_disturbance={})
    env=SteeringEnv(cfg)
    device=next(actor.parameters()).device if actor is not None else 'cpu'
    if actor is not None:actor.eval()
    cases=[]
    for case in definitions(settings):
        env.config['reference']['speed_deg_s']=case['speed']
        env.config['reference']['hold_seconds']=1.
        obs,initial=env.reset(seed=4200,options={'support':'both'})
        theta=initial['initial_angle_rad'];delta=np.deg2rad(case['sign']*case['amplitude'])
        env.set_reference([theta,theta+delta] if case['sign'] else [theta])
        obs=env.encoder.build(env.reference,0.);env.last_obs=obs
        duration=case['seconds'] if not case['sign'] else env.reference.duration+settings['final_hold_seconds']
        count=int(np.ceil(duration/env.period-1e-8));env.config['episode_seconds']=count*env.period
        hidden=actor.initial_hidden(1) if actor is not None else None
        rows=[];interventions=ticks=0
        for step in range(count):
            if actor is None:action=env.hold_action()
            else:
                dist,hidden=actor.distribution(torch.as_tensor(obs['actor'],device=device).view(1,1,-1),hidden,
                                               torch.tensor([[step==0]],device=device))
                action=dist.deterministic()[0,0].cpu().numpy()
            obs,reward,term,trunc,info=env.step(action)
            if not info['observation_valid']:raise FloatingPointError('Invalid command evaluation state')
            rows.append(dict(time_s=float(env.sim.data.time),reference_deg=float(np.rad2deg(info['reference_angle_rad'])),
                             angle_deg=float(np.rad2deg(info['wheel_angle_rad'])),error_deg=float(np.rad2deg(info['error_rad'])),reward=reward))
            interventions+=info['execution_interventions'];ticks+=info['physics_steps']
            if term or trunc:break
        complete=bool(not term and len(rows)==count)
        errors=np.array([r['error_deg'] for r in rows]);rmse=float(np.sqrt(np.mean(errors**2)))
        tail=[r for r in rows if r['time_s']>=count*env.period-settings['final_hold_seconds']-1e-8] if complete else []
        endpoint=max(abs(r['error_deg']) for r in tail) if tail else None
        progress=case['sign']*float(np.mean([r['angle_deg'] for r in tail])-np.rad2deg(theta)) if tail and case['sign'] else None
        fraction=progress/case['amplitude'] if progress is not None else None
        direction_correct=bool(complete and fraction is not None and fraction>=settings['min_signed_progress_fraction']) if case['sign'] else None
        peak=float(np.max(np.abs(errors)))
        success=bool(complete and (rmse<=settings['rmse_gate_deg'] and endpoint<=settings['endpoint_gate_deg'] and direction_correct
                                   if case['sign'] else peak<=settings['hold_peak_gate_deg']))
        filename=case['name']+'.csv'
        with (output/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        cases.append(dict(name=case['name'],sign=case['sign'],amplitude_deg=case['amplitude'],speed_deg_s=case['speed'],
                          rmse_deg=rmse,peak_error_deg=peak,endpoint_peak_error_deg=endpoint,
                          signed_progress_deg=progress,signed_progress_fraction=fraction,direction_correct=direction_correct,
                          safety_completion=complete,tracking_success=success,termination_reason=info['termination_reason'],
                          steps=len(rows),intervention_fraction=interventions/max(ticks,1),trace=filename))
    directions={}
    for sign,label in ((1,'positive'),(-1,'negative')):
        rows=[r for r in cases if r['sign']==sign]
        baseline=[r for r in fixed_report['cases'] if r['sign']==sign] if fixed_report else None
        mean=float(np.mean([r['rmse_deg'] for r in rows]))
        improvement=1-mean/np.mean([r['rmse_deg'] for r in baseline]) if baseline else None
        directions[label]=dict(mean_rmse_deg=mean,worst_rmse_deg=max(r['rmse_deg'] for r in rows),
                               safety_completion_rate=float(np.mean([r['safety_completion'] for r in rows])),
                               tracking_success_rate=float(np.mean([r['tracking_success'] for r in rows])),
                               direction_correct_rate=float(np.mean([r['direction_correct'] for r in rows])),
                               mean_signed_progress_fraction=float(np.mean([r['signed_progress_fraction'] for r in rows if r['signed_progress_fraction'] is not None])) if any(r['signed_progress_fraction'] is not None for r in rows) else None,
                               baseline_improvement=float(improvement) if improvement is not None else None)
    holds=[r for r in cases if not r['sign']]
    qualified=bool(all(r['tracking_success'] for r in cases) and fixed_report is not None and
                   all(d['baseline_improvement']>=settings['min_baseline_improvement'] for d in directions.values()))
    report=dict(kind='M3_B_fixed_development_not_independent_acceptance',cases=cases,directions=directions,
                all_cases_complete=all(r['safety_completion'] for r in cases),
                tracking_success_rate=float(np.mean([r['tracking_success'] for r in cases])),
                safety_completion_rate=float(np.mean([r['safety_completion'] for r in cases])),
                hold_mean_rmse_deg=float(np.mean([r['rmse_deg'] for r in holds])),
                hold_worst_peak_deg=max(r['peak_error_deg'] for r in holds),
                development_qualified=qualified,independent_stage_qualified=False,
                signed_progress_gate_fraction=settings['min_signed_progress_fraction'])
    write_json(output/'summary.json',report)
    return report


def command_selection_score(report,anchor):
    holds=[r for r in report['cases'] if not r['sign']]
    initial={r['name']:r['rmse_deg'] for r in anchor['cases'] if not r['sign']}
    eligible=bool(report['all_cases_complete'] and report['hold_worst_peak_deg']<=1.
                  and all(r['rmse_deg']<=initial[r['name']]+.5 for r in holds))
    ds=list(report['directions'].values())
    score=(min(d['tracking_success_rate'] for d in ds),min(d['baseline_improvement'] for d in ds),
           -max(d['worst_rmse_deg'] for d in ds),-float(np.mean([d['mean_rmse_deg'] for d in ds])))
    return eligible,score
