"""Replay the exact old 240 cases with the current driving actor/executor.

These previously consumed cases are retention tests, never independent data.
Directional fixed-command baselines use matching physical settings and resets.
"""
import argparse
from dataclasses import fields
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from unified_steering.structured.driving_task import DrivingConfig, DrivingEnv
from unified_steering.structured.env import SteeringTeacherEnv
from unified_steering.structured.evaluate import load_teacher, run_case
from unified_steering.structured.legacy.regrasp.controller import RegraspControlConfig


class RetentionEnv(DrivingEnv):
    _make_commands = SteeringTeacherEnv._make_commands


def init(model_path, protocol, output):
    global _model, _config, _output
    torch.set_num_threads(1)
    _model = load_teacher(model_path)
    keys = {f.name for f in fields(DrivingConfig)} - {'controller'}
    settings = {k: v for k, v in protocol.items() if k in keys}
    _config = DrivingConfig(controller=RegraspControlConfig(**protocol.get('controller', {})), **settings)
    _output = Path(output)


def job(item):
    index, group, case = item
    args = dict(case=case['case'], seed=case['seed'], seconds=case['seconds'],
                amplitude=case['amplitude_deg'], support=case['support'],
                env_class=RetentionEnv, reset_options=case['reset_options'])
    record, rows = run_case(_model, _config, **args)
    record.update(group=group, hard_failure=bool(record['qp_failures'] or
        record['termination_reason'] or record['collision_peak_n'] > 1. or
        record['min_arm_margin_deg'] < 2.))
    record['passed'] = bool(record['passed'] and not record['hard_failure'])
    baseline = None
    if case['case'] in ('positive', 'negative'):
        baseline, _ = run_case(None, _config, fixed=True, **args)
        baseline['group'] = group
    np.savez_compressed(_output/'traces'/f'{index:03d}.npz', **{
        k: np.asarray([r[k] for r in rows]) for k in ('time','wheel_angle','target_angle',
        'error_deg','physics_peak_error_deg','physics_min_arm_margin_deg',
        'physics_collision_peak_n','free_force','qp_failures')})
    (_output/'cases'/f'{index:03d}.json').write_text(json.dumps(
        dict(record=record, baseline=baseline), indent=2)+'\n')
    return record, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--executor-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.executor_protocol.read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    jobs = []
    for group, path in [('nominal','results/r24_nominal_acceptance_v4/summary.json'),
                        ('screened','results/r24_screened_acceptance_v4/summary.json')]:
        jobs += [(len(jobs)+i, group, c) for i,c in enumerate(json.loads(Path(path).read_text())['cases'])]
    assert len(jobs) == 240
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('cases','traces'): (args.output/name).mkdir()
    shutil.copy2(__file__,args.output/'runner_snapshot.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(cases=jobs,
        executor=protocol, model=str(args.model), model_sha256=digest,
        independent=False, purpose='exact consumed240 with current finite physical executor'),indent=2)+'\n')
    records, baselines = [], []
    start = time.monotonic()
    with mp.get_context('fork').Pool(args.workers,initializer=init,
            initargs=(args.model,protocol,args.output)) as pool:
        for record, baseline in pool.imap(job,jobs):
            records.append(record)
            if baseline is not None: baselines.append(baseline)
            improvements = {}
            for support in ('both','left','right'):
                for direction in ('positive','negative'):
                    selected = [r for r in records if r['support']==support and r['case']==direction]
                    paired = [b for b in baselines if b['support']==support and b['case']==direction]
                    if selected:
                        assert len(selected)==len(paired)
                        improvements[support+'_'+direction] = 1-float(np.mean([c['rmse_deg'] for c in selected]))/float(np.mean([c['rmse_deg'] for c in paired]))
            result = dict(cases=records,baselines=baselines,count=len(records),
                passes=sum(c['passed'] for c in records),hard_failures=sum(c['hard_failure'] for c in records),
                directional_improvements=improvements, independent=False,model_sha256=digest,
                passed=len(records)==240 and all(c['passed'] for c in records) and all(v>=.3 for v in improvements.values()),
                wall_seconds=time.monotonic()-start)
            (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(count=result['count'],passes=result['passes'],hard_failures=result['hard_failures'],last_case={k:record[k] for k in ['case','seed','support','passed','rmse_deg','termination_reason']})),flush=True)
    assert hashlib.sha256(args.model.read_bytes()).hexdigest()==digest
    print('SUITE '+json.dumps({k:v for k,v in result.items() if k not in ('cases','baselines')}),flush=True)


if __name__=='__main__': main()
