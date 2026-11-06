"""Derived M3-0 evidence: paired errors, physical events and checkpoint checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from unified_steering.audit import ROOT, digest, write_json
from unified_steering.checkpoint import read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--monitor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    trials = {(r['group'], r['seed']): r for r in summary['noise_trials']}
    seeds = sorted({r['seed'] for r in summary['noise_trials']})
    comparisons = {}
    for group in ('initial_arm_low', 'initial_hand_low'):
        paired = [dict(seed=seed, delta_rmse_deg=trials[group,seed]['rmse_deg']-trials['initial_default',seed]['rmse_deg'],
                       both_complete=trials[group,seed]['safety_completion'] and trials['initial_default',seed]['safety_completion'])
                  for seed in seeds]
        common = [r['delta_rmse_deg'] for r in paired if r['both_complete']]
        comparisons[group] = dict(pairs=paired, observed_prefix_improved_pairs=sum(r['delta_rmse_deg']<0 for r in paired),
                                  common_complete_pairs=len(common), mean_common_complete_delta_rmse_deg=float(np.mean(common)))
    events = {}
    for name in ('neutral_hold', 'roundtrip_plus_first', 'roundtrip_minus_first'):
        with np.load(args.run/'frozen/final'/f'{name}.npz', allow_pickle=False) as trace:
            physics = trace['physics']
            fields = list(trace['physics_fields'])
            p = lambda field: physics[:,fields.index(field)]
            angle = np.rad2deg(p('wheel_angle_rad'))
            low = (p('left_contact_force_n')<=.2)&(p('right_contact_force_n')<=.2)
            after = p('time_s')>=8.
            bound = p('arm_bound_axes')>0
            events[name] = dict(max_angle_deg=float(angle.max()), max_angle_time_s=float(p('time_s')[angle.argmax()]),
                                end_angle_deg=float(angle[-1]), arm_lead_any_tick_fraction=float(np.mean(p('arm_lead_axes')>0)),
                                arm_bound_any_tick_fraction=float(np.mean(bound)),
                                first_arm_bound_time_s=float(p('time_s')[bound][0]) if np.any(bound) else None,
                                min_arm_margin_deg=float(np.rad2deg(p('min_arm_margin_rad').min())),
                                both_contact_low_fraction=float(np.mean(low)),
                                both_contact_low_after_8s_fraction=float(np.mean(low[after])) if np.any(after) else None)
    violations = {}
    for group in summary['noise_groups']:
        rows = [r for r in summary['noise_trials'] if r['group']==group and not r['safety_completion']]
        violations[group] = [dict(seed=r['seed'],reason=r['trace']['termination_reason'],
                                 joints=r['trace']['joint_limit_violations']) for r in rows]
    checkpoints = {p.name:read_checkpoint(p) for p in sorted(args.monitor.glob('*.pt'))}
    def same_actor(a, b):
        return all(torch.equal(checkpoints[a]['actor'][k], value) for k,value in checkpoints[b]['actor'].items())
    monitor = dict(steps_by_file={k:v['steps'] for k,v in checkpoints.items()},
                   best_matches_initial_actor=same_actor('best.pt','initial.pt'),
                   latest_matches_final_actor=same_actor('latest.pt','final.pt'),
                   numbered_final_matches_final_actor=same_actor('step_000001024.pt','final.pt'))
    old_checks = {}
    for name in ('m1_environment_v2','m2_smoke_seed7_v1'):
        manifest = json.loads((ROOT/'results'/name/'artifacts.json').read_text())
        check = {path:digest(ROOT/path)==expected for path,expected in manifest['sha256'].items()}
        check.update({value['path']:digest(Path(value['path']))==value['sha256'] for value in manifest.get('checkpoints',{}).values()})
        if not all(check.values()):
            raise RuntimeError('A frozen archive or checkpoint changed')
        old_checks[name] = dict(checked=len(check), all_unchanged=True)
    if not all((monitor['best_matches_initial_actor'],monitor['latest_matches_final_actor'],monitor['numbered_final_matches_final_actor'])):
        raise RuntimeError('Monitoring integration checkpoint mismatch')
    write_json(args.output,dict(paired_comparisons=comparisons,final_policy_events=events,
                               noise_failures=violations,monitor_checkpoint_checks=monitor,old_archive_checks=old_checks))


if __name__ == '__main__':
    main()
