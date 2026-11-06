"""Audit existing R0--R2 evidence; never run, train, or change a policy."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/home/nubot/humanlike_drive/unified_steering')
REPORT = ROOT / 'results/DRIVING_R0_R2_ACCEPTANCE_V121'
FROZEN = ROOT / 'runs/driving_r2_frozen_v121'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


digest = sha(FROZEN / 'teacher.zip')
manifest = read(FROZEN / 'manifest.json')
acceptance = read(FROZEN / 'ACCEPTANCE.json')
aliases = read(FROZEN / 'audit_snapshot.json')['historical_hash_aliases']
require(digest == manifest['model_sha256'] == acceptance['model_sha256'], 'model hash')
for relative, expected in manifest['files'].items():
    require(sha(ROOT / aliases.get(relative, relative)) == expected, relative)
source_count = 0
for frozen in (FROZEN / 'source_snapshot').rglob('*.py'):
    relative = frozen.relative_to(FROZEN / 'source_snapshot')
    current = ROOT / ('scripts' if relative.name == 'evaluate_driving_retention.py'
                      else 'src/unified_steering/structured') / relative
    require(frozen.read_bytes() == current.read_bytes(), f'runtime source: {relative}')
    source_count += 1

pipeline = read(ROOT / 'runs/driving_r2_acceptance_pipeline_v121.json')
require(pipeline['status'] == 'complete_all_gates_passed', 'pipeline incomplete')
suites = {}
for stage in pipeline['stages']:
    directory = ROOT / stage['output']
    summary, protocol = read(directory / 'summary.json'), read(directory / 'protocol.json')
    cases = summary['cases']
    require(stage['returncode'] == 0 and summary['passed'], stage['name'])
    require(len(cases) == stage['count'] == summary['count'], 'case count')
    require(all(c['passed'] and c['complete'] and not c['hard_failure'] for c in cases),
            f"failed case: {stage['name']}")
    require(all(not c['termination_reason'] and c['collision_peak_n'] <= 1
                and c['qp_failures'] == 0 and c['min_arm_margin_deg'] >= 2 for c in cases),
            f"safety: {stage['name']}")
    require(len(list((directory / 'traces').glob('*.npz'))) == len(cases), 'trace count')
    require(len(list((directory / 'cases').glob('*.json'))) == len(cases), 'case files')
    if stage['name'] == 'old240_retention':
        require(summary['model_sha256'] == digest, 'old retention model')
        require(min(summary['directional_improvements'].values()) >= .3,
                'old directional improvement')
    else:
        metadata = read(directory / 'metadata.json')
        require(metadata['sha256'] == digest and not metadata['collection']
                and not metadata['script_at_learned_evaluation'], 'policy-only evaluation')
        require(sorted(json.dumps(c['case'], sort_keys=True) for c in cases)
                == sorted(json.dumps(c, sort_keys=True) for c in protocol['cases']),
                'protocol coverage')
    suites[stage['name']] = dict(
        cases=len(cases), complete=sum(c['complete'] for c in cases),
        passed=sum(c['passed'] for c in cases), hard_failures=0,
        min_arm_margin_deg=min(c['min_arm_margin_deg'] for c in cases),
        source=str((directory / 'summary.json').relative_to(ROOT)))

contact_audits = {}
for suite, folder in [('r1', 'driving_r1_exact72_retention_v121'),
                      ('r2', 'driving_r2_independent_v121')]:
    directory = ROOT / 'results' / folder
    records, metric_differences = [], []
    for file in sorted((directory / 'cases').glob('*.json')):
        case = read(file)
        with np.load(directory / 'traces' / f'{file.stem}.npz') as trace:
            time, error = trace['time'], trace['error_deg']
            moving, peaks = np.zeros(len(time), dtype=bool), []
            for leg in case['timeline']['segments']:
                moving |= (time >= leg['start']) & (time <= leg['end'])
                mask = (time >= leg['hold_end'] - 3) & (time <= leg['hold_end'])
                peaks.append(float(trace['physics_peak_error_deg'][mask].max()))
            rmse = float(np.sqrt(np.mean(error[moving] ** 2)))
            rmse_difference = abs(rmse - case['motion_rmse_deg'])
            peak_difference = float(np.max(np.abs(np.asarray(peaks) - case['stop_peaks_deg'])))
            metric_differences.append(max(rmse_difference, peak_difference))
            require(rmse_difference < 1e-12, 'RMSE reproduction')
            require(peak_difference < 1e-12,
                    'stop peak reproduction')
            require(rmse <= 1 and max(peaks) <= 1, 'precision')
            require(not np.any(trace['expert_stage'] != '')
                    and not np.any(trace['expert_mode'] != ''), 'runtime expert')
            force, torque, centering = trace['forces_n'], trace['torques_nm'], trace['centering_nm']
            release = (force < .2) & (force[:, ::-1] > 1)
            # Passive diagnostics only. These do not alter frozen acceptance gates.
            loaded = release & (abs(centering[:, None]) > .1) & (
                torque[:, ::-1] * np.sign(-centering[:, None]) > .05)
            records.append(dict(case=case['case'],
                both_contact_fraction=float(np.mean(np.all(force > 1, axis=1))),
                release_seconds=(release.sum(0) * .05).tolist(),
                loaded_release_seconds=(loaded.sum(0) * .05).tolist(),
                both_below_02N_samples=int(np.sum(np.all(force < .2, axis=1))),
                reproduced_motion_rmse_deg=rmse, reproduced_stop_peaks_deg=peaks))
    if suite == 'r1':
        require(all(c['both_contact_fraction'] == 1 for c in records), 'R1 both-hand contact')
    else:
        require(all(min(c['loaded_release_seconds']) >= .25 for c in records),
                'R2 loaded release evidence')
        require(not sum(c['both_below_02N_samples'] for c in records), 'both hands detached')
    contact_audits[suite] = dict(
        cases=len(records),
        both_contact_fraction_min=min(c['both_contact_fraction'] for c in records),
        loaded_release_cases_by_hand=[sum(c['loaded_release_seconds'][s] > .25
                                         for c in records) for s in range(2)],
        min_loaded_release_seconds_by_hand=[min(c['loaded_release_seconds'][s]
                                               for c in records) for s in range(2)],
        both_below_02N_samples=sum(c['both_below_02N_samples'] for c in records),
        raw_metric_reproduction_max_error=max(metric_differences), details=records)

log = (ROOT / 'outputs/driving_full_tests_v121.log').read_text()
require('Ran 126 tests' in log and log.rstrip().endswith('OK'), 'engineering tests')
config = read(FROZEN / 'config.json')
require(config['action_dim'] == 24 and not config['script_at_execution']
        and not config['physical_or_clock_reset'], 'unified 24-action policy')
require(config['task']['strict_safety'] and config['task']['centering_stiffness'] == .3,
        'safety and centering settings')
require((ROOT / 'results/DRIVING_R2_REJECTED_V119/README.md').is_file(), 'failure history')

requirements = [
    dict(requirement='R0 physical range, interfaces, sign, initial grasp and feasibility audit',
         status='proved', evidence=['results/DRIVING_R0_AUDIT_V1/README.md',
                                   'docs/RESEARCH_PLAN.md: R0 and R2 physical-entry records']),
    dict(requirement='One frozen 24-action wrist/hand policy using the physical executor',
         status='proved', evidence=['runs/driving_r2_frozen_v121/config.json',
                                   'src/unified_steering/structured/driving_path_basis.py',
                                   'src/unified_steering/structured/driving_learning.py']),
    dict(requirement='R1 0--90deg bidirectional both-hand tracking, holds and controlled return',
         status='proved', evidence=['results/driving_r1_exact72_retention_v121',
                                   'contact_audits.r1']),
    dict(requirement='R2 0--180deg load-bearing/release coordination and controlled return',
         status='proved', evidence=['results/driving_r2_independent_v121',
                                   'contact_audits.r2']),
    dict(requirement='Preserved physical safety, centering load, joint and actuator limits',
         status='proved', evidence=['src/unified_steering/structured/env.py:_physics_step',
                                   'runs/driving_r2_frozen_v121/config.json',
                                   'all six suite case records and physical-step metrics']),
    dict(requirement='Two complete development rounds before freeze and independent evaluation',
         status='proved', evidence=['runs/driving_r2_acceptance_pipeline_v121.json',
                                   'runs/driving_r2_frozen_v121/manifest.json',
                                   'results/driving_r2_independent_v121/metadata.json']),
    dict(requirement='Same-weight R1 and exact old 240-case capability retention',
         status='proved', evidence=['results/driving_r1_exact72_retention_v121',
                                   'results/driving_r2_old240_retention_v121']),
    dict(requirement='Preserve result, failed-candidate, model and plan artifacts',
         status='proved', evidence=['docs/RESEARCH_PLAN.md',
                                   'results/DRIVING_R2_REJECTED_V119/README.md',
                                   'results/DRIVING_R0_R2_ACCEPTANCE_V121/README.md',
                                   'runs/driving_r2_frozen_v121']),
]
result = dict(status='all_active_goal_requirements_proved',
    audited_utc=datetime.now(timezone.utc).isoformat(), model_sha256=digest,
    model_and_protocol_manifest_files_verified=len(manifest['files']),
    runtime_source_files_verified=source_count,
    total_suite_cases=sum(x['cases'] for x in suites.values()), suites=suites,
    scope='Planned R0--R2: <=90deg and nominal180deg early load-transfer variant, '
          'fixed executable reference sequences at 5deg/s, nominal and reset-preturned2deg.',
    method='Demonstration-assisted structured mixed actor plus IK/QP; not autonomous PPO discovery.',
    exclusions=['Original fixed-upper-hand-to180 human-style variant',
                'Unvalidated arbitrary90--180deg endpoints or online command changes',
                '360deg/540deg and hardware/history-student deployment'],
    transient_error_disclosure=dict(independent_full_physics_peak_deg=5.199953734907234,
        criterion='Motion RMSE and final3s stop peaks <=1deg; no all-time1deg claim.'),
    engineering_tests=126, requirements=requirements, contact_audits=contact_audits,
    passive_contact_diagnostic=dict(sample_hz=20, released_hand_force_below_n=.2,
        other_hand_force_above_n=1., abs_centering_torque_above_nm=.1,
        other_hand_opposing_centering_torque_above_nm=.05,
        acceptance_gate_changed=False, physics_or_policy_run=False))
destination = REPORT / 'completion_audit.json'
destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
runner = REPORT / 'completion_audit_runner.py'
if Path(__file__).resolve() != runner.resolve():
    shutil.copy2(__file__, runner)
print(json.dumps({k: result[k] for k in ['status', 'model_sha256',
    'model_and_protocol_manifest_files_verified', 'runtime_source_files_verified',
    'total_suite_cases', 'engineering_tests']}, ensure_ascii=False))
print(destination)
