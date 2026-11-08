"""Run the frozen nominal simulation P0 regression cases into a fresh folder."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def cases():
    return [
        ('A_hold', 'p0_restart', ['--case', 'hold']),
        ('A_left_support', 'p0_motion', ['--case', 'release', '--angle', '1']),
        ('A_right_support', 'p0_motion', ['--case', 'release', '--angle', '-1']),
        *[(f'B_{direction}{degree}', 'p0_motion', ['--angle', str(sign*degree)])
          for degree in (30, 60, 90) for direction, sign in (('right', 1), ('left', -1))],
        ('D_right30', 'p0_motion', ['--case', 'handover', '--angle', '30']),
        ('D_left30', 'p0_motion', ['--case', 'handover', '--angle', '-30']),
        ('E_right15', 'p0_motion', ['--case', 'moving_handover', '--angle', '15']),
        ('E_left15', 'p0_motion', ['--case', 'moving_handover', '--angle', '-15']),
        ('F_positive70', 'p0_motion', ['--case', 'continuous', '--angle', '1', '--maximum', '70']),
        ('F_negative70', 'p0_motion', ['--case', 'continuous', '--angle', '-1', '--maximum', '70']),
        ('negative_no_contact', 'p0_motion', ['--angle', '30', '--negative-control']),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--only', nargs='+', help='explicit case names for a focused regression')
    args = parser.parse_args()
    selected = cases()
    if args.only:
        known = {c[0] for c in selected}
        if set(args.only)-known:
            parser.error(f'Unknown cases: {set(args.only)-known}')
        selected = [c for c in selected if c[0] in args.only]
    if args.output.exists():
        raise FileExistsError('Use a fresh suite directory to preserve all runs')
    args.output.mkdir(parents=True)
    (args.output/'suite.json').write_text(json.dumps(selected, indent=2)+'\n')

    def run(case):
        name, module, options = case
        output = args.output/name
        command = [sys.executable, '-m', 'humanlike_drive.env_runner.'+module,
                   '--output', str(output), '--grasp', 'assets/scenes/p0_clear_grasp_v1.json', *options]
        with (args.output/(name+'.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((output/'evaluation.json').read_text()) if (output/'evaluation.json').exists() else None
        accepted = bool(result and (result['passed'] if name != 'negative_no_contact' else
                                   not result['passed'] and not result['checks']['motion_RMSE_le_2deg']))
        print(f'{name}: process={process.returncode}, accepted={accepted}', flush=True)
        return dict(name=name, returncode=process.returncode, accepted=accepted,
                    physics_passed=result['passed'] if result else None)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, selected))
    (args.output/'suite_results.json').write_text(json.dumps(results, indent=2)+'\n')
    if not all(r['accepted'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
