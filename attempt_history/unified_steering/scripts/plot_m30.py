"""Export frozen M3-0 comparisons, explicitly marking truncated trials."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    for suffix in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run/'summary.json').read_text())
    groups = summary['noise_groups']
    names = list(groups)
    labels = [name.replace('initial_', 'Init: ').replace('final_default', 'Final: default').replace('_', ' ')
              for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].bar(x-.18, [groups[n]['safety_completion_rate'] for n in names], .36,
                label='No hard failure for full 3 s', color='#277DA1')
    axes[0].bar(x+.18, [groups[n]['hold_success_rate'] for n in names], .36,
                label='Full 3 s and peak error <= 2 deg', color='#43AA8B')
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel('Fraction of 10 paired trials')
    axes[0].legend(fontsize=8, frameon=False)
    rows = summary['noise_trials']
    seeds = sorted({r['seed'] for r in rows})
    for seed in seeds:
        trial = [next(r for r in rows if r['group']==name and r['seed']==seed) for name in names]
        axes[1].plot(x, [r['rmse_deg'] for r in trial], color='gray', alpha=.3, linewidth=.8)
        for index, row in enumerate(trial):
            axes[1].scatter(index, row['rmse_deg'], marker='o' if row['safety_completion'] else 'x',
                            color='#277DA1' if row['safety_completion'] else '#F94144', s=28)
    axes[1].plot([], [], 'o', color='#277DA1', label='Complete trial')
    axes[1].plot([], [], 'x', color='#F94144', label='Early failure: observed prefix only')
    axes[1].set_ylabel('Wheel error RMSE (deg)')
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha='right')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen policies: exploration scale affects physical holding')
    save(fig, args.output/'noise_comparison')

    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex='col', constrained_layout=True)
    for col, controller in enumerate(('initial', 'final')):
        with np.load(args.run/'frozen'/controller/'roundtrip_plus_first.npz', allow_pickle=False) as trace:
            physics = trace['physics']
            fields = list(trace['physics_fields'])
            p = lambda name: physics[:, fields.index(name)]
            axes[0, col].plot(trace['time_s'], np.rad2deg(trace['reference_rad']), '--', color='black', label='Reference')
            axes[0, col].plot(p('time_s'), np.rad2deg(p('wheel_angle_rad')), color='#277DA1', label='Actual')
            axes[0, col].set_title(controller.capitalize()+' actor: positive-first roundtrip')
            axes[0, col].set_ylabel('Wheel angle (deg)')
            axes[1, col].plot(p('time_s'), p('left_contact_force_n'), label='Left', color='#277DA1')
            axes[1, col].plot(p('time_s'), p('right_contact_force_n'), label='Right', color='#F9844A')
            axes[1, col].set_ylabel('Sum of contact force norms (N)')
            axes[2, col].plot(p('time_s'), p('arm_lead_axes'), label='Arm lead', color='#F94144')
            axes[2, col].plot(p('time_s'), p('arm_bound_axes'), label='Arm joint bound', color='#9B5DE5')
            axes[2, col].plot(p('time_s'), p('hand_slew_axes'), label='Hand slew', color='#43AA8B', alpha=.7)
            axes[2, col].set_ylabel('Affected drive axes / tick')
            axes[2, col].set_xlabel('Time (s)')
            for ax in axes[:, col]:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Contact-force proxies and target processing, sampled at 500 Hz')
    save(fig, args.output/'roundtrip_telemetry')

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for row, controller in enumerate(('initial', 'final')):
        probes = [r for r in summary['direction_probes'] if r['controller']==controller]
        for col, probe in enumerate(probes):
            ax = axes[row, col]
            if not probe['valid']:
                ax.set_title('Invalid anchor: '+probe['anchor'])
                continue
            for branch, color in zip(probe['branches'], ('#277DA1', '#F9844A')):
                with np.load(args.run/'probes'/branch['trace_file'], allow_pickle=False) as trace:
                    metadata = json.loads(str(trace['metadata_json']))
                    t = trace['time_s']-metadata['time_origin_s']
                    fields = list(trace['physics_fields'])
                    p = trace['physics']
                    pt = p[:, fields.index('time_s')]-metadata['time_origin_s']
                    delta = np.rad2deg(p[:, fields.index('wheel_angle_rad')])-probe['anchor_angle_deg']
                    ref = np.rad2deg(trace['reference_rad'])-probe['anchor_angle_deg']
                    ax.plot(t, ref, '--', color=color, alpha=.6)
                    ax.plot(pt, delta, color=color, label=f"{branch['sign']:+d} reference")
            ax.set_title(controller+': '+probe['anchor'], fontsize=10)
            ax.set_xlabel('Time since cloned state (s)')
            ax.set_ylabel('Wheel change (deg)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Same physical state and GRU history; signed smooth-reference branches')
    save(fig, args.output/'direction_probes')


if __name__ == '__main__':
    main()
