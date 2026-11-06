"""Standalone plots of M2 engineering metrics and complete/failed development traces."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, path):
    for suffix in ('.png','.pdf'):
        fig.savefig(path.with_suffix(suffix),dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows = read_csv(args.run/'updates.csv')
    x = np.array([float(row['steps'])/1000 for row in rows])
    fig,axes = plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for ax,key,label in ((axes[0,0],'mean_reward','Mean reward per control step'),
                         (axes[0,1],'approx_kl','Approximate KL (updated batches)'),
                         (axes[1,0],'value_loss','Critic half MSE')):
        y = np.array([float(row[key]) for row in rows])
        ax.plot(x,y,color='#277DA1',alpha=.4,linewidth=1)
        if len(y) >= 10:
            ax.plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'),color='#277DA1',label='10-update mean')
        ax.set_ylabel(label)
    axes[0,1].axhline(.015,color='#F9844A',linestyle='--',label='KL target')
    failures = np.array([int(row['failure_count']) for row in rows])
    steps = np.array([int(row['steps']) for row in rows])
    rate = np.diff(np.r_[0,failures])/np.diff(np.r_[0,steps])*1000
    axes[1,1].plot(x,rate,color='#F9844A',alpha=.8)
    axes[1,1].set_ylabel('Safety terminations per 1,000 steps')
    for ax in axes.ravel():
        ax.set_xlabel('Environment interaction steps (thousands)')
        ax.grid(alpha=.2)
    axes[0,1].legend(frameon=False,fontsize=8)
    fig.suptitle('M2 engineering smoke: training metrics are not task success')
    save(fig,args.output/'training')

    for name in ('roundtrip_plus_first','offset_plus5'):
        fig,axes = plt.subplots(2,1,figsize=(8,5),sharex=True,constrained_layout=True)
        baseline = read_csv(args.run/'evaluation_fixed_targets'/f'{name}.csv')
        axes[0].plot([float(r['time_s']) for r in baseline],[float(r['reference_deg']) for r in baseline],
                     '--',color='black',linewidth=1.8,label='Time reference')
        for directory,label,color in (('evaluation_initial','Initial actor','#277DA1'),
                                      ('evaluation_final','Trained actor','#F94144'),
                                      ('evaluation_fixed_targets','Fixed joint targets','#43AA8B')):
            trace = read_csv(args.run/directory/f'{name}.csv')
            summary = json.loads((args.run/directory/'summary.json').read_text())
            case = next(case for case in summary['cases'] if case['name'] == name)
            times = [float(row['time_s']) for row in trace]
            if not case['safety_completion']:
                label += f" (failed at {times[-1]:.2f}s)"
            for ax,key in zip(axes,('angle_deg','error_deg')):
                values = [float(row[key]) for row in trace]
                ax.plot(times,values,color=color,linewidth=1.5,label=label)
                if not case['safety_completion']:
                    ax.plot(times[-1],values[-1],marker='x',color=color,markersize=7)
        axes[0].set_ylabel('Accumulated wheel angle (deg)')
        axes[1].set_ylabel('Reference minus actual (deg)')
        axes[1].set_xlabel('Time (s)')
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle(f'M2 fixed development test: {name}')
        save(fig,args.output/name)


if __name__ == '__main__':
    main()
