"""Standalone frozen holding and disturbance qualification plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run/'summary.json').read_text())
    rows=summary['deterministic_cases']
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for controller,color in (('initial','#277DA1'),('fixed_targets','#43AA8B')):
        baseline=next(r for r in rows if r['controller']==controller and r['name']=='hold_10s')
        with np.load(args.run/'traces'/baseline['trace_file'],allow_pickle=False) as b:
            axes[0,0].plot(b['time_s'],np.rad2deg(b['error_rad']),label=controller,color=color)
            field=list(b['physics_fields']).index('wheel_angle_rad')
            for name,style in (('pulse_plus','-'),('pulse_minus','--')):
                row=next(r for r in rows if r['controller']==controller and r['name']==name)
                with np.load(args.run/'traces'/row['trace_file'],allow_pickle=False) as a:
                    delta=np.rad2deg(a['physics'][:,field]-b['physics'][:,field])
                    axes[0,1].plot(a['physics'][:,0],delta,style,label=controller+' '+name,color=color)
    axes[0,0].set(title='Nominal 10 s: frozen actor and fixed targets',ylabel='Target minus wheel angle (deg)',xlabel='Time (s)')
    axes[0,1].axvspan(2.,2.2,color='gray',alpha=.2,label='Pulse interval')
    for level in (-.25,.25):axes[0,1].axhline(level,color='black',alpha=.4,linestyle=':')
    axes[0,1].set(title='Pulse deviation is below the 0.25 deg review gate',ylabel='Pulse minus no-pulse wheel angle (deg)',xlabel='Time (s)')
    for ax,seconds in zip(axes[1],(3.,10.)):
        group=[r for r in summary['noise_trials'] if r['seconds']==seconds]
        for seed in sorted({r['seed'] for r in group}):
            pair=[next(r for r in group if r['seed']==seed and r['name']==name) for name in ('default','arm_low')]
            ax.plot([0,1],[r['rmse_deg'] for r in pair],color='gray',alpha=.3)
            for x,r in enumerate(pair):
                ax.scatter(x,r['rmse_deg'],color='#277DA1' if r['complete'] else '#F94144',marker='o' if r['complete'] else 'x',s=30)
        ax.plot([],[],'o',color='#277DA1',label='Complete trial')
        ax.plot([],[],'x',color='#F94144',label='Failed: prefix RMSE only')
        ax.set_xticks([0,1],['Arm std 0.35','Arm std 0.10'])
        ax.set(title=f'{seconds:g} s paired exploration (hand std 0.50)',ylabel='Observed error RMSE (deg)')
    for ax in axes.ravel():ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Step 4.1 frozen baseline: nominal holding does not qualify active recovery')
    for suffix in ('.png','.pdf'):fig.savefig(args.output/('holding_baseline'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
