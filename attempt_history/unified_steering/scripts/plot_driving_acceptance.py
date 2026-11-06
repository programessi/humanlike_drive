"""Export representative physical traces for an accepted or rejected suite."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    candidates=[]
    for path in (args.result/'cases').glob('*.json'):
        c=json.loads(path.read_text())
        if c['template']=='sequence' and c['complete']:
            candidates.append((c,path))
    chosen=[]
    for direction in (-1,1):
        selected=[r for r in candidates if r[0]['direction']==direction]
        if not selected:raise ValueError('Need complete sequence traces in both directions')
        chosen.append(max(selected,key=lambda r:r[0]['motion_rmse_deg']))
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(5,2,figsize=(13,11),sharex='col',layout='constrained')
    evidence=[]
    for col,(case,path) in enumerate(chosen):
        phase=case['case'][4].get('phase_deg',0.) if len(case['case'])>4 else 0.
        trace=np.load(args.result/'traces'/path.with_suffix('.npz').name)
        t=trace['time'];axes[0,col].set_title(f"Initial direction {case['direction']:+d}, seed {case['seed']}, offset {phase:g} deg\nMotion RMSE {case['motion_rmse_deg']:.3f} deg")
        axes[0,col].plot(t,trace['target_deg'],color='black',ls='--',label='Reference',lw=1.2)
        axes[0,col].plot(t,trace['angle_deg'],color='#009988',label='Measured',lw=1.1)
        axes[1,col].plot(t,trace['error_deg'],color='#009988',lw=1.)
        for side,color,name in [(0,'#0072B2','Left'),(1,'#D55E00','Right')]:
            axes[2,col].plot(t,trace['forces_n'][:,side],color=color,lw=.8,label=name)
            axes[3,col].plot(t,trace['torques_nm'][:,side],color=color,lw=.8,label=name)
        axes[3,col].plot(t,-trace['centering_nm'],color='black',ls='--',lw=1,label='Opposing centering load')
        axes[4,col].plot(t,trace['min_arm_margin_deg'],color='#0072B2',lw=1.)
        axes[4,col].axhline(2,color='#AA3377',ls='--',label='Safety threshold')
        rebases=np.flatnonzero(np.diff(trace['learned_internal_rebases'],prepend=0))
        for row in range(5):
            ax=axes[row,col];ax.grid(alpha=.18)
            for segment in case['timeline']['segments']:ax.axvspan(segment['end'],segment['hold_end'],color='grey',alpha=.1,zorder=-1)
            for i in rebases:ax.axvline(t[i],color='#AA3377',alpha=.45,lw=.8,ls=':')
        for row in (0,2,3,4):axes[row,col].legend(fontsize=8,loc='upper right')
        for row,label in enumerate(['Wheel angle (deg)','Tracking error (deg)','Hand contact force (N)','Axial contact torque (Nm)','Minimum arm margin (deg)']):axes[row,col].set_ylabel(label)
        axes[4,col].set_xlabel('External reference time (s)')
        evidence.append(dict(case=case['case'],case_path=str(path),motion_rmse_deg=case['motion_rmse_deg'],stop_peaks_deg=case['stop_peaks_deg'],internal_anchor_updates=[dict(time_s=float(t[i]),wheel_deg=float(trace['angle_deg'][i])) for i in rebases],both_hands_below_02N_samples=int(np.sum((trace['forces_n']<.2).all(1))),contact_sampling_hz=20))
    fig.suptitle('Learned policy: complete 180° round trips\nGrey: commanded holds; dotted: internal memory updates (no physical or clock reset)',fontsize=13)
    fig.savefig(args.output/'physical_sequences.png',dpi=180)
    fig.savefig(args.output/'physical_sequences.pdf')
    plt.close(fig)
    (args.output/'representative_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(args.output/'physical_sequences.png')


if __name__=='__main__':main()
