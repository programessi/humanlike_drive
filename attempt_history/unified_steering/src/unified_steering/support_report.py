"""Archive S00-B evidence, curate validated reset states and plot measured traces."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from .audit import ROOT, digest, write_json


def traces(path):
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    return {key:np.array([float(r[key]) for r in rows]) for key in (
        "time_s","wheel_angle_deg","reference_angle_deg","tracking_error_deg","free_hand_wheel_gap_m")}


def run(open_loop,feedback,jitter,output,states):
    # Run only after all input processes have finished writing their summaries.
    folders={"open_loop":Path(open_loop),"feedback":Path(feedback),"jitter":Path(jitter)}
    reports={name:json.loads((folder/'summary.json').read_text()) for name,folder in folders.items()}
    for report in reports.values():
        if 'counts' not in report:
            raise ValueError('Input run is incomplete')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,folder in folders.items():
        target=output/name;target.mkdir()
        for filename in ('summary.json','config.yaml'):
            shutil.copy2(folder/filename,target/filename)
    # A small initial-state library, not a claim of broad grasp generalization.
    states=Path(states);states.mkdir(parents=True,exist_ok=False)
    manifest={"kind":"physically_validated_parked_single_hand_initial_states",
              "source_run":str(folders['jitter'].resolve()),
              "source_summary_sha256":digest(folders['jitter']/'summary.json'),
              "scene_sha256":reports['jitter']['scene_provenance']['snapshot_sha256'],
              "config_sha256":reports['jitter']['config_sha256'],"states":[],
              "restore_notes":"Restore qpos/qvel/act/target, reset time and external forces, then mj_forward. Never restore the preparation jig. qacc_warmstart is saved for reproducibility.",
              "scope":"Only nominal wheel load, +30 degree wheel initial states, +/-2 degree free-arm joint jitter; fixed support grasp."}
    for case in reports['jitter']['cases']:
        if not case['passed']:
            continue
        source=folders['jitter']/case['initial_state'];target=states/source.name
        shutil.copy2(source,target)
        manifest['states'].append({"file":target.name,"sha256":digest(target),"support":case['support'],
                                   "requested_angle_deg":case['requested_angle_deg'],
                                   "actual_angle_deg":case['preparation']['actual_angle_deg'],
                                   "free_arm_jitter_rad":case['joint_jitter_rad'],
                                   "validation_trace":str((folders['jitter']/case['trace']).resolve()),
                                   "hold_max_error_deg":case['max_abs_error_deg']})
    write_json(states/'manifest.json',manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for side,ax in zip(('left','right'),axes[0]):
        for name,color,label in (('open_loop','tab:red','Open loop'),('feedback','tab:blue','Bounded feedback script')):
            data=traces(folders[name]/f'motion_{side}_+30deg.csv')
            initial=data['reference_angle_deg'][0] # first desired increment is negligible
            ax.plot(data['time_s'],data['wheel_angle_deg']-initial,color=color,label=label)
            if name=='open_loop':
                ax.plot(data['time_s'],data['reference_angle_deg']-initial,'k--',label='Time reference')
        ax.set(title=f'{side.capitalize()} support: motion at +30 deg',xlabel='Time (s)',ylabel='Relative wheel angle (deg)')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    ax=axes[1,0]
    for name,style in (('open_loop','--'),('feedback','-')):
        for side,color in (('left','tab:blue'),('right','tab:orange')):
            data=traces(folders[name]/f'release_{side}_+30deg.csv')
            ax.plot(data['time_s'],data['tracking_error_deg'],style,color=color,label=f'{side}: {name}')
    ax.axhline(3,color='gray',lw=.8);ax.axhline(-3,color='gray',lw=.8)
    ax.set(title='Release at +30 deg: tracking error',xlabel='Time (s)',ylabel='Error (deg)')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    ax=axes[1,1]
    for side,color in (('left','tab:blue'),('right','tab:orange')):
        data=traces(folders['feedback']/f'release_{side}_+30deg.csv')
        ax.plot(data['time_s'],100*data['free_hand_wheel_gap_m'],color=color,label=f'{side} support')
    ax.axhline(2,color='gray',ls='--',label='Parked gap threshold')
    ax.axvline(3,color='gray',ls=':');ax.axvline(6,color='gray',ls=':')
    ax.set(title='Released hand: minimum wheel clearance',xlabel='Time (s)',ylabel='Gap (cm)')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('S00-B scripted physical diagnostics — no learned policy')
    fig.savefig(output/'overview.png',dpi=160);fig.savefig(output/'overview.pdf');plt.close(fig)
    summary={name:report['counts'] for name,report in reports.items()}
    summary['curated_initial_states']=len(manifest['states'])
    summary['all_cases_meet_current_gates']=all(report['all_executed_cases_passed'] for report in reports.values())
    write_json(output/'comparison.json',summary)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-loop',type=Path,required=True)
    parser.add_argument('--feedback',type=Path,required=True)
    parser.add_argument('--jitter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--states',type=Path,required=True)
    a=parser.parse_args();run(a.open_loop,a.feedback,a.jitter,a.output,a.states)


if __name__=='__main__':
    main()
