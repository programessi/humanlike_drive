"""M3-B first segment: reward penalties, KL guards and signed evaluation outcomes."""
import argparse
import csv
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
    with (args.run/'updates.csv').open() as stream:rows=list(csv.DictReader(stream))
    x=np.array([int(r['steps'])/1000 for r in rows])
    reward=np.array([float(r['mean_reward']) for r in rows])
    fails=np.diff(np.r_[0,[int(r['failure_count']) for r in rows]])
    failure_penalty=-20*fails/512
    records=json.loads((args.run/'development_index.json').read_text())['records']
    ex=np.array([r['steps']/1000 for r in records])
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    axes[0,0].plot(x,reward,label='Logged total reward',color='#277DA1')
    axes[0,0].plot(x,reward-failure_penalty,label='Without terminal failure penalty',color='#43AA8B',alpha=.7)
    axes[0,0].scatter(x[fails>0],reward[fails>0],color='#F94144',marker='x',label='Batch with hard failure')
    axes[0,0].set_ylabel('Mean reward per control call')
    axes[0,1].plot(x,[float(r['value_loss']) for r in rows],color='#277DA1',label='Critic half MSE')
    stops=np.array([r['kl_early_stop']=='True' for r in rows])
    axes[0,1].scatter(x[stops],np.zeros(stops.sum()),marker='|',color='#F9844A',label='KL guard: rest of update skipped')
    axes[0,1].set_ylabel('Value loss')
    for direction,color in (('positive','#277DA1'),('negative','#F9844A')):
        axes[1,0].plot(ex,[r['directions'][direction]['mean_rmse_deg'] for r in records],'-o',color=color,label=direction)
        axes[1,1].plot(ex,[r['directions'][direction]['mean_signed_progress_fraction'] for r in records],'-o',color=color,label=direction)
    axes[1,0].axhline(1.,linestyle=':',color='black',label='1 deg RMSE gate')
    axes[1,0].set_ylabel('Development mean RMSE by direction (deg)')
    axes[1,1].axhline(0.,linestyle=':',color='black')
    axes[1,1].axhline(1.,linestyle='--',color='gray',label='Full target displacement')
    axes[1,1].set_ylabel('Signed progress / target amplitude')
    for ax in axes.ravel():
        ax.set_xlabel('Interaction steps (thousands)');ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('4.2 first segment: no stable bidirectional tracking after 51,200 steps')
    for suffix in ('.png','.pdf'):fig.savefig(args.output/('training_diagnosis'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
