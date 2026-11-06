"""TensorBoard metrics with per-direction development results and weight histograms."""
import csv
from numbers import Real

import numpy as np
import torch


class TrainingDashboard:
    def __init__(self,output,config):
        self.writer=None
        self.histogram_interval=config.get('tensorboard',{}).get('histogram_every_updates',20)
        if config.get('tensorboard',{}).get('enabled',False):
            from torch.utils.tensorboard import SummaryWriter
            self.writer=SummaryWriter(str(output/'tensorboard'),flush_secs=5)
            self.writer.add_text('experiment/config',str(config),0)
            self.writer.flush()

    def update(self,row,actor,critic,algorithm,episodes):
        if self.writer is None:return
        step=row['steps']
        for key,value in row.items():
            if isinstance(value,Real):self.writer.add_scalar('training/'+key,value,step)
        self.writer.add_scalar('optimizer/actor_learning_rate',algorithm.actor_optimizer.param_groups[0]['lr'],step)
        self.writer.add_scalar('optimizer/critic_learning_rate',algorithm.critic_optimizer.param_groups[0]['lr'],step)
        for name,module in (('state_encoder',actor.state_encoder),('reference_encoder',actor.reference_encoder),('gru',actor.gru),('action_head',actor.action_head)):
            norm=sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5
            self.writer.add_scalar('gradients/'+name,norm,step)
        for task in ('hold','positive','negative'):
            rows=[e for e in episodes if e['task']==task]
            if rows:
                self.writer.add_scalar(f'episodes/{task}/rmse_deg',np.mean([r['rmse_deg'] for r in rows]),step)
                self.writer.add_scalar(f'episodes/{task}/safety_completion_rate',np.mean([not r['terminated'] for r in rows]),step)
        if row['update']%self.histogram_interval==0:
            for kind,module in (('actor',actor),('critic',critic)):
                for name,p in module.named_parameters():self.writer.add_histogram(f'parameters/{kind}/{name}',p.detach().cpu(),step)
        self.writer.flush()

    def evaluation(self,report,directory,steps):
        if self.writer is None or report is None:return
        for key,value in report.items():
            if isinstance(value,Real):self.writer.add_scalar('development/'+key,value,steps)
        for direction,metrics in report.get('directions',{}).items():
            for key,value in metrics.items():
                if isinstance(value,Real):self.writer.add_scalar(f'development/{direction}/{key}',value,steps)
        for case in report['cases']:
            for key in ('rmse_deg','peak_error_deg','endpoint_peak_error_deg','signed_progress_fraction','direction_correct','intervention_fraction'):
                value=case.get(key)
                if value is not None:self.writer.add_scalar(f'cases/{case["name"]}/{key}',value,steps)
        if 'directions' in report:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
            # Cover both directions and representative amplitudes at the lowest
            # evaluation speed, including curricula beyond the original 1–2deg.
            moving=[c for c in report['cases'] if c['sign']]
            amplitudes=sorted({c['amplitude_deg'] for c in moving})
            chosen=([amplitudes[0],amplitudes[len(amplitudes)//2],amplitudes[-1]]
                    if len(amplitudes)>3 else amplitudes)
            speed=min((c['speed_deg_s'] for c in moving),default=None)
            selected=[c for c in moving if c['amplitude_deg'] in chosen and c['speed_deg_s']==speed]
            for ax,case in zip(axes.ravel(),selected):
                with (directory/case['trace']).open() as stream:rows=list(csv.DictReader(stream))
                t=[float(r['time_s']) for r in rows]
                ax.plot(t,[float(r['reference_deg']) for r in rows],'--',label='Reference',color='black')
                ax.plot(t,[float(r['angle_deg']) for r in rows],label='Wheel',color='#277DA1')
                ax.set_title(case['name'],fontsize=9);ax.set_xlabel('Time (s)');ax.set_ylabel('Wheel angle (deg)');ax.grid(alpha=.2)
            for ax in axes.ravel()[len(selected):]:ax.set_visible(False)
            if selected:axes[0,0].legend(fontsize=8)
            self.writer.add_figure('development/angle_tracking',fig,global_step=steps,close=True)
        self.writer.flush()

    def close(self):
        if self.writer is not None:self.writer.close()
