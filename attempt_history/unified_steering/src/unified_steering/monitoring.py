"""Periodic development evaluation and retained checkpoints, isolated from training RNG."""
from contextlib import contextmanager
import json
import random

import numpy as np
import torch

from .audit import write_json
from .checkpoint import save_checkpoint
from .evaluate import evaluate
from .evaluate_commands import evaluate_commands,command_selection_score


@contextmanager
def isolated_evaluation(actor, critic):
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    numpy_state, python_state = np.random.get_state(), random.getstate()
    actor_mode,critic_mode = actor.training,critic.training
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        actor.train(actor_mode)
        critic.train(critic_mode)


def selection_score(report, anchor_hold_rmse):
    """Protect holding and completion before ranking worst/mean motion error."""
    hold = next(case for case in report['cases'] if case['name']=='neutral_hold')
    eligible = (report['all_cases_complete'] and hold['rmse_deg'] <= anchor_hold_rmse+.5
                and hold['hold_3s_drift_deg'] is not None and hold['hold_3s_drift_deg'] <= 2.)
    moving = [case for case in report['cases'] if case['name']!='neutral_hold']
    score = (report['tracking_success_rate'],-max(case['rmse_deg'] for case in moving),
             -float(np.mean([case['rmse_deg'] for case in moving])))
    return bool(eligible),score


class DevelopmentMonitor:
    def __init__(self, output, config, actor, critic, ppo, initial_action, evaluator=None):
        self.output,self.config = output,config
        self.actor,self.critic,self.ppo = actor,critic,ppo
        self.initial_action = initial_action
        self.command_suite=config.get('evaluation',{}).get('suite')=='single_direction'
        self.evaluator = evaluator or (evaluate_commands if self.command_suite else evaluate)
        self.fixed_report=None
        self.anchor_report=None
        self.last_report=None
        self.last_directory=None
        self.qualification_streak=0
        settings = config.get('monitoring',{})
        self.interval = int(settings.get('evaluation_every_steps',0))
        self.retain = bool(settings.get('retain_step_checkpoints',False))
        self.diagnostics = bool(settings.get('diagnostics',False))
        batch = config['ppo']['num_envs']*config['ppo']['rollout_steps']
        if self.interval < 0 or (self.interval and self.interval % batch):
            raise ValueError('Evaluation interval must be a positive multiple of a rollout batch, or zero')
        self.best_score = None
        self.best_step = None
        self.anchor_hold = None
        self.records = []

    def save_named(self, steps, updates):
        path = self.output/f'step_{steps:09d}.pt'
        if path.exists():
            return path
        save_checkpoint(path,self.actor,self.critic,self.ppo,self.config,self.initial_action,steps,updates)
        return path

    def check(self, steps, updates, initial=False):
        if not self.interval or (not initial and steps % self.interval):
            return None
        checkpoint = self.save_named(steps,updates)
        with isolated_evaluation(self.actor,self.critic):
            if self.command_suite and self.fixed_report is None:
                self.fixed_report=self.evaluator(self.config,None,self.output/'development'/'fixed_targets')
            self.last_directory=self.output/'development'/f'step_{steps:09d}'
            kwargs=dict(diagnostics=self.diagnostics)
            if self.command_suite:kwargs['fixed_report']=self.fixed_report
            report = self.evaluator(self.config,self.actor,self.last_directory,**kwargs)
        self.last_report=report
        if self.anchor_report is None:self.anchor_report=report
        hold = (dict(rmse_deg=report['hold_mean_rmse_deg']) if self.command_suite else
                next(case for case in report['cases'] if case['name']=='neutral_hold'))
        if self.anchor_hold is None:
            self.anchor_hold = hold['rmse_deg']
        eligible,score = (command_selection_score(report,self.anchor_report) if self.command_suite else
                          selection_score(report,self.anchor_hold))
        improved = eligible and (self.best_score is None or score > self.best_score)
        if improved:
            self.best_score,self.best_step = score,steps
            save_checkpoint(self.output/'best.pt',self.actor,self.critic,self.ppo,self.config,self.initial_action,steps,updates)
        record = dict(steps=steps,updates=updates,checkpoint=checkpoint.name,eligible=eligible,
                      selected_as_best=improved,score=list(score),hold_rmse_deg=hold['rmse_deg'],
                      mean_moving_case_rmse_deg=float(np.mean([c['rmse_deg'] for c in report['cases'] if (c['sign']!=0 if self.command_suite else c['name']!='neutral_hold')])),
                      safety_completion_rate=report['safety_completion_rate'],tracking_success_rate=report['tracking_success_rate'])
        if self.command_suite:
            self.qualification_streak=self.qualification_streak+1 if eligible and report['development_qualified'] else 0
            record.update(directions=report['directions'],hold_worst_peak_deg=report['hold_worst_peak_deg'],
                          development_qualified=report['development_qualified'],qualification_streak=self.qualification_streak)
            if self.qualification_streak>=2:
                save_checkpoint(self.output/'qualified_development.pt',self.actor,self.critic,self.ppo,self.config,self.initial_action,steps,updates)
        self.records.append(record)
        with (self.output/'development.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        write_json(self.output/'development_index.json',dict(schema='m30-development-monitor-v1',
                   anchor_hold_rmse_deg=self.anchor_hold,best_step=self.best_step,
                   selection=('Complete all cases, hold peak <=1deg, each hold RMSE <= anchor+0.5deg; rank worst direction success/improvement then errors'
                              if self.command_suite else 'Complete all cases, hold RMSE <= anchor+0.5deg, hold drift <=2deg; rank success, worst motion, mean motion'),
                   full_stage_qualification=False,records=self.records))
        return record
