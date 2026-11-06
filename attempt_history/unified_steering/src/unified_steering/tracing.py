"""Compressed per-action and physics traces for frozen-policy diagnostics."""
import json
from pathlib import Path

import numpy as np

from .audit import write_json
from .contract import MOTOR_NAMES
from .execution import Executor
from .telemetry import PHYSICS_FIELDS


def first_sustained(times, mask, duration, dt):
    run = 0
    needed = int(np.ceil(duration/dt-1e-9))
    for index, value in enumerate(mask):
        run = run+1 if value else 0
        if run >= needed:
            return float(times[index-needed+1])
    return None


class TraceRecorder:
    def __init__(self, env, metadata):
        self.metadata = {**metadata,'schema':'steering-trace-v2','time_origin_s':float(env.sim.data.time),
                         'initial_angle_rad':float(env.sim.data.qpos[env.sim.wheel_q]),
                         'physics_dt_s':float(env.dt),'motor_names':list(MOTOR_NAMES),
                         'joint_names':[env.sim.model.joint(i).name for i in range(env.sim.model.njnt)]}
        self.rows = []
        self.physics = []
        self.final_execution = None
        self.reason = None
        self.violations = []

    def record(self, env, info, observation, mean, std, latent, action):
        if not info['observation_valid'] or 'diagnostics' not in info:
            raise FloatingPointError('Diagnostic trace requires valid instrumented state')
        s,e = env.sim,env.executor
        detail = info['diagnostics']
        contacts = detail['contacts']
        self.rows.append(dict(time_s=np.array(s.data.time),actor_input=np.asarray(observation).copy(),
                              latent_mean=np.asarray(mean).copy(),latent_std=np.asarray(std).copy(),
                              sampled_latent=np.asarray(latent).copy(),sampled_action=np.asarray(action).copy(),
                              executed_action=e.previous.copy(),target=e.target.copy(),qpos=s.data.qpos.copy(),
                              qvel=s.data.qvel.copy(),actuator_force=s.data.actuator_force.copy(),
                              target_error=e.target-s.data.qpos[s.qidx],
                              reference_rad=np.array(info['reference_angle_rad']),
                              error_rad=np.array(info['error_rad']),wrench=contacts['wrench'].copy(),
                              force_norm_sum=contacts['force_norm_sum'].copy(),contact_count=contacts['contact_count'].copy(),
                              arm_margins=detail['joints']['arm_margins'].copy(),
                              native_margins=detail['joints']['native_margins'].copy(),
                              execution_counts=detail['execution']['step_counts'].copy()))
        self.physics.append(detail['physics'].copy())
        self.final_execution = detail['execution']
        self.reason = info['termination_reason']
        self.violations = detail['joints']['violations']

    def save(self, path):
        path = Path(path)
        arrays = {key:np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        physics = np.concatenate(self.physics,axis=0)
        arrays['physics'] = physics
        arrays['physics_fields'] = np.array(PHYSICS_FIELDS)
        arrays['execution_types'] = np.array(Executor.DIAGNOSTIC_TYPES)
        arrays['metadata_json'] = np.array(json.dumps(self.metadata,ensure_ascii=False,allow_nan=False))
        np.savez_compressed(path,**arrays)
        dt = self.metadata['physics_dt_s']
        columns = {name:index for index,name in enumerate(PHYSICS_FIELDS)}
        times = physics[:,0] if len(physics) else np.array([])
        summary = dict(schema='m30-trace-summary-v1',metadata=self.metadata,control_calls=len(self.rows),
                       physics_steps=len(physics),termination_reason=self.reason,joint_limit_violations=self.violations,
                       first_both_contacts_low_100ms_s=None,first_arm_lead_projection_s=None,
                       first_hand_slew_s=None,contact_low_fraction_per_hand=[],
                       execution_rates_per_joint={},longest_execution_runs_s={})
        if len(physics):
            forces = physics[:,[columns['left_contact_force_n'],columns['right_contact_force_n']]]
            low = forces <= .2
            summary['contact_low_fraction_per_hand'] = low.mean(axis=0).tolist()
            summary['first_both_contacts_low_100ms_s'] = first_sustained(times,np.all(low,axis=1),.1,dt)
            summary['first_either_contact_low_100ms_s'] = first_sustained(times,np.any(low,axis=1),.1,dt)
            for key,column in (('first_arm_lead_projection_s','arm_lead_axes'),('first_hand_slew_s','hand_slew_axes')):
                indices = np.flatnonzero(physics[:,columns[column]]>0)
                summary[key] = float(times[indices[0]]) if len(indices) else None
            summary['min_arm_margin_deg'] = float(np.rad2deg(np.min(physics[:,columns['min_arm_margin_rad']])))
            summary['min_native_joint_margin_rad'] = float(np.min(physics[:,columns['min_native_joint_margin_rad']]))
            summary['max_actuator_force_cap_ratio'] = float(np.max(physics[:,columns['max_force_cap_ratio']]))
            # Use only this recorder's frames, not warmup counters in a branched probe.
            counts = arrays['execution_counts'].sum(axis=0)
            for index,name in enumerate(Executor.DIAGNOSTIC_TYPES):
                summary['execution_rates_per_joint'][name] = (counts[index]/len(physics)).tolist()
                # Per-joint maxima recorded by executor are episode-wide. Label this
                # explicitly rather than claim these maxima are local to a branch.
                summary['longest_execution_runs_s'][name] = (self.final_execution['longest_runs'][index]*dt).tolist()
            summary['longest_run_scope'] = 'whole episode including any prefix before this trace'
        write_json(path.with_suffix('.summary.json'),summary)
        return summary
