"""Frozen-policy driving-envelope diagnostics; never updates weights or scene.

Absolute endpoints and a timed return replace only the reference generator.
The prepared physical state, observation fields and finite executor are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch

from .demonstrator import Demonstrator, DemonstratorConfig
from .env import TaskConfig
from .evaluate import load_teacher
from .parking import SupportTeacherEnv


def duration(delta_rad, speed_deg_s, acceleration_deg_s2):
    return max(1., 1.875*abs(delta_rad)/np.deg2rad(speed_deg_s),
               math.sqrt(5.8*abs(delta_rad)/np.deg2rad(acceleration_deg_s2)))


def reference(start_rad, endpoint_rad, dt=.05, speed_deg_s=5., acceleration_deg_s2=40., hold_seconds=3.):
    """Minimum-jerk outbound, endpoint hold, return to absolute zero, zero hold."""
    if min(dt,speed_deg_s,acceleration_deg_s2,hold_seconds)<=0 or not np.isfinite([start_rad,endpoint_rad]).all():
        raise ValueError('Finite angles and positive reference parameters required')
    out=duration(endpoint_rad-start_rad,speed_deg_s,acceleration_deg_s2)
    back=duration(endpoint_rad,speed_deg_s,acceleration_deg_s2)
    timeline=dict(departure=.5, outbound_end=.5+out, return_start=.5+out+hold_seconds,
                  return_end=.5+out+hold_seconds+back)
    seconds=math.ceil((timeline['return_end']+hold_seconds)/dt)*dt
    count=math.ceil((seconds+.6)/dt)+2
    times=np.arange(count)*dt
    angles=np.full(count,start_rad); velocities=np.zeros(count)
    for left,right,initial,final in [(.5,.5+out,start_rad,endpoint_rad),
                                    (timeline['return_start'],timeline['return_end'],endpoint_rad,0.)]:
        u=np.clip((times-left)/(right-left),0,1); active=times>=left
        angles[active]=initial+(final-initial)*(10*u[active]**3-15*u[active]**4+6*u[active]**5)
        velocities[active]=(final-initial)/(right-left)*(30*u[active]**2-60*u[active]**3+30*u[active]**4)
    timeline.update(seconds=seconds,hold_seconds=hold_seconds)
    return angles,velocities,timeline


class DrivingAuditEnv(SupportTeacherEnv):
    """Diagnostic reference only; this is not the new R1/R2 training environment."""
    def _make_commands(self):
        sign=int(self.reset_options['direction'])
        angles,speeds,self.timeline=reference(self.wheel_angle,np.deg2rad(sign*self.config.amplitude_deg),
            self.dt,self.config.target_speed_deg,self.config.target_acceleration_deg_s2,
            self.reset_options.get('audit_hold_seconds',3.))
        self.config.episode_seconds=self.timeline['seconds']
        self.episode_direction=sign;self.episode_mode='driving_audit_roundtrip'
        return angles,speeds


def make_config(protocol, amplitude=10.):
    return TaskConfig(amplitude_deg=amplitude,episode_seconds=100.,hand_action_mode='leaky',
        hand_leak_alpha=.995,hand_joint_lead_rad=.15,angle_reward_sigma_deg=2.,
        target_speed_deg=protocol['speed_deg_s'],target_acceleration_deg_s2=protocol['acceleration_deg_s2'])


def clock_hours(env):
    c=env.controller
    local=(env.data.site_xpos[c.grip_ids]-c.center)@c.axis_frame
    hours=(9+np.rad2deg(np.arctan2(local[:,2],local[:,1]))/30)%12
    return np.where(hours<1e-8,12,hours)


def metadata(protocol,model_path):
    env=SupportTeacherEnv(make_config(protocol)); records=[]
    try:
        for support,free in [('both',-1),('left',1),('right',0)]:
            obs,info=env.reset(seed=protocol['seeds'][0],options={'free_side':free,'exact_curriculum':True,'direction':1,'mode':'hold'})
            c=env.controller
            local=(env.data.site_xpos[c.grip_ids]-c.center)@c.axis_frame
            records.append(dict(support=support,clock_hours=clock_hours(env).tolist(),
                grip_local_m=local.tolist(),initial_wheel_deg=float(np.rad2deg(env.wheel_angle)),
                force_n=[info['left_force'],info['right_force']],
                arm_margin_deg=float(np.rad2deg(env._arm_margin())),actor_dim=len(obs['actor']),privileged_dim=len(obs['privileged'])))
        return dict(model=str(model_path),model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            scene=str(env.scene_path),scene_sha256=hashlib.sha256(env.scene_path.read_bytes()).hexdigest(),
            wheel_range_deg=np.rad2deg(env.model.joint('wheel_joint').range).tolist(),
            model_limit_540_compatible=False,task_angle_limit_deg=env.config.max_wheel_deg,
            config_amplitude_limit_deg=100,strict_substep_wheel_limit_deg=440,
            observations=dict(actor_dim=152,privileged_dim=54,latent_dim=16,action_dim=24,
                wheel_angle_index=120,wheel_speed_index=121,preview_indices=[122,128],
                reference_speed_index=128,wheel_angle_encoding='accumulated radians; not wrapped or clipped',
                preview_encoding='(future command - measured wheel)/10deg; not clipped',
                support_role_label=False),
            positive_wheel_direction='clockwise as seen by driver; left hand rises, right hand falls',
            left_turn_direction=-1,right_turn_direction=1,
            reset_records=records,task=asdict(env.config),
            limitations=['Current task limit110 and amplitude validation100 do not support180/540.',
                'Existing prepared wheel angle may drift from calibrated zero; no physical reset correction applied.',
                'Static IK and explicit controller are diagnostic evidence, not learned-policy acceptance.'])
    finally:env.close()


def _initialize(model_path,protocol):
    global _model,_protocol
    torch.set_num_threads(1);_protocol=protocol;_model=load_teacher(model_path)


def _case(actor,support,sign,amplitude,seed,output):
    protocol=_protocol
    env=DrivingAuditEnv(make_config(protocol,amplitude));rows=[]
    free={'both':-1,'left':1,'right':0}[support]
    wall=time.monotonic()
    try:
        obs,prep=env.reset(seed=seed,options={'free_side':free,'exact_curriculum':True,'direction':sign,
            'audit_hold_seconds':protocol['hold_seconds'],'mode':'step'})
        initial=float(np.rad2deg(env.wheel_angle));initial_clock=clock_hours(env).tolist()
        expert=Demonstrator(env,DemonstratorConfig(**protocol['demonstrator'])) if actor=='expert' else None
        done=False; diagnostic_stop=''
        for _ in range(round(env.config.episode_seconds*env.config.policy_hz)):
            action=(expert if expert is not None else _model).predict(obs,deterministic=True)[0]
            obs,_,done,truncated,info=env.step(action)
            c=env.controller
            actual=env.data.qpos[c.arm_q]
            margins=np.minimum(actual-c.limits[:,0],c.limits[:,1]-actual)
            group=np.rad2deg([min(margins[:7]),min(margins[7:])])
            rows.append(dict(time=env.time,angle_deg=float(np.rad2deg(env.wheel_angle)),
                target_deg=float(np.rad2deg(env.target_angle)),error_deg=info['error_deg'],
                physics_peak_error_deg=info['physics_peak_error_deg'],
                force_left=info['left_force'],force_right=info['right_force'],
                torque_left=info['left_wheel_contact_torque_nm'],torque_right=info['right_wheel_contact_torque_nm'],
                spring_torque=info['centering_torque_nm'],
                margin_left_deg=group[0],margin_right_deg=group[1],
                min_margin_deg=info['physics_min_arm_margin_deg'],collision_n=info['physics_collision_peak_n'],
                qp_failures=info['qp_failures'],collision_qp_failures=info['collision_qp_failures'],
                qp_error_mm=info['qp_position_error_mm'],workspace_clips=info['workspace_clips'],
                wrist_command_phase_deg=(np.rad2deg(c.phase)).tolist(),slip_deg=np.rad2deg(env.slip).tolist(),
                clock_hours=clock_hours(env).tolist(),
                max_actor_abs=float(np.max(np.abs(obs['actor']))),max_preview_abs=float(np.max(np.abs(obs['actor'][122:128]))),
                action_saturated_channels=int(np.sum(np.abs(action)>=.999))))
            if info['qp_failures']>0:
                diagnostic_stop='first_qp_failure'
            if diagnostic_stop or done or truncated:break
        times=np.array([r['time'] for r in rows]);errors=np.array([r['error_deg'] for r in rows])
        complete=bool(not done and not diagnostic_stop and times[-1]+1e-8>=env.config.episode_seconds)
        def metric(mask,key='error_deg',rmse=False):
            data=np.array([r[key] for r in rows])[mask]
            return (float(np.sqrt(np.mean(data**2))) if rmse else float(np.max(np.abs(data)))) if len(data) else None
        t=env.timeline
        outbound=times<=t['outbound_end']+1e-8
        returning=(times>=t['return_start']-1e-8)&(times<=t['return_end']+1e-8)
        hold=(times>t['outbound_end']+1e-8)&(times<t['return_start']-1e-8)
        tail=times>=t['seconds']-3-1e-8
        rmse=float(np.sqrt(np.mean(errors**2)))
        tail_peak=metric(tail,'physics_peak_error_deg') if complete else None
        end_hold_peak=metric(hold,'physics_peak_error_deg') if complete else None
        reason=diagnostic_stop or info['termination_reason']
        collision=max(r['collision_n'] for r in rows)
        passed=complete and rmse<=1 and tail_peak<=1 and end_hold_peak<=1 and collision<=1
        record=dict(actor=actor,support=support,direction=sign,turn='left' if sign<0 else 'right',amplitude_deg=amplitude,
            seed=seed,complete=complete,passed=bool(passed),termination_reason=reason,
            planned_seconds=t['seconds'],actual_seconds=env.time,timeline=t,initial_wheel_deg=initial,
            initial_clock_hours=initial_clock,initial_forces_n=[prep['left_force'],prep['right_force']],
            prefix_rmse_deg=rmse,full_rmse_deg=rmse if complete else None,
            outbound_rmse_deg=metric(outbound,rmse=True),return_rmse_deg=metric(returning,rmse=True) if complete else None,
            peak_error_deg=max(r['physics_peak_error_deg'] for r in rows),tail_peak_deg=tail_peak,
            endpoint_hold_peak_deg=end_hold_peak,
            max_abs_wheel_deg=max(abs(r['angle_deg']) for r in rows),last_wheel_deg=rows[-1]['angle_deg'],
            last_target_deg=rows[-1]['target_deg'],last_forces_n=[rows[-1]['force_left'],rows[-1]['force_right']],
            min_arm_margin_deg=min(r['min_margin_deg'] for r in rows),collision_peak_n=collision,
            max_qp_error_mm=max(r['qp_error_mm'] for r in rows),qp_failures=info['qp_failures'],
            workspace_clips=info['workspace_clips'],last_wrist_clock=rows[-1]['clock_hours'],
            limiting_joint_names=[info['left_limiting_joint'],info['right_limiting_joint']],
            wall_seconds=time.monotonic()-wall,learned_policy=actor=='policy',independent_acceptance=False)
        stem=f'{actor}_{support}_{sign:+d}_{amplitude:g}_s{seed}'
        path=Path(output)/'traces'/f'{stem}.npz'
        np.savez_compressed(path,**{key:np.array([r[key] for r in rows]) for key in rows[0]})
        record['trace']=str(path.relative_to(Path(output)))
        (Path(output)/'cases'/f'{stem}.json').write_text(json.dumps(record,indent=2))
        return record
    finally:env.close()


def _ladder(job):
    actor,support,sign,seed,output=job
    records=[]
    for amplitude in _protocol['amplitudes_deg']:
        try:r=_case(actor,support,sign,amplitude,seed,output)
        except Exception as exc:
            r=dict(actor=actor,support=support,direction=sign,amplitude_deg=amplitude,seed=seed,
                   complete=False,passed=False,termination_reason='exception',exception=repr(exc))
        records.append(r)
        print('CASE '+json.dumps(r),flush=True)
        if r['termination_reason']:
            break
    return records


def report(output,protocol,records):
    lines=['# 新版R0：冻结网络扩角与显式物理基线','',
        '仅为开发诊断，不更新权重，不计作新版R1/R2独立验收。参考为绝对目标角→保持→绝对零位，按时间推进；初始短保持参考使用真实settling轮角。',
        '遇到硬终止或首次QP失败后停止该梯级，不把未完成前缀RMSE算完整成绩。未运行的更大角度属于未知。','',
        '| 执行者 | 支撑 | 转向 | 目标 | 完整/通过 | 全程RMSE | 终止原因 | 最小臂余量 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(records,key=lambda r:(r['actor'],r['support'],r['direction'],r['amplitude_deg'],r['seed'])):
        rmse=r.get('full_rmse_deg');margin=r.get('min_arm_margin_deg')
        lines.append(f"| {r['actor']} | {r['support']} | {'左' if r['direction']<0 else '右'} | {r['amplitude_deg']}° | {r['complete']}/{r['passed']} | {rmse if rmse is not None else '未完成'} | {r['termination_reason'] or '无'} | {round(margin,3) if margin is not None else '—'} |")
    (Path(output)/'README.md').write_text('\n'.join(lines)+'\n')
    (Path(output)/'summary.json').write_text(json.dumps(dict(stage='driving-R0',training=False,
        independent_acceptance=False,protocol=protocol,cases=records),indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('runs/r2_frozen_v4/teacher.zip'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(1)
    protocol=json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['traces','cases','source_snapshot']:(args.output/name).mkdir()
    shutil.copy2(args.protocol,args.output/'protocol.json')
    for source in [Path(__file__),Path(__file__).with_name('demonstrator.py'),Path(__file__).with_name('env.py')]:
        shutil.copy2(source,args.output/'source_snapshot'/source.name)
    meta=metadata(protocol,args.model)
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('METADATA '+json.dumps(meta),flush=True)
    jobs=[(actor,support,sign,seed,str(args.output)) for actor in protocol['actors']
        for support in protocol['supports'] for sign in protocol['directions'] for seed in protocol['seeds']]
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork'),
            initializer=_initialize,initargs=(str(args.model),protocol)) as pool:
        for group in pool.map(_ladder,jobs):
            records.extend(group);report(args.output,protocol,records)
    assert hashlib.sha256(args.model.read_bytes()).hexdigest()==meta['model_sha256']
    print('DONE '+json.dumps(dict(cases=len(records),complete=sum(r['complete'] for r in records),passed=sum(r['passed'] for r in records))),flush=True)


if __name__=='__main__':main()
