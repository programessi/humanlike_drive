"""Replay selected frozen audit failures with passive substep joint diagnostics."""
import json,time
from pathlib import Path
import mujoco,numpy as np,torch
from unified_steering.structured.driving_audit import DrivingAuditEnv,make_config
from unified_steering.structured.evaluate import load_teacher
from unified_steering.structured.demonstrator import Demonstrator,DemonstratorConfig

class Probe(DrivingAuditEnv):
    def _physics_step(self):
        super()._physics_step()
        if self.strict_reason and not getattr(self,'failure',None):
            m,d,c=self.model,self.data,self.controller
            values=d.qpos[m.jnt_qposadr[self.limited_ids]];limits=m.jnt_range[self.limited_ids]
            outside=(values<limits[:,0]-.02)|(values>limits[:,1]+.02)
            near=[]
            for j,value,lim,violate in zip(self.limited_ids,values,limits,outside):
                margin=min(value-lim[0],lim[1]-value)
                if margin<np.deg2rad(3) or violate:
                    near.append(dict(joint=m.joint(j).name,value_rad=float(value),range_rad=lim.tolist(),
                        margin_deg=float(np.rad2deg(margin)),strict_violation=bool(violate)))
            self.failure=dict(reason=self.strict_reason,time=float(d.time),wheel_deg=float(np.rad2deg(self.wheel_angle)),
                target_deg=float(np.rad2deg(self.target_angle)),joints=near,
                wrist_position_lead_mm=(np.linalg.norm(c.wrist_positions-d.site_xpos[c.grip_ids],axis=1)*1000).tolist(),
                forces=self.contact_forces.tolist())

torch.set_num_threads(1)
root=Path('/home/nubot/humanlike_drive/unified_steering');protocol=json.loads((root/'configs/driving_r0_v1.json').read_text())
model=load_teacher(root/'runs/r2_frozen_v4/teacher.zip');rows=[]
for actor,support,sign,amplitude in [('policy','both',-1,60),('policy','both',1,60),('expert','left',-1,90),('expert','left',1,30)]:
    e=Probe(make_config(protocol,amplitude));e.failure=None
    try:
        obs,_=e.reset(seed=40001,options={'free_side':{'both':-1,'left':1,'right':0}[support],
            'exact_curriculum':True,'direction':sign,'audit_hold_seconds':3.,'mode':'step'})
        expert=Demonstrator(e,DemonstratorConfig(**protocol['demonstrator']))
        angles=[];times=[]
        for _ in range(round(e.config.episode_seconds*e.config.policy_hz)):
            a=(model if actor=='policy' else expert).predict(obs,deterministic=True)[0]
            obs,_,done,trunc,info=e.step(a);angles.append(np.rad2deg(e.wheel_angle));times.append(e.time)
            if done or trunc or info['qp_failures']:break
        original=root/'results/driving_r0_v1/traces'/f'{actor}_{support}_{sign:+d}_{amplitude}_s40001.npz'
        with np.load(original) as trace:
            parity=bool(np.array_equal(np.asarray(angles),trace['angle_deg']) and np.array_equal(np.asarray(times),trace['time']))
        row=dict(actor=actor,support=support,direction=sign,amplitude_deg=amplitude,
            failure=e.failure,termination_reason=info['termination_reason'],passive_trace_identical=parity)
        rows.append(row);print(json.dumps(row),flush=True)
    finally:e.close()
(root/'results/driving_r0_v1/failure_details.json').write_text(json.dumps(rows,indent=2))
