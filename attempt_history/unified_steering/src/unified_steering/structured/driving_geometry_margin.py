"""Search rim-tangent palm orientations with explicit arm reserve.

This is offline static diagnosis, using the actual palm-offset geometry of the
dynamic expert. It neither assigns runtime states nor changes physical limits.
"""
import argparse,json,time
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .driving_task import DrivingConfig,DrivingEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--margin',type=float,default=6.);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=DrivingEnv(DrivingConfig(amplitude_deg=90));rng=np.random.default_rng(61);rows=[]
    try:
        env.reset(seed=44501,options=dict(free_side=-1,direction=1,template='short'))
        c=env.controller;m=env.model;d=mujoco.MjData(m);base=env.data.qpos.copy()
        p0=env.data.site_xpos[c.grip_ids].copy();r0=env.data.site_xmat[c.grip_ids].reshape(2,3,3).copy()
        previous={}
        for angle in [0,*range(-15,-181,-15),*range(15,181,15)]:
            row=dict(rotation_from_prepared_grip_deg=angle,physical_grasp_proven=False,
                     continuous_path_proven=False,static_arm_reserve_deg=args.margin)
            turn=Rotation.from_rotvec(c.axis_frame[:,0]*np.deg2rad(angle)).as_matrix()
            for side,name in enumerate(['left','right']):
                ids=slice(side*7,(side+1)*7);q0=base[c.arm_q[ids]]
                local=c.axis_frame.T@(p0[side]-c.center)
                radial=np.r_[0.,local[1:]/np.linalg.norm(local[1:])]
                rim=c.center+c.axis_frame@(env.rim_radius*radial)
                tangent=c.axis_frame@np.array([0.,0.,1.])
                lo=np.r_[c.limits[ids,0]+np.deg2rad(args.margin),-np.pi/2]
                hi=np.r_[c.limits[ids,1]-np.deg2rad(args.margin),np.pi/2]
                initial=np.r_[q0,0.];prev=previous.get((side,int(np.sign(angle))),initial)
                def pose(x):
                    roll=Rotation.from_rotvec(tangent*x[7]).as_matrix()
                    return c.center+turn@(rim-c.center+roll@(p0[side]-rim)),turn@roll@r0[side]
                def residual(x):
                    d.qpos[:]=base;d.qpos[c.arm_q[ids]]=x[:7];mujoco.mj_kinematics(m,d)
                    pos,rot=pose(x);actual=d.site_xmat[c.grip_ids[side]].reshape(3,3)
                    margin=np.minimum(x[:7]-c.limits[ids,0],c.limits[ids,1]-x[:7])
                    return np.r_[d.site_xpos[c.grip_ids[side]]-pos,
                        .12*Rotation.from_matrix(rot@actual.T).as_rotvec(),
                        .0002*(x[:7]-q0),.01*np.maximum(0.,np.deg2rad(12.)-margin)]
                solutions=[]
                for guess in [prev,initial,initial+rng.normal(0,.6,8)]:
                    sol=least_squares(residual,np.clip(guess,lo,hi),bounds=(lo,hi),max_nfev=200,
                        ftol=1e-8,xtol=1e-8,gtol=1e-8)
                    res=residual(sol.x);solutions.append((np.linalg.norm(res[:6]),sol.x,res))
                _,x,res=min(solutions,key=lambda item:item[0]);previous[(side,int(np.sign(angle)))]=x
                margin=np.minimum(x[:7]-c.limits[ids,0],c.limits[ids,1]-x[:7])
                row[name]=dict(q=x[:7].tolist(),tangent_roll_deg=float(np.rad2deg(x[7])),
                    position_error_mm=float(np.linalg.norm(res[:3])*1000),
                    orientation_error_deg=float(np.rad2deg(np.linalg.norm(res[3:6])/.12)),
                    min_arm_margin_deg=float(np.rad2deg(min(margin))))
            rows.append(row)
            (args.output/'summary.json').write_text(json.dumps(dict(cases=rows,scope='offline reserve-oriented geometry; not physical or learned success'),indent=2)+'\n')
            print(json.dumps(row),flush=True)
    finally:env.close()

if __name__=='__main__':main()
