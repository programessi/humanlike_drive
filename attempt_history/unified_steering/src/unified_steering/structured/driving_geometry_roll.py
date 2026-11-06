"""Multi-start static IK diagnostics; no execution or learned-policy claims."""
import argparse,json,shutil,time
from pathlib import Path
from itertools import product
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .driving_audit import make_config
from .parking import SupportTeacherEnv


def disallowed_contacts(env,data):
    records=[];m=env.model
    for item in data.contact:
        a,b=m.geom_bodyid[[item.geom1,item.geom2]]
        allowed=any((a==env.wheel_body and b in g) or (b==env.wheel_body and a in g) or (a in g and b in g)
                    for g in env.hand_body_groups)
        if not allowed:
            records.append(dict(geom1=m.geom(item.geom1).name,geom2=m.geom(item.geom2).name,
                                distance_m=float(item.dist)))
    return records


def run(output,seed=40001):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,output/'source_snapshot.py')
    env=SupportTeacherEnv(make_config(dict(speed_deg_s=5,acceleration_deg_s2=40)))
    rng=np.random.default_rng(41);rows=[]
    try:
        env.reset(seed=seed,options={'free_side':-1,'mode':'hold','exact_curriculum':True})
        c=env.controller;m=env.model
        data=mujoco.MjData(m)
        origins=env.data.site_xpos[c.grip_ids].copy();rotations=env.data.site_xmat[c.grip_ids].reshape(2,3,3).copy()
        base=env.data.qpos.copy();nominal=base[c.arm_q].copy();previous={}
        def solve(side,angle,mode):
            inds=slice(side*7,(side+1)*7);site=c.grip_ids[side]
            rot=c.axis_frame@Rotation.from_rotvec([np.deg2rad(angle),0,0]).as_matrix()@c.axis_frame.T
            pos=c.center+rot@(origins[side]-c.center);orientation=rot@rotations[side]
            q0=nominal[inds];lo=c.limits[inds,0]+np.deg2rad(3);hi=c.limits[inds,1]-np.deg2rad(3)
            tangent=rot@c.axis_frame@np.array([0.,0.,1.])
            q0=np.r_[q0,0.];lo=np.r_[lo,-np.pi/2];hi=np.r_[hi,np.pi/2]
            prev=previous.get((side,mode,int(np.sign(angle))),q0)
            seeds=[prev,q0]+[np.clip(q0+rng.normal(0,s,8),lo,hi) for s in (.35,.8,1.5)]
            solutions=[]
            def residual(q):
                data.qpos[:]=base;data.qpos[c.arm_q[inds]]=q[:7]
                mujoco.mj_kinematics(m,data)
                wanted=Rotation.from_rotvec(tangent*q[7]).as_matrix()@orientation
                error=Rotation.from_matrix(wanted@data.site_xmat[site].reshape(3,3).T).as_rotvec()
                return np.r_[data.site_xpos[site]-pos,.12*error if mode in ('rigid','tangent_roll') else np.zeros(3),.0002*(q-q0)]
            for guess in seeds:
                sol=least_squares(residual,np.clip(guess,lo,hi),bounds=(lo,hi),max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
                res=residual(sol.x);poserr=np.linalg.norm(res[:3])
                wanted=Rotation.from_rotvec(tangent*sol.x[7]).as_matrix()@orientation
                orierr=Rotation.from_matrix(wanted@data.site_xmat[site].reshape(3,3).T).magnitude()
                solutions.append(dict(q=sol.x[:7].tolist(),tangent_roll_deg=float(np.rad2deg(sol.x[7])),position_error_mm=float(poserr*1000),
                    orientation_error_deg=float(np.rad2deg(orierr)),cost=float(np.linalg.norm(res)),
                    residual_passed=bool(poserr<=.003 and (mode not in ('rigid','tangent_roll') or orierr<=np.deg2rad(5)))))
            best=min(solutions,key=lambda x:x['cost']);previous[(side,mode,int(np.sign(angle)))]=np.r_[best['q'],np.deg2rad(best['tangent_roll_deg'])]
            return sorted(solutions,key=lambda x:x['cost'])
        for mode in ['tangent_roll']:
            for angle in ([0,-10,-20,-25,-30,-45,-60,-90,-120,-150,-180,10,20,25,30,45,60,90,120,150,180] if mode in ('rigid','tangent_roll') else [-30,-60,-90,-120,-180,30,60,90,120,180]):
                t=time.monotonic();sides=[solve(s,angle,mode) for s in range(2)]
                combinations=[]
                for left,right in product(sides[0],sides[1]):
                    data.qpos[:]=base;data.qpos[c.arm_q]=left['q']+right['q'];data.qpos[env.wheel_q]=env.wheel_angle+np.deg2rad(angle)
                    mujoco.mj_forward(m,data)
                    contacts=disallowed_contacts(env,data)
                    penetration=min([v['distance_m'] for v in contacts]+[0.])
                    residual=left['residual_passed'] and right['residual_passed']
                    combinations.append(dict(left=left,right=right,min_disallowed_contact_distance_m=penetration,
                        disallowed_contacts=contacts,residual_passed=residual,
                        feasible_in_search=bool(residual and penetration>=-1e-5)))
                pair=min(combinations,key=lambda x:(not x['feasible_in_search'],not x['residual_passed'],
                          -x['min_disallowed_contact_distance_m'],x['left']['cost']+x['right']['cost']))
                row=dict(mode=mode,rotation_from_prepared_grip_deg=angle,starts_per_side=5,
                    independent_static_search=True,continuous_path_proven=False,physical_grasp_proven=False,
                    **pair,wall_seconds=time.monotonic()-t)
                rows.append(row)
                (output/'summary.json').write_text(json.dumps(dict(seed=seed,cases=rows,
                    scope='Tangent-axis palm-roll geometric search with same joint and collision constraints; no dynamic grasp proof.'),indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ['left','right','disallowed_contacts']})+
                    ' ERR '+json.dumps([[r['position_error_mm'],r['orientation_error_deg']] for r in [pair['left'],pair['right']]]),flush=True)
    finally:env.close()
    print('GEOMETRY_DONE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
