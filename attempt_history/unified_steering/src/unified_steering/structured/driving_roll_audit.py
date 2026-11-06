"""Explicit tangent-roll grasp baseline for R0 physical diagnosis only.

Rotates both palm pose and its offset about the local wheel-rim tangent.
Outputs the same 24 finite actions; never writes simulated state or joint targets.
"""
import json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import driving_audit
from .demonstrator import Demonstrator

_TABLE=None

class TangentRollDemonstrator(Demonstrator):
    def __init__(self,env,config=None):
        super().__init__(env,config)
        rows=json.loads(Path(_TABLE).read_text())['cases']
        self.grid=np.array(sorted({r['rotation_from_prepared_grip_deg'] for r in rows}),float)
        self.roll=np.array([[next(r for r in rows if r['rotation_from_prepared_grip_deg']==deg)[side]['tangent_roll_deg']
                            for deg in self.grid] for side in ['left','right']])
        c=env.controller
        self.p0=env.data.site_xpos[c.grip_ids].copy()
        self.rim=[];self.tangent=[]
        for side in range(2):
            local=c.axis_frame.T@(self.p0[side]-c.center)
            radial=np.r_[0.,local[1:]/np.linalg.norm(local[1:])]
            self.rim.append(c.center+c.axis_frame@(env.rim_radius*radial))
            self.tangent.append(c.axis_frame@np.cross([1.,0.,0.],radial))
        self.normal=c.axis_frame[:,0]

    def pose(self,side,angle):
        c=self.env.controller
        psi=np.deg2rad(np.interp(np.rad2deg(angle),self.grid,self.roll[side]))
        # Static roll search uses the positive left-hand tangent for both sides.
        tangent=self.tangent[side]*(1 if side==0 else -1)
        turn=Rotation.from_rotvec(self.normal*angle).as_matrix()
        roll=Rotation.from_rotvec(tangent*psi).as_matrix()
        position=c.center+turn@(self.rim[side]-c.center+roll@(self.p0[side]-self.rim[side]))
        rotation=turn@roll@self.rotations[side]
        return position,rotation

    def predict(self,observation,deterministic=True):
        action,_=super().predict(observation,deterministic)
        e,c,cfg=self.env,self.env.controller,self.config
        future=min(e.time+cfg.preview_seconds,(len(e.command_angles)-1)*e.dt)
        ind=future/e.dt;i=int(ind);j=min(i+1,len(e.command_angles)-1)
        reference=e.command_angles[i]+(ind-i)*(e.command_angles[j]-e.command_angles[i])
        velocity=e.command_velocities[i]+(ind-i)*(e.command_velocities[j]-e.command_velocities[i])
        omega=velocity+cfg.angle_gain*(reference-e.wheel_angle)
        angle=e.wheel_angle-self.initial_angle
        eps=1e-4
        for side in range(2):
            if not self.support[side]:continue
            desired,rotation=self.pose(side,angle)
            next_pos,next_rotation=self.pose(side,angle+eps)
            linear=(next_pos-desired)/eps*omega+cfg.position_gain*(desired-c.wrist_positions[side])
            local=c.axis_frame.T@(c.wrist_positions[side]-c.center)
            phi=np.arctan2(local[2],local[1])
            tangent=c.axis_frame@np.array([0.,-np.sin(phi),np.cos(phi)])
            radial=c.axis_frame@np.array([0.,np.cos(phi),np.sin(phi)])
            action[side*6:side*6+3]=np.array([linear@tangent,linear@radial,linear@self.normal])/c.config.wrist_linear_speed
            ff=Rotation.from_matrix(next_rotation@rotation.T).as_rotvec()/eps*omega
            error=Rotation.from_matrix(rotation@c.wrist_rotations[side].T).as_rotvec()
            action[side*6+3:side*6+6]=(c.axis_frame.T@(ff+cfg.orientation_gain*error))/np.deg2rad(c.config.wrist_angular_speed_deg)
        return np.clip(action,-1.,1.),None


def main():
    global _TABLE
    protocol=Path(sys.argv[sys.argv.index('--protocol')+1])
    output=Path(sys.argv[sys.argv.index('--output')+1])
    p=json.loads(protocol.read_text());_TABLE=p['tangent_geometry']
    driving_audit.Demonstrator=TangentRollDemonstrator
    driving_audit.main()
    shutil.copy2(__file__,output/'source_snapshot'/'driving_roll_audit.py')
    shutil.copy2(_TABLE,output/'source_snapshot'/'geometry_roll_table.json')
    (output/'roll_baseline.json').write_text(json.dumps(dict(scripted_physical_baseline=True,learned_policy=False,
        tangent_geometry_source=_TABLE,table_sha256=hashlib.sha256(Path(_TABLE).read_bytes()).hexdigest(),
        description='Same24 actions, finite executor. Tangent-roll schedule from static IK; position and orientation rotated about rim tangent. No scene or state assignment.'),indent=2))

if __name__=='__main__':main()
