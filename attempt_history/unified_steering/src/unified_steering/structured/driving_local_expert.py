"""Memoryless local-pose training labels, without a hidden phase integral."""
import numpy as np
from scipy.spatial.transform import Rotation
from .demonstrator import Demonstrator
from .driving_roll_integral_audit import TangentRollDemonstrator
from .driving_halfturn_expert import HalfturnDemonstrator

class LocalRollDemonstrator(TangentRollDemonstrator):
    tangent_position_gain=0.
    def predict(self,observation,deterministic=True):
        action,_=Demonstrator.predict(self,observation,deterministic)
        e,c,cfg=self.env,self.env.controller,self.config
        future=min(e.time+cfg.preview_seconds,(len(e.command_angles)-1)*e.dt)
        index=future/e.dt;i=int(index);j=min(i+1,len(e.command_angles)-1)
        reference=e.command_angles[i]+(index-i)*(e.command_angles[j]-e.command_angles[i])
        speed=e.command_velocities[i]+(index-i)*(e.command_velocities[j]-e.command_velocities[i])
        omega=speed+cfg.angle_gain*(reference-e.wheel_angle)
        phase=e.wheel_angle-self.initial_angle;self.controlled_phase=phase
        for side in range(2):
            if not self.support[side]:continue
            desired,rotation=self.pose(side,phase);p1,r1=self.pose(side,phase+1e-4)
            local=c.axis_frame.T@(c.wrist_positions[side]-c.center);phi=np.arctan2(local[2],local[1])
            tangent=c.axis_frame@np.array([0.,-np.sin(phi),np.cos(phi)])
            radial=c.axis_frame@np.array([0.,np.cos(phi),np.sin(phi)])
            error=desired-c.wrist_positions[side]
            correction=cfg.position_gain*error+(self.tangent_position_gain-cfg.position_gain)*(error@tangent)*tangent
            velocity=(p1-desired)/1e-4*omega+correction
            action[6*side:6*side+3]=np.array([velocity@tangent,velocity@radial,velocity@self.normal])/c.config.wrist_linear_speed
            ff=Rotation.from_matrix(r1@rotation.T).as_rotvec()/1e-4*omega
            error=Rotation.from_matrix(rotation@c.wrist_rotations[side].T).as_rotvec()
            action[6*side+3:6*side+6]=c.axis_frame.T@(ff+cfg.orientation_gain*error)/np.deg2rad(c.config.wrist_angular_speed_deg)
        return np.clip(action,-1.,1.),None

class LocalHalfturnDemonstrator(HalfturnDemonstrator,LocalRollDemonstrator):
    pass
