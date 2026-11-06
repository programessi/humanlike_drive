"""Contact-confirmed early regrasp candidate used for training labels only.

A physical route exploration for this robot's limited fixed-grip half-turn. No
policy evaluation imports or invokes this state machine.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .driving_roll_integral_audit import TangentRollDemonstrator
from .driving_halfturn_expert import smoothstep

class HandoffDemonstrator(TangentRollDemonstrator):
    release_start_deg=60.
    release_end_deg=80.
    clearance_m=.12
    regrasp_phase_deg=120.
    grip_preload_rad=.15
    support_extra_rad=.15
    clear_radial_m=.03
    clear_rotate_to_initial=True
    def __init__(self,env,config=None):
        super().__init__(env,config)
        self.direction=int(np.sign(env.timeline['segments'][0]['target']))
        self.offset=np.zeros(2);self.active=[0,1];self.stage='initial';self.mode='out'
        self.lower=0 if self.direction<0 else 1
        self.free=self.lower;self.contact_age=0.;self.goal_phase=-self.direction*np.deg2rad(self.regrasp_phase_deg)
        self.clear_origin=None;self.events=[]

    def wrist(self,action,side,position,rotation,linear=None,angular=None):
        c,cfg=self.env.controller,self.config
        raw_linear=linear;raw_angular=angular
        linear=np.zeros(3) if linear is None else linear
        angular=np.zeros(3) if angular is None else angular
        if hasattr(self,'intent'):
            omega=getattr(self,'intent_omega',0.)
            self.intent[side]=(position.copy(),rotation.copy(),linear/omega if raw_linear is not None and abs(omega)>1e-8 else np.zeros(3),angular/omega if raw_angular is not None and abs(omega)>1e-8 else np.zeros(3))
        pg=getattr(self,'free_position_gain',cfg.position_gain) if raw_linear is None else cfg.position_gain
        og=getattr(self,'free_orientation_gain',cfg.orientation_gain) if raw_angular is None else cfg.orientation_gain
        if hasattr(self,'intent') and (pg!=cfg.position_gain or og!=cfg.orientation_gain):
            pos=c.wrist_positions[side]+pg/cfg.position_gain*(position-c.wrist_positions[side])
            err=Rotation.from_matrix(rotation@c.wrist_rotations[side].T).as_rotvec()
            rot=Rotation.from_rotvec(og/cfg.orientation_gain*err).as_matrix()@c.wrist_rotations[side]
            old=self.intent[side];self.intent[side]=(pos,rot,old[2],old[3])
        velocity=linear+pg*(position-c.wrist_positions[side])
        local=c.axis_frame.T@(c.wrist_positions[side]-c.center);phi=np.arctan2(local[2],local[1])
        tangent=c.axis_frame@np.array([0.,-np.sin(phi),np.cos(phi)])
        radial=c.axis_frame@np.array([0.,np.cos(phi),np.sin(phi)])
        action[6*side:6*side+3]=np.array([velocity@tangent,velocity@radial,velocity@self.normal])/c.config.wrist_linear_speed
        error=Rotation.from_matrix(rotation@c.wrist_rotations[side].T).as_rotvec()
        action[6*side+3:6*side+6]=c.axis_frame.T@(angular+og*error)/np.deg2rad(c.config.wrist_angular_speed_deg)

    def drive(self,action,side,omega):
        angle=self.controlled_phase+self.offset[side];p,r=self.pose(side,angle);p1,r1=self.pose(side,angle+1e-4)
        self.wrist(action,side,p,r,(p1-p)/1e-4*omega,Rotation.from_matrix(r1@r.T).as_rotvec()/1e-4*omega)

    def begin_free(self,side,mode,goal_phase):
        self.free=side;self.mode=mode;self.goal_phase=goal_phase;self.stage='clear'
        self.clear_origin=self.env.controller.wrist_positions[side].copy()
        self.clear_rotation=(self.rotations[side] if self.clear_rotate_to_initial else self.env.controller.wrist_rotations[side]).copy();self.contact_age=0.
        self.events.append(dict(time=self.env.time,event='free_'+mode,side=side,wheel_deg=float(np.rad2deg(self.env.wheel_angle))))

    def predict(self,observation,deterministic=True):
        action,_=super().predict(observation,deterministic)
        e,c,cfg=self.env,self.env.controller,self.config
        index=min(round(e.time/e.dt),len(e.command_velocities)-1)
        omega=e.command_velocities[index]+cfg.angle_gain*(e.target_angle-e.wheel_angle)
        target=self.hand_target.copy()
        for ids in [slice(2,6),slice(8,12)]:target[ids]+=self.grip_preload_rad
        for side in self.active:
            self.drive(action,side,omega)
            target[side*6+2:side*6+6]+=self.support_extra_rad*smoothstep((abs(np.rad2deg(e.wheel_angle-self.initial_angle))-20.)/40.)
        if self.stage=='initial':
            magnitude=abs(np.rad2deg(e.wheel_angle-self.initial_angle))
            opening=smoothstep((magnitude-self.release_start_deg)/(self.release_end_deg-self.release_start_deg))
            ids=slice(self.lower*6,(self.lower+1)*6)
            target[ids]=(1-opening)*target[ids]+opening*c.open_hand[ids]
            # An opened palm can still touch the rim. Withdrawal, rather than
            # waiting for zero force before withdrawal, establishes clearance.
            if opening>.999:
                self.active=[1-self.lower];self.begin_free(self.lower,'out',-self.direction*np.deg2rad(self.regrasp_phase_deg))
        if self.stage not in ('initial','done'):
            side=self.free;ids=slice(side*6,(side+1)*6)
            target[ids]=c.open_hand[ids]
            p,r=self.pose(side,self.goal_phase)
            local=c.axis_frame.T@(e.data.site_xpos[c.grip_ids[side]]-c.center)
            if self.stage=='clear':
                vector=self.clear_origin-c.center;radial=vector-self.normal*(vector@self.normal);radial/=np.linalg.norm(radial)
                desired=self.clear_origin-self.normal*(self.clearance_m+(c.axis_frame.T@(self.clear_origin-c.center))[0])+self.clear_radial_m*radial
                self.wrist(action,side,desired,self.clear_rotation)
                if local[0]<-.8*self.clearance_m:self.stage='relocate'
            elif self.stage=='relocate':
                desired=p-self.normal*self.clearance_m
                self.wrist(action,side,desired,r)
                poserr=np.linalg.norm(e.data.site_xpos[c.grip_ids[side]]-desired)
                roterr=Rotation.from_matrix(r@e.data.site_xmat[c.grip_ids[side]].reshape(3,3).T).magnitude()
                returning=self.direction*e.target_velocity<-np.deg2rad(.1)
                separated=np.linalg.norm(p-e.data.site_xpos[c.grip_ids[1-side]])>.16
                allowed=(self.mode=='out' and separated) or (self.mode=='back' and returning and separated) or (self.mode=='restore' and abs(np.rad2deg(e.wheel_angle))<20.)
                if poserr<.012 and roterr<.18 and allowed:self.stage='approach'
            elif self.stage=='approach':
                if self.mode=='restore':
                    self.goal_phase=e.wheel_angle-self.initial_angle;p,r=self.pose(side,self.goal_phase)
                self.wrist(action,side,p,r)
                if np.linalg.norm(e.data.site_xpos[c.grip_ids[side]]-p)<.012:
                    self.stage='close';self.offset[side]=self.goal_phase-self.controlled_phase
            elif self.stage=='close':
                target[ids]=self.hand_target[ids]
                target[side*6+2:side*6+6]+=self.grip_preload_rad
                self.drive(action,side,omega)
                self.contact_age=self.contact_age+e.dt if e.contact_forces[side]>1. else 0.
                if self.contact_age>.4:
                    self.events.append(dict(time=e.time,event='regrasp_'+self.mode,side=side,wheel_deg=float(np.rad2deg(e.wheel_angle))))
                    if self.mode=='restore':
                        self.active=[0,1];self.stage='done'
                    else:
                        previous=self.active[0];self.active=[side]
                        self.begin_free(previous,'back' if self.mode=='out' else 'restore',self.direction*np.deg2rad(self.regrasp_phase_deg) if self.mode=='out' else 0.)
        current,measured=c.target[14:],e.data.qpos[c.hand_q]
        target=np.clip(target,c.hand_limits[:,0]+.03,c.hand_limits[:,1]-.03)
        action[12:]=(cfg.hand_gain*(target-current)+(1-e.config.hand_leak_alpha)*(current-measured))/e.config.hand_delta_rad
        return np.clip(action,-1.,1.),None
