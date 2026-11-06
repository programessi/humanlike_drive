"""Training-only same-side shuffle route for physical feasibility exploration.

Uses ordinary24 wrist/hand actions. It neither changes joint states nor writes
executor joint/posture targets. This is an earlier-regrasp style alternative,
not evidence for fixed-grip upper-hand motion all the way to180 degrees.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .driving_handoff_expert import HandoffDemonstrator
from .driving_halfturn_expert import smoothstep
from .driving_roll_integral_audit import TangentRollDemonstrator

class ShuffleDemonstrator(HandoffDemonstrator):
    release_start_deg=40.
    release_end_deg=60.
    clearance_m=.08
    cycle_start_deg=35.
    approach_tolerance_m=.007
    contact_seconds=.6
    min_contact_force_n=8.
    min_contact_torque_nm=.08
    clear_rotate_to_initial=False
    pregrasp_thumb_yaw=None
    pregrasp_thumb_pitch=None
    regrasp_axial_offset_m=0.
    regrasp_radial_offset_m=0.
    approach_radial_out_m=0.
    verify_thumb_before_transfer=False
    opposite_regrasp=False
    ipsilateral_regrasp=False
    ipsilateral_span_deg=50.
    ipsilateral_signed_cycle=True
    opposite_phase_deg=180.
    enforce_free_phase_unwind=False
    free_orientation_speed_deg_s=35.
    gradual_load_transfer=False
    transfer_ramp_seconds=2.
    follow_reference_direction=False
    free_axial_plane_m=None
    arc_relocate=False
    rebase_on_contact=False
    free_arc_radius_m=.11
    adaptive_grasp_roll_gain=0.
    load_share_phase_gain=0.
    drive_from_wrist_phase=False
    load_servo_gain=0.
    load_share_rate_deg_s=5.
    load_share_max_deg=25.
    adaptive_pitch_soft_rad=.32
    restore_on_return=False
    restore_start_deg=80.
    restore_separation_m=.16
    low_load_restore_transfer=False
    thumb_first_seconds=0.
    finger_close_seconds=.5
    park_after_two_transfers=False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.grasp_offsets=np.zeros((2,2));self.anchor_position=np.zeros((2,3));self.anchor_rotation=np.tile(np.eye(3),(2,1,1));self.anchor_ready=[False,False];self.adaptive_roll=np.zeros(2);self.motion_direction=self.direction;self.leg_start_time=0.

    def begin_free(self,side,mode,goal_phase):
        self.free_rotation_phase=self.wrist_phase(side) if self.drive_from_wrist_phase else self.controlled_phase+self.offset[side]
        super().begin_free(side,mode,goal_phase)
        self.grasp_offsets[side]=[self.regrasp_axial_offset_m,self.regrasp_radial_offset_m]
        self.anchor_position[side]=0.;self.anchor_rotation[side]=np.eye(3);self.anchor_ready[side]=False;self.adaptive_roll[side]=0.
        transfers=sum(ev['event']=='shuffle_transfer' and ev['time']>=self.leg_start_time for ev in self.events)
        if self.park_after_two_transfers and transfers>=2:self.mode='park'

    def pose(self,side,angle):
        p,r=super().pose(side,angle)
        if hasattr(self,'grasp_offsets'):
            vector=p-self.env.controller.center;radial=vector-self.normal*(vector@self.normal);radial/=np.linalg.norm(radial)
            p+=self.normal*self.grasp_offsets[side,0]+radial*self.grasp_offsets[side,1]
        if hasattr(self,'anchor_position'):
            p=p+r@self.anchor_position[side];r=r@self.anchor_rotation[side]
        if hasattr(self,'adaptive_roll') and self.adaptive_roll[side]!=0.:
            radial=p-self.env.controller.center;radial-=self.normal*(radial@self.normal);radial/=np.linalg.norm(radial)
            pivot=self.env.controller.center+self.env.rim_radius*radial;tangent=np.cross(self.normal,radial)
            correction=Rotation.from_rotvec(tangent*self.adaptive_roll[side]).as_matrix();p=pivot+correction@(p-pivot);r=correction@r
        return p,r

    def update_adaptive_roll(self):
        if not self.adaptive_grasp_roll_gain:return
        e=self.env;c=e.controller;m=e.model;d=c.ik_data;d.qpos[:]=e.data.qpos;mujoco.mj_forward(m,d)
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        for side in self.active:
            ids=slice(7*side,7*(side+1));q=e.data.qpos[c.arm_q[ids]];pitch=q[5];error=pitch-np.clip(pitch,-self.adaptive_pitch_soft_rad,self.adaptive_pitch_soft_rad)
            if not error:continue
            site=c.grip_ids[side];point=d.site_xpos[site];radial=point-c.center;radial-=self.normal*(radial@self.normal);radial/=np.linalg.norm(radial)
            pivot=c.center+e.rim_radius*radial;tangent=np.cross(self.normal,radial);linear=np.cross(tangent,point-pivot)
            mujoco.mj_jacSite(m,d,jp,jr,site);jac=np.vstack([jp[:,c.arm_v[ids]],.12*jr[:,c.arm_v[ids]]]);sensitivity=(np.linalg.pinv(jac,rcond=1e-5)@np.r_[linear,.12*tangent])[5]
            rate=-self.adaptive_grasp_roll_gain*error*sensitivity/(sensitivity*sensitivity+.05)
            self.adaptive_roll[side]=np.clip(self.adaptive_roll[side]+e.dt*np.clip(rate,-.6,.6),-np.pi/4,np.pi/4)

    def wrist_phase(self,side):
        c=self.env.controller
        here=c.axis_frame.T@(c.wrist_positions[side]-c.center)
        initial=c.axis_frame.T@(self.p0[side]-c.center)
        phi=np.arctan2(here[2],here[1])-np.arctan2(initial[2],initial[1])
        return np.arctan2(np.sin(phi),np.cos(phi))

    def drive(self,action,side,omega):
        if not self.drive_from_wrist_phase:return super().drive(action,side,omega)
        angle=self.wrist_phase(side)
        if self.stage=='unload' and self.load_servo_gain:
            demand=-self.env.load.centering_torque(self.env.wheel_angle)
            fraction=smoothstep(self.unload_progress)
            wanted=demand*(fraction if side==self.free else 1-fraction)
            torque=self.env._wheel_contact_torques()[side]
            angle+=np.clip(self.load_servo_gain*(wanted-torque)/self.config.position_gain,-np.deg2rad(8.),np.deg2rad(8.))
        p,r=self.pose(side,angle);p1,r1=self.pose(side,angle+1e-4)
        self.wrist(action,side,p,r,(p1-p)/1e-4*omega,Rotation.from_matrix(r1@r.T).as_rotvec()/1e-4*omega)

    def capture_anchor(self,side):
        p,r=self.pose(side,self.controlled_phase+self.offset[side]);c=self.env.controller
        actual_p=self.env.data.site_xpos[c.grip_ids[side]];actual_r=self.env.data.site_xmat[c.grip_ids[side]].reshape(3,3)
        self.anchor_position[side]=r.T@(actual_p-p);self.anchor_rotation[side]=r.T@actual_r;self.anchor_ready[side]=True
        self.events.append(dict(time=self.env.time,event='capture_actual_contact_anchor',side=side,position_offset_mm=float(np.linalg.norm(actual_p-p)*1000),orientation_offset_deg=float(np.rad2deg(Rotation.from_matrix(self.anchor_rotation[side]).magnitude()))))

    def predict(self,observation,deterministic=True):
        self.intent={}
        if self.restore_on_return:
            returning=self.env.target_velocity*self.env.wheel_angle<0 and abs(np.rad2deg(self.env.wheel_angle))<self.restore_start_deg
            if returning and not getattr(self,'return_restoring',False) and self.stage!='initial':
                self.return_restoring=True;self.restored_sides=[]
                previous=getattr(self,'transfer_old',self.active[0]) if self.stage=='unload' else self.active[0]
                self.active=[previous];self.begin_free(1-previous,'restore',self.controlled_phase)
            if self.stage=='dual' and self.env.target_velocity*self.env.wheel_angle>0 and abs(np.rad2deg(self.env.wheel_angle))>3:
                self.return_restoring=False;self.stage='initial';self.lower=0 if self.env.target_velocity<0 else 1;self.free=self.lower
        self.update_adaptive_roll()
        action,_=TangentRollDemonstrator.predict(self,observation,deterministic)
        e,c,cfg=self.env,self.env.controller,self.config
        future=min(round((e.time+cfg.preview_seconds)/e.dt),len(e.command_angles)-1)
        omega=e.command_velocities[future]+cfg.angle_gain*(e.command_angles[future]-e.wheel_angle)
        if self.omega_limit_deg_s is not None:omega=self.last_control_omega
        if abs(e.target_velocity)>np.deg2rad(.1) and np.sign(e.target_velocity)!=self.motion_direction:
            self.motion_direction=int(np.sign(e.target_velocity));self.leg_start_time=e.time
            if self.mode=='park':self.mode='shuffle';self.stage='relocate'
        for side in range(2):
            angle=self.controlled_phase+self.offset[side];p,r=self.pose(side,angle);p1,r1=self.pose(side,angle+1e-4)
            self.intent[side]=(p.copy(),r.copy(),(p1-p)/1e-4,Rotation.from_matrix(r1@r.T).as_rotvec()/1e-4)
        self.intent_omega=omega
        target=self.hand_target.copy()
        for side in range(2):target[side*6+2:side*6+6]+=self.grip_preload_rad+self.support_extra_rad
        for side in self.active:self.drive(action,side,omega)
        if self.stage=='initial':
            magnitude=abs(np.rad2deg(e.wheel_angle-self.initial_angle))
            opening=smoothstep((magnitude-self.release_start_deg)/(self.release_end_deg-self.release_start_deg))
            ids=slice(self.lower*6,(self.lower+1)*6)
            target[ids]=(1-opening)*target[ids]+opening*c.open_hand[ids]
            if opening>.999:
                self.active=[1-self.lower];self.begin_free(self.lower,'shuffle',0.)
        self.action_stage=self.stage;self.action_mode=self.mode;self.action_side=self.free
        if self.stage!='initial':
            side=self.free;ids=slice(side*6,(side+1)*6);target[ids]=c.open_hand[ids]
            if self.pregrasp_thumb_yaw is not None:target[6*side]=self.pregrasp_thumb_yaw
            if self.pregrasp_thumb_pitch is not None:target[6*side+1]=self.pregrasp_thumb_pitch
            motion_direction=self.direction
            if self.follow_reference_direction:
                if abs(e.target_velocity)>np.deg2rad(.1):self.regrasp_direction=int(np.sign(e.target_velocity))
                motion_direction=getattr(self,'regrasp_direction',self.direction)
            goal_phase=(-motion_direction*np.deg2rad(self.opposite_phase_deg) if side==(1 if motion_direction>0 else 0) else 0.) if self.opposite_regrasp else 0.
            if self.ipsilateral_regrasp:goal_phase=-motion_direction*np.deg2rad(self.ipsilateral_span_deg)
            if self.mode=='restore':goal_phase=self.controlled_phase
            elif self.stage in ('approach','insert','close','unload') and hasattr(self,'approach_goal_phase'):goal_phase=self.approach_goal_phase
            p,r=self.pose(side,goal_phase)
            actual=e.data.site_xpos[c.grip_ids[side]]
            vector=p-c.center;radial=vector-self.normal*(vector@self.normal);radial/=np.linalg.norm(radial)
            pregrasp=p+radial*self.approach_radial_out_m
            if self.mode=='park':
                vector=self.clear_origin-c.center;radial=vector-self.normal*(vector@self.normal);radial/=np.linalg.norm(radial)
                desired=self.clear_origin+.07*radial-self.normal*self.clearance_m
                self.wrist(action,side,desired,self.clear_rotation)
            elif self.stage=='clear':
                vector=self.clear_origin-c.center
                radial=vector-self.normal*(vector@self.normal);radial/=np.linalg.norm(radial)
                desired=self.clear_origin-self.normal*self.clearance_m+.03*radial
                if self.free_axial_plane_m is not None:
                    desired+=self.normal*(self.free_axial_plane_m-(desired-c.center)@self.normal)
                self.wrist(action,side,desired,self.clear_rotation)
                progress=(actual-self.clear_origin)@(-self.normal)
                cleared=progress>.8*self.clearance_m if self.free_axial_plane_m is None else (actual-c.center)@self.normal<self.free_axial_plane_m+.015
                if cleared:self.stage='relocate'
            elif self.stage in ('relocate','ready'):
                desired=pregrasp-self.normal*self.clearance_m
                if self.free_axial_plane_m is not None:
                    desired+=self.normal*(self.free_axial_plane_m-(desired-c.center)@self.normal)
                final_position=desired.copy()
                if self.arc_relocate:
                    here=c.axis_frame.T@(actual-c.center);there=c.axis_frame.T@(desired-c.center)
                    phi=np.arctan2(here[2],here[1]);goal_phi=np.arctan2(there[2],there[1]);error=np.arctan2(np.sin(goal_phi-phi),np.cos(goal_phi-phi))
                    if abs(error)>.12:
                        radius=self.free_arc_radius_m;phi+=np.clip(error,-np.pi/6,np.pi/6)
                        desired=c.center+c.axis_frame@np.array([there[0],radius*np.cos(phi),radius*np.sin(phi)])
                wanted_rotation=r
                if self.enforce_free_phase_unwind:
                    rate=np.deg2rad(self.free_orientation_speed_deg_s)*e.dt
                    self.free_rotation_phase+=np.clip(goal_phase-self.free_rotation_phase,-rate,rate)
                    wanted_rotation=self.pose(side,self.free_rotation_phase)[1]
                self.wrist(action,side,desired,wanted_rotation)
                roterr=Rotation.from_matrix(r@e.data.site_xmat[c.grip_ids[side]].reshape(3,3).T).magnitude()
                if np.linalg.norm(actual-final_position)<.01 and roterr<.15:
                    self.stage='ready'
                    active=self.active[0];phase=abs(np.rad2deg(self.controlled_phase+self.offset[active]))
                    if self.ipsilateral_regrasp and self.ipsilateral_signed_cycle:phase=motion_direction*np.rad2deg(self.controlled_phase+self.offset[active])
                    if self.drive_from_wrist_phase:phase=abs(np.rad2deg(self.wrist_phase(active)))
                    separated=np.linalg.norm(p-e.data.site_xpos[c.grip_ids[active]])>(self.restore_separation_m if self.mode=='restore' else .16)
                    if (phase>self.cycle_start_deg or self.mode=='restore') and separated:
                        self.approach_goal_phase=goal_phase;self.stage='approach'
            elif self.stage=='approach':
                if self.mode=='restore':
                    p1,r1=self.pose(side,goal_phase+1e-4);self.wrist(action,side,pregrasp,r,(p1-p)/1e-4*omega,Rotation.from_matrix(r1@r.T).as_rotvec()/1e-4*omega)
                else:self.wrist(action,side,pregrasp,r)
                if np.linalg.norm(actual-pregrasp)<self.approach_tolerance_m:
                    self.stage='insert'
            elif self.stage=='insert':
                if self.mode=='restore':
                    p1,r1=self.pose(side,goal_phase+1e-4);self.wrist(action,side,p,r,(p1-p)/1e-4*omega,Rotation.from_matrix(r1@r.T).as_rotvec()/1e-4*omega)
                else:self.wrist(action,side,p,r)
                if np.linalg.norm(actual-p)<self.approach_tolerance_m:
                    self.offset[side]=goal_phase-self.controlled_phase;self.stage='close';self.contact_age=0.;self.close_elapsed=0.
            elif self.stage=='close':
                self.close_elapsed=getattr(self,'close_elapsed',0.)+e.dt
                target[ids]=self.hand_target[ids]
                target[side*6+2:side*6+6]+=self.grip_preload_rad+self.support_extra_rad
                if self.thumb_first_seconds:
                    blend=smoothstep((self.close_elapsed-self.thumb_first_seconds)/self.finger_close_seconds)
                    fingers=slice(side*6+2,side*6+6);target[fingers]=(1-blend)*c.open_hand[fingers]+blend*target[fingers]
                self.drive(action,side,omega)
                torque=abs(e._wheel_contact_torques()[side]);load=abs(e.load.centering_torque(e.wheel_angle))
                useful=e.contact_forces[side]>self.min_contact_force_n and (torque>self.min_contact_torque_nm or load<.1)
                if self.thumb_first_seconds:useful=useful and self.close_elapsed>=self.thumb_first_seconds+self.finger_close_seconds
                if self.verify_thumb_before_transfer:
                    thumb=e.data.qpos[c.hand_q[6*side:6*side+2]]
                    useful=useful and thumb[0]<1.05 and thumb[1]>.15
                self.contact_age=self.contact_age+e.dt if useful else 0.
                if self.contact_age>self.contact_seconds:
                    if self.rebase_on_contact and not self.anchor_ready[side]:self.capture_anchor(side)
                    if self.mode=='restore' and self.restored_sides:
                        self.active=[0,1];self.stage='dual';self.offset[:]=0.;self.events.append(dict(time=e.time,event='restore_dual',wheel_deg=float(np.rad2deg(e.wheel_angle))))
                    elif self.gradual_load_transfer:
                        self.stage='unload';self.unload_progress=0.;self.transfer_old=self.active[0];self.active=[self.transfer_old,side];self.contact_age=0.;self.load_share_origin=self.offset[side]
                    else:
                        self.events.append(dict(time=e.time,event='shuffle_transfer',side=side,wheel_deg=float(np.rad2deg(e.wheel_angle)),torque_nm=torque))
                        previous=self.active[0];self.active=[side];self.begin_free(previous,'shuffle',0.)
            elif self.stage=='unload':
                target[ids]=self.hand_target[ids];target[side*6+2:side*6+6]+=self.grip_preload_rad+self.support_extra_rad
                old=self.transfer_old;oldids=slice(old*6,(old+1)*6)
                demand=-e.load.centering_torque(e.wheel_angle);torque=float(e._wheel_contact_torques()[side]);load=abs(demand)
                capacity=torque*np.sign(demand)/max(load,.05)
                if self.load_share_phase_gain:
                    wanted=demand*min(1.1,.4+.9*self.unload_progress)
                    rate=np.clip(self.load_share_phase_gain*(wanted-torque),-np.deg2rad(self.load_share_rate_deg_s),np.deg2rad(self.load_share_rate_deg_s))
                    reserve=np.deg2rad(self.load_share_max_deg)
                    self.offset[side]=np.clip(self.offset[side]+e.dt*rate,self.load_share_origin-reserve,self.load_share_origin+reserve)
                if self.unload_progress<.25 or capacity>=min(.2+.7*self.unload_progress,.8) or (self.low_load_restore_transfer and self.mode=='restore' and load<.1):
                    self.unload_progress=min(1.,self.unload_progress+e.dt/self.transfer_ramp_seconds)
                fraction=smoothstep(self.unload_progress)
                opened=c.open_hand[oldids].copy()
                if self.pregrasp_thumb_yaw is not None:opened[0]=self.pregrasp_thumb_yaw
                if self.pregrasp_thumb_pitch is not None:opened[1]=self.pregrasp_thumb_pitch
                target[oldids]=(1-fraction)*target[oldids]+fraction*opened
                useful=(capacity>.8 or load<.1) and e.contact_forces[side]>self.min_contact_force_n and abs(np.rad2deg(e.target_angle-e.wheel_angle))<2.
                self.contact_age=self.contact_age+e.dt if useful and self.unload_progress>.999 else 0.
                if self.contact_age>self.contact_seconds:
                    self.events.append(dict(time=e.time,event='shuffle_transfer',side=side,wheel_deg=float(np.rad2deg(e.wheel_angle)),torque_nm=torque,capacity_ratio=float(capacity)))
                    self.active=[side]
                    if self.mode=='restore':self.restored_sides.append(side)
                    self.begin_free(old,'restore' if getattr(self,'return_restoring',False) else 'shuffle',0.)
        if self.stage=='dual':
            target=self.hand_target.copy()
            for j in range(2):target[j*6+2:j*6+6]+=self.grip_preload_rad+self.support_extra_rad
        current,measured=c.target[14:],e.data.qpos[c.hand_q]
        target=np.clip(target,c.hand_limits[:,0]+.03,c.hand_limits[:,1]-.03)
        self.last_hand_target=target.copy()
        action[12:]=(cfg.hand_gain*(target-current)+(1-e.config.hand_leak_alpha)*(current-measured))/e.config.hand_delta_rad
        return np.clip(action,-1.,1.),None
