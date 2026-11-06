"""Driving curriculum v1: accumulated, timed references; unchanged v4 executor.

The 185 degree software envelope supports half-turn tests within the original
450 degree physical stops. It does not alter joint ranges or safety gates.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from .env import TaskConfig
from .parking import SupportTeacherEnv, ParkedController
from .driving_audit import duration


@dataclass
class DrivingConfig(TaskConfig):
    teacher_contact_observation: bool = False
    command_range_observation: bool = False
    arm_command_margin_deg: float = 3.
    guard_elbows_against_wheel: bool = False
    guard_opposite_hand_meshes: bool = False
    guard_full_physical_pairs: bool = False
    retain_preparation_hand_command: bool = False
    nullspace_joint_limit_gain: float = 0.
    qp_backend: str = 'slsqp'
    canonical_distance_guard_order: bool = False
    collision_guard_margin_m: float = .012
    collision_guard_activation_m: float = .035
    max_wheel_deg: float = 200.
    amplitude_deg: float = 90.
    reference_envelope_deg: float = 185.
    hand_action_mode: str = 'leaky'
    hand_leak_alpha: float = .995
    hand_joint_lead_rad: float = .15
    angle_reward_sigma_deg: float = 2.
    episode_seconds: float = 200.

    def validate(self):
        # Reuse every old validation except its small-angle amplitude ceiling.
        super(DrivingConfig, replace(self, amplitude_deg=min(self.amplitude_deg, 100.))).validate()
        if not (0 <= self.amplitude_deg <= self.reference_envelope_deg <= 185.):
            raise ValueError('Driving v1 reference outside its audited software envelope')
        if not self.reference_envelope_deg < self.max_wheel_deg < 440.:
            raise ValueError('Reference and abort envelopes require braking space')
        if not .012 <= self.collision_guard_margin_m < self.collision_guard_activation_m <= .10:
            raise ValueError('Collision guard margins may only be tightened')
        if self.qp_backend not in ('slsqp','active-set'):
            raise ValueError('Unknown versioned QP backend')
        if not np.isfinite(self.nullspace_joint_limit_gain) or not 0 <= self.nullspace_joint_limit_gain <= 1.:
            raise ValueError('Invalid nullspace joint-limit gain')
        if not 3. <= self.arm_command_margin_deg <= 8.:
            raise ValueError('Driving arm command reserve may only be tightened from3deg')


def timed_reference(start, amplitude, direction, template, dt=.05, speed=5., acceleration=40.):
    if template not in ('short', 'long', 'sequence') or direction not in (-1, 1):
        raise ValueError('Unknown driving template or direction')
    if not np.isfinite([start, amplitude, dt, speed, acceleration]).all() or min(dt, speed, acceleration) <= 0 or amplitude <= 0:
        raise ValueError('Invalid driving reference')
    hold = 10. if template == 'long' else 3.
    endpoints = [direction*amplitude, 0.]
    if template == 'sequence':
        endpoints += [-direction*amplitude, 0.]
    cursor, value, segments = .5, start, []
    for endpoint in endpoints:
        end = cursor+duration(endpoint-value, speed, acceleration)
        segments.append(dict(start=cursor, end=end, hold_end=end+hold,
                             initial=value, target=endpoint))
        cursor, value = end+hold, endpoint
    seconds = math.ceil(cursor/dt)*dt
    t = np.arange(math.ceil((seconds+.6)/dt)+2)*dt
    angles = np.full(len(t), start); velocities = np.zeros(len(t))
    for segment in segments:
        a, b = segment['start'], segment['end']
        initial, final = segment['initial'], segment['target']
        u = np.clip((t-a)/(b-a), 0., 1.); active = t >= a
        angles[active] = initial+(final-initial)*(10*u[active]**3-15*u[active]**4+6*u[active]**5)
        velocities[active] = (final-initial)/(b-a)*(30*u[active]**2-60*u[active]**3+30*u[active]**4)
    return angles, velocities, dict(seconds=seconds, hold_seconds=hold, segments=segments)


class DrivingController(ParkedController):
    arm_command_margin_deg = 3.

    def _bounds(self, previous, speed, acceleration, limits, measured, dt):
        if previous.start == 0 and getattr(self,'strict_safety',False):
            extra = np.deg2rad(self.arm_command_margin_deg-3.)
            limits = limits.copy();limits[:,0] += extra;limits[:,1] -= extra
        return super()._bounds(previous,speed,acceleration,limits,measured,dt)

    def _guard_constraints(self):
        canonical=getattr(self,'canonical_distance_guard_order',False)
        full_hands=getattr(self,'guard_opposite_hand_meshes',False) or getattr(self,'guard_full_physical_pairs',False)
        if not canonical and not full_hands:
            return super()._guard_constraints()
        # Installed MuJoCo's mixed primitive/mesh distance implementation orders
        # fromto by shape type even for a reversed caller pair. Query in its
        # canonical order so each endpoint belongs to the body used in mj_jac.
        # The distance and the physical collision model remain unchanged.
        original=self.guard_pairs
        selected=original
        if full_hands:
            pairs=np.asarray(original,int)
            delta=self.command_data.geom_xpos[pairs[:,0]]-self.command_data.geom_xpos[pairs[:,1]]
            lower=np.linalg.norm(delta,axis=1)-self.model.geom_rbound[pairs[:,0]]-self.model.geom_rbound[pairs[:,1]]
            selected=[original[i] for i in np.flatnonzero(lower<=self.config.collision_activation_m)]
        self.guard_pairs=[tuple(sorted(pair,key=lambda g:int(self.model.geom_type[g]))) for pair in selected] if canonical else selected
        try:return super()._guard_constraints()
        finally:self.guard_pairs=original

    def reset_reference(self, data):
        super().reset_reference(data)
        if getattr(self,'retain_preparation_hand_command',False):
            measured = data.qpos[self.hand_q]
            lead = self.hand_joint_lead_rad or self.config.joint_lead
            self.target[14:] = measured+np.clip(self.closed_hand-measured,-lead,lead)

    def _solve_backend(self,positions,rotations,measured,dt):
        if getattr(self,'qp_backend','slsqp')=='slsqp':return super().solve_velocity(positions,rotations,measured,dt)
        import mujoco
        from scipy.spatial.transform import Rotation
        from .box_qp import solve_qp
        c=self.config;self.command_data.qpos[:]=measured.qpos;self.command_data.qpos[self.arm_q]=self.target[:14]
        mujoco.mj_forward(self.model,self.command_data);matrices=[];targets=[]
        for side,site in enumerate(self.grip_ids):
            mujoco.mj_jacSite(self.model,self.command_data,self.jacp,self.jacr,site)
            error=positions[side]-self.command_data.site_xpos[site]
            current=self.command_data.site_xmat[site].reshape(3,3)
            rot_error=Rotation.from_matrix(rotations[side]@current.T).as_rotvec()
            matrices.extend([self.jacp[:,self.arm_v].copy(),c.orientation_weight*self.jacr[:,self.arm_v].copy()])
            targets.extend([c.position_gain*error,c.orientation_weight*c.position_gain*rot_error]);self.qp_error[side]=np.linalg.norm(error)
        a=np.vstack(matrices+[c.damping*np.eye(14),c.posture_weight*np.eye(14)])
        b=np.concatenate(targets+[np.zeros(14),c.posture_weight*(self.seed-self.target[:14])])
        lo,hi=self._bounds(slice(0,14),c.arm_speed,c.arm_acceleration,self.limits,measured.qpos[self.arm_q],dt)
        rows,lower=self._guard_constraints();self.collision_constraints+=len(rows)
        result=solve_qp(a.T@a,-a.T@b,lo,hi,rows,lower,start=self.velocity[:14])
        self.last_qp_result=result
        velocity=result.x
        if not result.success:
            self.qp_failures+=1;self.collision_qp_failures+=int(bool(rows));velocity=np.clip(np.zeros(14),lo,hi)
        self.velocity[:14]=velocity;self.target[:14]+=dt*velocity

    def solve_velocity(self,positions,rotations,measured,dt):
        gain=getattr(self,'nullspace_joint_limit_gain',0.)
        if not gain:return self._solve_backend(positions,rotations,measured,dt)
        import mujoco
        self.command_data.qpos[:]=measured.qpos
        self.command_data.qpos[self.arm_q]=self.target[:14]
        mujoco.mj_forward(self.model,self.command_data)
        rows=[]
        for site in self.grip_ids:
            mujoco.mj_jacSite(self.model,self.command_data,self.jacp,self.jacr,site)
            rows.extend([self.jacp[:,self.arm_v].copy(),self.config.orientation_weight*self.jacr[:,self.arm_v].copy()])
        jac=np.vstack(rows)
        projection=np.eye(14)-np.linalg.pinv(jac,rcond=1e-5)@jac
        q=self.target[:14];lo=q-self.limits[:,0];hi=self.limits[:,1]-q
        gradient=np.clip(gain*(1/(lo+.05)**2-1/(hi+.05)**2),-2.,2.)
        original=self.seed
        self.seed=q+projection@gradient
        try:return self._solve_backend(positions,rotations,measured,dt)
        finally:self.seed=original

    def enable_opposite_hand_guards(self):
        groups=[]
        for side in ('left','right'):
            wrist=self.model.body(f'{side}_wrist_yaw_link').id;bodies={wrist}
            for body in range(wrist+1,self.model.nbody):
                if self.model.body_parentid[body] in bodies:bodies.add(body)
            groups.append([int(g) for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies and self.model.geom_contype[g]])
        existing={frozenset(pair) for pair in self.guard_pairs}
        self.guard_pairs += [(a,b) for a in groups[0] for b in groups[1] if frozenset((a,b)) not in existing]

    def enable_full_physical_guards(self,hand_groups,wheel_body):
        m=self.model;mobile=set()
        for side in ('left','right'):
            root=m.body(f'{side}_shoulder_pitch_link').id;bodies={root}
            for body in range(root+1,m.nbody):
                if m.body_parentid[body] in bodies:bodies.add(body)
            mobile.update(bodies)
        excludes={int(value) for value in m.exclude_signature};existing={frozenset(pair) for pair in self.guard_pairs}
        physical=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        for i,a in enumerate(physical):
            ba=int(m.geom_bodyid[a])
            for b in physical[i+1:]:
                bb=int(m.geom_bodyid[b])
                if ba not in mobile and bb not in mobile:continue
                if not ((m.geom_contype[a]&m.geom_conaffinity[b]) or (m.geom_contype[b]&m.geom_conaffinity[a])):continue
                if m.body_weldid[ba]==m.body_weldid[bb]:continue
                if ((min(ba,bb)<<16)|max(ba,bb)) in excludes:continue
                if any((ba==wheel_body and bb in group) or (bb==wheel_body and ba in group) or (ba in group and bb in group) for group in hand_groups):continue
                key=frozenset((a,b))
                if key not in existing:self.guard_pairs.append((a,b));existing.add(key)

    def enable_wheel_elbow_guards(self):
        wheel_body = self.model.body('steering_wheel').id
        rims = np.flatnonzero((self.model.geom_bodyid == wheel_body)&(self.model.geom_contype != 0))
        elbows = [self.model.geom(f'{side}_elbow_link_guard').id for side in ['left','right']]
        self.guard_pairs += [(elbow,int(rim)) for elbow in elbows for rim in rims]


class DrivingEnv(SupportTeacherEnv):
    """One24-output policy; optional contact truth is training-teacher only."""
    def _observation(self):
        observation=super()._observation()
        if self.config.teacher_contact_observation:
            observation['teacher_contact']=np.r_[self.contact_forces/20.,self._wheel_contact_torques()[:2]/.5].astype(np.float32)
        if self.config.command_range_observation:
            observation['command_range']=np.array([np.min(self.command_angles),np.max(self.command_angles)],np.float32)
        return observation

    controller_class = DrivingController

    def __init__(self, config=None, **kwargs):
        config = config or DrivingConfig()
        super().__init__(config, **kwargs)
        if config.teacher_contact_observation:
            from gymnasium import spaces
            self.observation_space.spaces['teacher_contact']=spaces.Box(-np.inf,np.inf,(4,),np.float32)
        if config.command_range_observation:
            from gymnasium import spaces
            self.observation_space.spaces['command_range']=spaces.Box(-np.inf,np.inf,(2,),np.float32)
        self.controller.arm_command_margin_deg = config.arm_command_margin_deg
        self.controller.retain_preparation_hand_command = config.retain_preparation_hand_command
        self.controller.nullspace_joint_limit_gain = config.nullspace_joint_limit_gain
        self.controller.qp_backend = config.qp_backend
        self.controller.canonical_distance_guard_order = config.canonical_distance_guard_order
        self.controller.guard_opposite_hand_meshes = config.guard_opposite_hand_meshes
        self.controller.guard_full_physical_pairs = config.guard_full_physical_pairs
        self.controller.config.collision_margin_m = config.collision_guard_margin_m
        self.controller.config.collision_activation_m = config.collision_guard_activation_m
        if config.guard_elbows_against_wheel:
            self.controller.enable_wheel_elbow_guards()
        if config.guard_opposite_hand_meshes:
            self.controller.enable_opposite_hand_guards()
        if config.guard_full_physical_pairs:
            self.controller.enable_full_physical_guards(self.hand_body_groups,self.wheel_body)


    def _make_commands(self):
        self.episode_direction = int(self.reset_options['direction'])
        self.episode_mode = self.reset_options.get('template', 'short')
        angles, speeds, self.timeline = timed_reference(self.wheel_angle,
            np.deg2rad(self.config.amplitude_deg), self.episode_direction,
            self.episode_mode, self.dt, self.config.target_speed_deg,
            self.config.target_acceleration_deg_s2)
        self.config.episode_seconds = self.timeline['seconds']
        return angles, speeds

    def set_reference(self, timestamps, angles_rad):
        """Accept an executable accumulated-angle sequence without retiming it.

        Times are in the current episode's clock. Invalid angle/rate/acceleration
        commands are rejected, never clipped into a different evaluation task.
        """
        times, angles = np.asarray(timestamps, float), np.asarray(angles_rad, float)
        if times.ndim != 1 or times.shape != angles.shape or len(times) < 3:
            raise ValueError('Expected matching one-dimensional timestamp/angle arrays')
        if not np.isfinite([times, angles]).all() or np.any(np.diff(times) <= 0) or times[0] > self.time+1e-9 or times[-1] <= self.time:
            raise ValueError('Reference must span the present and have increasing finite timestamps')
        if max(abs(angles)) > np.deg2rad(self.config.reference_envelope_deg)+1e-10:
            raise ValueError('Reference exceeds the versioned driving envelope')
        velocity = np.diff(angles)/np.diff(times)
        acceleration = np.diff(velocity)/((np.diff(times)[:-1]+np.diff(times)[1:])/2)
        if max(abs(velocity)) > np.deg2rad(self.config.target_speed_deg)+1e-8 or max(abs(acceleration)) > np.deg2rad(self.config.target_acceleration_deg_s2)+1e-8:
            raise ValueError('Reference is not executable at the configured speed/acceleration')
        count = math.ceil((times[-1]+.6)/self.dt)+2
        grid = np.arange(count)*self.dt
        self.command_angles = np.interp(grid, times, angles)
        self.command_velocities = np.interp(grid, (times[:-1]+times[1:])/2, velocity, left=velocity[0], right=0.)
        self.config.episode_seconds = float(times[-1])
        return self._observation()
