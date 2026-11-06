"""Isolate hand contact from arm kinematics using bounded virtual wrist wrenches.

Free-hand poses and wheel state are written at reset only. During execution,
fingers use finite position actuators and palms receive bounded physical
wrenches. Wheel feedback never enters the Cartesian executor.
"""
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..common.cockpit import HAND_NAMES
from . import ACTION_DIM, OBSERVATION_DIM
from .scene import SCENE_PATH, build_scene
from .reachability import ArmReachabilityGuard, ReachabilityConfig


@dataclass
class HandOnlyConfig:
    amplitude_deg: float = 5.
    target_mode: str = "sequence"
    episode_seconds: float = 20.
    settle_seconds: float = 1.2
    target_speed_deg: float = 5.
    policy_hz: int = 20
    linear_speed: float = .05
    angular_speed_deg: float = 40.
    force_limit_n: float = 30.
    torque_limit_nm: float = 2.
    position_kp: float = 600.
    position_kd: float = 30.
    orientation_kp: float = 8.
    orientation_kd: float = .35
    no_contact: bool = False
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)

    def validate(self):
        self.reachability.validate()
        if self.target_mode not in ("hold", "positive", "negative", "sequence"):
            raise ValueError("Unknown hand-only reference")
        if not 0 <= self.amplitude_deg <= 90:
            raise ValueError("Invalid wheel amplitude")
        values = [self.episode_seconds, self.target_speed_deg, self.policy_hz,
                  self.linear_speed, self.angular_speed_deg, self.force_limit_n,
                  self.torque_limit_nm, self.position_kp, self.position_kd,
                  self.orientation_kp, self.orientation_kd]
        if not np.isfinite(values).all() or min(values) <= 0 or self.settle_seconds < 0:
            raise ValueError("Invalid hand-only execution parameters")


def limited(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-12))


class HandOnlyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config=None, render_mode=None, scene_path=None):
        self.config = config or HandOnlyConfig()
        self.config.validate()
        path = Path(scene_path or SCENE_PATH)
        if not path.exists():
            build_scene(path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.render_mode, self.renderer = render_mode, None
        self.hand_jids = np.array([self.model.joint(n).id for n in HAND_NAMES])
        self.hand_q = self.model.jnt_qposadr[self.hand_jids]
        self.hand_v = self.model.jnt_dofadr[self.hand_jids]
        self.hand_limits = self.model.jnt_range[self.hand_jids].copy()
        self.hand_limits[[1, 7]] = [0., .49]
        self.closed = np.tile([.6, .42, 1., 1., 1., 1.], 2)
        self.open = np.tile([.3, .1, .15, .15, .15, .15], 2)
        self.hand_body = [self.model.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]
        self.grip = [self.model.site(f"{s}_grip").id for s in ("left", "right")]
        self.grip_offset = self.model.site_pos[self.grip].copy()
        self.body_groups = []
        for wrist in self.hand_body:
            group = {wrist}
            for body in range(wrist + 1, self.model.nbody):
                if self.model.body_parentid[body] in group:
                    group.add(body)
            self.body_groups.append(group)
        self.wheel_body = self.model.body("steering_wheel").id
        self.wheel_q = int(self.model.joint("wheel_joint").qposadr[0])
        self.wheel_v = int(self.model.joint("wheel_joint").dofadr[0])
        self.wheel_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.wheel_body))
        if self.config.no_contact:
            for geom in self.wheel_geoms:
                self.model.geom_contype[geom] = self.model.geom_conaffinity[geom] = 0
        if self.model.nu != 12 or any(self.model.actuator_trnid[:, 0] == self.model.joint("wheel_joint").id):
            raise ValueError("Only the 12 finger drives may have actuators")
        mujoco.mj_forward(self.model, self.data)
        wheel_site = self.model.site("wheel_center").id
        self.center = self.data.site_xpos[wheel_site].copy()
        self.frame = self.data.site_xmat[wheel_site].reshape(3, 3).copy()
        self.arm_guard = ArmReachabilityGuard(self.config.reachability) if self.config.reachability.enabled else None
        if self.arm_guard is not None and (not np.allclose(self.center, self.arm_guard.center, atol=1e-8)
                                          or not np.allclose(self.frame, self.arm_guard.frame, atol=1e-8)):
            raise ValueError('Hand scene and H1 reachability fixture transforms differ')
        self.dt = 1 / self.config.policy_hz
        self.substeps = round(self.dt / self.model.opt.timestep)
        if abs(self.dt - self.substeps * self.model.opt.timestep) > 1e-9:
            raise ValueError("Policy clock must divide physics clock")
        self.action_space = spaces.Box(-1, 1, (ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), dtype=np.float32)
        self.jacp, self.jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.contact_forces = np.zeros(2)
        self.applied_wrenches = np.zeros((2, 6))
        self.previous_action = self.filtered_action = np.zeros(ACTION_DIM)
        self.time = 0.

    def _reference(self, time):
        t = np.asarray(time, dtype=float)
        amplitude = np.deg2rad(self.config.amplitude_deg)
        angle, velocity = np.zeros_like(t), np.zeros_like(t)
        if self.config.target_mode == "hold":
            return angle, velocity
        goals = {"positive": [amplitude], "negative": [-amplitude],
                 "sequence": [amplitude, 0., -amplitude, 0.]}[self.config.target_mode]
        cursor, value = .5, 0.
        for goal in goals:
            delta = goal - value
            duration = max(1., 1.875 * abs(delta) / np.deg2rad(self.config.target_speed_deg),
                           np.sqrt(5.8 * abs(delta) / np.deg2rad(40.)))
            u = np.clip((t - cursor) / duration, 0., 1.)
            active = t >= cursor
            angle = np.where(active, value + delta * (10*u**3 - 15*u**4 + 6*u**5), angle)
            velocity = np.where(active, delta / duration * (30*u**2 - 60*u**3 + 30*u**4), velocity)
            cursor, value = cursor + duration + 1., goal
        return angle, velocity

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.hand_q] = self.open
        for equality in range(self.model.neq):
            first, second = self.model.eq_obj1id[equality], self.model.eq_obj2id[equality]
            self.data.qpos[self.model.jnt_qposadr[first]] = np.polynomial.polynomial.polyval(
                self.data.qpos[self.model.jnt_qposadr[second]], self.model.eq_data[equality, :5])
        mujoco.mj_forward(self.model, self.data)
        self.positions = self.data.site_xpos[self.grip].copy()
        self.rotations = self.data.site_xmat[self.grip].reshape(2, 3, 3).copy()
        if self.arm_guard is not None:
            self.positions, self.rotations = self.arm_guard.reset(self.positions, self.rotations)
        self.hand_targets = self.open.copy()
        self.previous_action = np.zeros(ACTION_DIM)
        self.filtered_action = np.zeros(ACTION_DIM)
        for step in range(round(self.config.settle_seconds / self.model.opt.timestep)):
            fraction = min(1., (step + 1) * self.model.opt.timestep / .8)
            self.hand_targets = self.open + fraction * (self.closed - self.open)
            self._physics_step()
        mujoco.mj_forward(self.model, self.data)
        self.time = 0.
        self._contacts()
        if self.arm_guard is not None and not self.arm_guard.assess_actual(
                self.data.site_xpos[self.grip], self.data.site_xmat[self.grip].reshape(2, 3, 3)):
            raise ValueError(f'Settled hand grasp leaves the H1 envelope: {self.arm_guard.actual_reason}')
        self.initial_material = self._material_phase()
        self.last_hand_q = self.data.qpos[self.hand_q].copy()
        return self._observation(), self._info()

    def _integrate_action(self, action):
        self.filtered_action += (1 - np.exp(-self.dt / .1)) * (action - self.filtered_action)
        c = self.config
        for side, site in enumerate(self.grip):
            a = self.filtered_action[6*side:6*(side+1)]
            local = self.frame.T @ (self.positions[side] - self.center)
            radius = max(.085, np.linalg.norm(local[1:]))
            phase = np.arctan2(local[2], local[1]) + self.dt * c.linear_speed * a[0] / radius
            radius = np.clip(radius + self.dt * c.linear_speed * a[1], .085, .28)
            axis = np.clip(local[0] + self.dt * c.linear_speed * a[2], -.14, .10)
            desired = self.center + self.frame @ [axis, radius*np.cos(phase), radius*np.sin(phase)]
            self.positions[side] = self.data.site_xpos[site] + limited(desired - self.data.site_xpos[site], .012)
            increment = self.frame @ (np.deg2rad(c.angular_speed_deg) * self.dt * a[3:])
            desired_rotation = Rotation.from_rotvec(increment).as_matrix() @ self.rotations[side]
            actual = self.data.site_xmat[site].reshape(3, 3)
            error = Rotation.from_matrix(desired_rotation @ actual.T).as_rotvec()
            self.rotations[side] = Rotation.from_rotvec(limited(error, np.deg2rad(10))).as_matrix() @ actual
        if self.arm_guard is not None:
            # Feasibility solve runs on separate kinematic data only. The
            # physical executor still receives Cartesian targets and wrenches.
            self.positions, self.rotations = self.arm_guard.filter(self.positions, self.rotations, self.dt)
        lo, hi = self.hand_limits[:, 0] + .005, self.hand_limits[:, 1] - .005
        a = self.filtered_action[12:]
        desired = np.where(a < 0, self.closed + a*(self.closed-lo), self.closed + a*(hi-self.closed))
        self.hand_targets += np.clip(desired-self.hand_targets, -1.5*self.dt, 1.5*self.dt)

    def _physics_step(self):
        c, m, d = self.config, self.model, self.data
        d.qfrc_applied[:] = 0.
        for side, body in enumerate(self.hand_body):
            rotation = self.rotations[side]
            desired = self.positions[side] - rotation @ self.grip_offset[side]
            actual_rotation = d.xmat[body].reshape(3, 3)
            rotation_error = Rotation.from_matrix(rotation @ actual_rotation.T).as_rotvec()
            # Forward velocity measurement only; no inverse/Jacobian solve.
            mujoco.mj_jacBody(m, d, self.jacp, self.jacr, body)
            force = c.position_kp * (desired-d.xpos[body]) - c.position_kd * (self.jacp @ d.qvel)
            torque = c.orientation_kp * rotation_error - c.orientation_kd * (self.jacr @ d.qvel)
            for child in self.body_groups[side]:
                compensation = -m.body_mass[child] * m.opt.gravity
                force += compensation
                torque += np.cross(d.xipos[child]-d.xpos[body], compensation)
            force, torque = limited(force, c.force_limit_n), limited(torque, c.torque_limit_nm)
            self.applied_wrenches[side] = np.r_[force, torque]
            mujoco.mj_applyFT(m, d, force, torque, d.xpos[body], body, d.qfrc_applied)
        d.ctrl[:] = self.hand_targets + d.qfrc_bias[self.hand_v] / 1.5
        mujoco.mj_step(m, d)

    def _contacts(self):
        self.contact_forces[:] = 0.
        self.robot_collision = 0.
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = [self.model.geom_bodyid[g] for g in (contact.geom1, contact.geom2)]
            sides = [next((s for s, group in enumerate(self.body_groups) if body in group), -1) for body in bodies]
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            if contact.geom1 in self.wheel_geoms or contact.geom2 in self.wheel_geoms:
                for side in sides:
                    if side >= 0:
                        self.contact_forces[side] += np.linalg.norm(wrench[:3])
            elif sides[0] >= 0 and sides[1] >= 0 and sides[0] != sides[1]:
                self.robot_collision = max(self.robot_collision, np.linalg.norm(wrench[:3]))

    def _material_phase(self):
        local = (self.data.site_xpos[self.grip]-self.center) @ self.frame
        return np.arctan2(local[:, 2], local[:, 1]) - self.data.qpos[self.wheel_q]

    def _observation(self):
        d = self.data
        positions = (d.site_xpos[self.grip]-self.center) @ self.frame
        rotations = np.array([self.frame.T @ d.site_xmat[s].reshape(3, 3) for s in self.grip])
        pose = np.c_[positions/.2, rotations[:, :, :2].reshape(2, 6)].ravel()
        velocities, errors = [], []
        for side, body in enumerate(self.hand_body):
            mujoco.mj_jacBody(self.model, d, self.jacp, self.jacr, body)
            velocities.extend(self.frame.T @ (self.jacp @ d.qvel))
            velocities.extend(self.frame.T @ (self.jacr @ d.qvel))
            errors.extend(self.frame.T @ (self.positions[side]-d.site_xpos[self.grip[side]])/.02)
            actual = d.site_xmat[self.grip[side]].reshape(3, 3)
            errors.extend(self.frame.T @ Rotation.from_matrix(self.rotations[side] @ actual.T).as_rotvec()/.2)
        preview, velocity = self._reference(self.time + np.arange(6)*.1)
        angle = d.qpos[self.wheel_q]
        geometry = np.r_[self.center/.8, self.frame[:, :2].ravel(), 1., 1.]
        obs = np.r_[d.qpos[self.hand_q], d.qvel[self.hand_v]/5.,
                    (self.hand_targets-d.qpos[self.hand_q])/.15, self.previous_action,
                    pose, velocities, errors, angle, d.qvel[self.wheel_v],
                    (preview-angle)/np.deg2rad(10), velocity[0]/np.deg2rad(20), geometry,
                    self.arm_guard.observation() if self.arm_guard is not None else np.zeros(15)]
        if obs.shape != (OBSERVATION_DIM,) or not np.isfinite(obs).all():
            raise ValueError("Invalid hand-only observation")
        return obs.astype(np.float32)

    def _info(self):
        target, velocity = self._reference(self.time)
        actual = self.data.site_xpos[self.grip]
        material = self._material_phase()-self.initial_material
        drift = np.rad2deg(np.arctan2(np.sin(material), np.cos(material)))
        info = dict(time=self.time, target_deg=float(np.rad2deg(target)),
                    target_velocity_deg_s=float(np.rad2deg(velocity)),
                    wheel_deg=float(np.rad2deg(self.data.qpos[self.wheel_q])),
                    error_deg=float(np.rad2deg(target-self.data.qpos[self.wheel_q])),
                    left_contact_n=float(self.contact_forces[0]), right_contact_n=float(self.contact_forces[1]),
                    max_wrist_error_mm=float(1000*np.linalg.norm(self.positions-actual, axis=1).max()),
                    max_material_phase_drift_deg=float(np.abs(drift).max()),
                    left_wrist_force_n=float(np.linalg.norm(self.applied_wrenches[0, :3])),
                    right_wrist_force_n=float(np.linalg.norm(self.applied_wrenches[1, :3])),
                    left_wrist_torque_nm=float(np.linalg.norm(self.applied_wrenches[0, 3:])),
                    right_wrist_torque_nm=float(np.linalg.norm(self.applied_wrenches[1, 3:])),
                    hand_collision_peak_n=float(self.robot_collision), termination_reason="")
        info.update(self.arm_guard.info() if self.arm_guard is not None else dict(
            arm_reachability_enabled=False, arm_accepted_fraction=1., arm_filter_reason='',
            arm_clipped_steps=0, arm_rejected_steps=0, arm_min_joint_margin_deg=float('nan'),
            arm_min_guard_clearance_mm=float('nan'), arm_min_singular_value=float('nan'),
            arm_actual_pose_valid=True, arm_actual_pose_reason=''))
        return info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Expected 24 finite hand/wrist actions")
        action = np.clip(action, -1, 1)
        previous = self.previous_action.copy()
        self._integrate_action(action)
        for _ in range(self.substeps):
            self._physics_step()
        mujoco.mj_forward(self.model, self.data)
        self.time += self.dt
        self.previous_action = action.copy()
        self._contacts()
        if self.arm_guard is not None:
            self.arm_guard.assess_actual(self.data.site_xpos[self.grip],
                                         self.data.site_xmat[self.grip].reshape(2, 3, 3))
        info = self._info()
        error = np.deg2rad(info['error_deg'])
        target, target_velocity = self._reference(self.time)
        reward = 2*np.exp(-(error/np.deg2rad(4))**2)
        reward += .5*np.exp(-((target_velocity-self.data.qvel[self.wheel_v])/np.deg2rad(5))**2)
        reward += .1*np.any(self.contact_forces > .5)
        reward -= .01*np.mean((action-previous)**2) + .002*np.mean(action**2)
        # Holding is explicitly evaluated separately from tracking. No fixed
        # material-point reward is imposed for the eventual regrasp task.
        if abs(target_velocity) < np.deg2rad(.1):
            wrist_speed = 0.
            for body in self.hand_body:
                mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, body)
                wrist_speed += np.linalg.norm(self.jacp @ self.data.qvel) + .05*np.linalg.norm(self.jacr @ self.data.qvel)
            reward -= .2*wrist_speed
        terminated = not np.isfinite(self.data.qpos).all() or abs(self.data.qpos[self.wheel_q]) > np.deg2rad(110)
        if terminated:
            info['termination_reason'] = 'invalid_state_or_wheel_envelope'
        if self.arm_guard is not None and not self.arm_guard.actual_valid:
            terminated = True
            reward -= 2.
            info['termination_reason'] = 'actual_wrist_outside_arm_envelope'
        return self._observation(), float(reward), bool(terminated), self.time+1e-9 >= self.config.episode_seconds, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=540, width=960)
        self.renderer.update_scene(self.data, camera="hands")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
