"""Versioned wrist/hand teacher task with persistent load and audited limits."""
from dataclasses import dataclass, field, replace
import mujoco
import numpy as np

from .legacy.regrasp.env import RegraspEnv, RegraspConfig
from .legacy.regrasp.controller import RegraspController, RegraspControlConfig
from ..wheel_load import WheelLoadConfig


@dataclass
class TaskConfig(RegraspConfig):
    strict_safety: bool = True
    centering_stiffness: float = .3
    relative_reference: bool = True
    randomization_strength: float = 0.
    shaping_weight: float = 0.
    target_mode: str = "mixed"
    angle_reward_sigma_deg: float = 4.
    hand_action_mode: str = "absolute"
    hand_delta_rad: float = .03
    hand_leak_alpha: float = .95
    hand_joint_lead_rad: float | None = None
    controller: RegraspControlConfig = field(default_factory=RegraspControlConfig)


class StructuredController(RegraspController):
    def update(self, data, action, dt):
        self.hand_measured = data.qpos[self.hand_q].copy()
        self.hand_dt = dt
        super().update(data, action, dt)

    def hand_targets(self, action):
        if getattr(self,"hand_action_mode","absolute") == "absolute":
            return super().hand_targets(action)
        # Alpha and delta are defined per 20Hz policy period, then converted
        # to the actual 50Hz command update. Integrator state is already in
        # the actor through q and (motor_target-q), with no hidden reset anchor.
        ratio = self.hand_dt/.05
        target = (self.hand_measured + self.hand_leak_alpha**ratio*(self.target[14:]-self.hand_measured)
                  + self.hand_delta_rad*ratio*np.asarray(action))
        return np.clip(target,self.hand_limits[:,0]+self.config.joint_margin,
                       self.hand_limits[:,1]-self.config.joint_margin)

    def _bounds(self, previous, speed, acceleration, limits, measured, dt):
        # Preserve the original hand mapping while reserving a braking margin
        # for arms. The actual-state hard gate remains exactly two degrees.
        hand_lead=getattr(self,"hand_joint_lead_rad",None)
        if previous.start==14 and hand_lead is not None:
            original=self.config
            self.config=replace(original,joint_lead=hand_lead)
            try:
                return super()._bounds(previous,speed,acceleration,limits,measured,dt)
            finally:
                self.config=original
        if getattr(self, "strict_safety", False) and previous.start == 0:
            limits = limits.copy()
            reserve = np.deg2rad(3.) - self.config.joint_margin
            limits[:, 0] += reserve
            limits[:, 1] -= reserve
        return super()._bounds(previous, speed, acceleration, limits, measured, dt)


class SteeringTeacherEnv(RegraspEnv):
    controller_class = StructuredController

    def __init__(self, config=None, **kwargs):
        config = config or TaskConfig()
        if config.hand_action_mode not in ("absolute","leaky") or not 0<config.hand_leak_alpha<=1 or not 0<config.hand_delta_rad<=.1:
            raise ValueError("Invalid versioned hand target generation")
        if config.hand_joint_lead_rad is not None and (not np.isfinite(config.hand_joint_lead_rad) or not 0<config.hand_joint_lead_rad<=.25):
            raise ValueError("Hand target lead must lie within (0,.25] rad")
        self.load = WheelLoadConfig(centering_stiffness_nm_per_rad=config.centering_stiffness)
        self.measuring = False
        self.strict_reason = ""
        self.interval_min_margin = np.inf
        self.interval_collision = self.interval_error_peak = 0.
        self.integrated_ticks = 0
        super().__init__(config, **kwargs)
        self.controller.strict_safety = config.strict_safety
        self.controller.hand_action_mode = config.hand_action_mode
        self.controller.hand_delta_rad = config.hand_delta_rad
        self.controller.hand_leak_alpha = config.hand_leak_alpha
        self.controller.hand_joint_lead_rad = config.hand_joint_lead_rad
        if config.strict_safety:
            self.controller.config.joint_lead = np.deg2rad(5.)
        self.limited_ids = np.flatnonzero(self.model.jnt_limited)
        self.load.configure_joint(self.model, self.model.joint("wheel_joint").id)
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_friction = self.model.dof_frictionloss.copy()

    def _make_commands(self):
        if not self.config.relative_reference:
            return super()._make_commands()
        count = int(np.ceil((self.config.episode_seconds+.6)/self.dt))+2
        times = np.arange(count)*self.dt
        start = self.wheel_angle
        sign = float(self.reset_options.get("direction", self.np_random.choice([-1, 1])))
        self.episode_direction = int(sign)
        mode = self.reset_options.get("mode", self.config.target_mode)
        if mode == "mixed":
            mode = str(self.np_random.choice(["hold", "step", "sine", "sequence"], p=[.25, .4, .1, .25]))
        self.episode_mode = mode
        amplitude = np.deg2rad(self.config.amplitude_deg)
        if mode == "hold":
            return np.full(count, start), np.zeros(count)
        if mode == "sine":
            frequency = min(.06, np.deg2rad(self.config.target_speed_deg)/max(2*np.pi*amplitude, 1e-9))
            return start+sign*amplitude*np.sin(2*np.pi*frequency*times), sign*amplitude*2*np.pi*frequency*np.cos(2*np.pi*frequency*times)
        endpoints = start+sign*amplitude*np.array([1., 0., -1., 0.])
        if mode == "step":
            endpoints = endpoints[:1]
        angles, velocities = np.full(count, start), np.zeros(count)
        cursor, value = .5, start
        for endpoint in endpoints:
            duration = max(1., 1.875*abs(endpoint-value)/np.deg2rad(self.config.target_speed_deg),
                           np.sqrt(5.8*abs(endpoint-value)/np.deg2rad(self.config.target_acceleration_deg_s2)))
            u = np.clip((times-cursor)/duration, 0., 1.)
            active = times >= cursor
            angles[active] = value+(endpoint-value)*(10*u[active]**3-15*u[active]**4+6*u[active]**5)
            velocities[active] = (endpoint-value)/duration*(30*u[active]**2-60*u[active]**3+30*u[active]**4)
            cursor += duration+1.
            value = endpoint
        return angles, velocities

    def reset(self, seed=None, options=None):
        self.measuring = False
        self.strict_reason = ""
        obs, info = super().reset(seed=seed, options=options)
        self.measuring = True
        self.interval_min_margin = self._arm_margin()
        info.update(strict_reason="", load_stiffness=self.config.centering_stiffness)
        return obs, info

    def _reward(self, previous):
        reward = super()._reward(previous)
        sigma = self.config.angle_reward_sigma_deg
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Angle reward width must be positive and finite")
        if sigma != 4.:
            error = self.target_angle-self.wheel_angle
            term = float(2*np.exp(-(error/np.deg2rad(sigma))**2)
                         -.3*min((error/np.deg2rad(10))**2,9))
            reward += term-self.reward_terms["angle"]
            self.reward_terms["angle"] = term
        return reward

    def _arm_margin(self):
        c = self.controller
        q = self.data.qpos[c.arm_q]
        return float(np.min(np.minimum(q-c.limits[:, 0], c.limits[:, 1]-q)))

    def _physics_step(self):
        self.data.qfrc_applied[self.wheel_v] = self.load.centering_torque(self.wheel_angle)
        mujoco.mj_step(self.model, self.data)
        if not self.measuring:
            return
        self.integrated_ticks += 1
        self.interval_min_margin = min(self.interval_min_margin, self._arm_margin())
        if len(self.command_angles):
            index = min(self.data.time / self.dt, len(self.command_angles)-1)
            left = int(index)
            right = min(left+1, len(self.command_angles)-1)
            reference = float(self.command_angles[left]+(index-left)*(self.command_angles[right]-self.command_angles[left]))
            self.interval_error_peak = max(self.interval_error_peak, abs(reference-self.wheel_angle))
        collision = self._collision_force()
        self.interval_collision = max(self.interval_collision, collision)
        if not self.config.strict_safety:
            return
        m, d = self.model, self.data
        values = d.qpos[m.jnt_qposadr[self.limited_ids]]
        limits = m.jnt_range[self.limited_ids]
        if not all(np.isfinite(x).all() for x in (d.qpos, d.qvel, d.actuator_force)):
            self.strict_reason = "nonfinite_physics"
        elif np.any(d.warning.number):
            self.strict_reason = "mujoco_warning"
        elif self.interval_min_margin < np.deg2rad(2.):
            self.strict_reason = "arm_margin"
        elif np.any(values < limits[:, 0]-.02) or np.any(values > limits[:, 1]+.02):
            self.strict_reason = "joint_limit"
        elif collision > 1.:
            self.strict_reason = "collision"
        elif abs(self.wheel_angle) >= np.deg2rad(440):
            self.strict_reason = "wheel_envelope"
        elif np.any(d.actuator_force < m.actuator_forcerange[:, 0]-1e-6) or np.any(d.actuator_force > m.actuator_forcerange[:, 1]+1e-6):
            self.strict_reason = "actuator_cap"

    def _collision_force(self):
        maximum = 0.
        wrench = np.zeros(6)
        for k in range(self.data.ncon):
            contact = self.data.contact[k]
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            allowed = any((a == self.wheel_body and b in group) or
                          (b == self.wheel_body and a in group) or
                          (a in group and b in group) for group in self.hand_body_groups)
            if not allowed:
                mujoco.mj_contactForce(self.model, self.data, k, wrench)
                maximum = max(maximum, float(np.linalg.norm(wrench[:3])))
        return maximum

    def _advance_action(self, action):
        c = self.controller
        self.strict_reason = ""
        self.integrated_ticks = 0
        self.interval_min_margin = self._arm_margin()
        self.interval_collision = self.interval_error_peak = 0.
        self.interval_peak = self.interval_other = self.interval_robot_peak = 0.
        self.interval_command_speed = self.interval_command_acceleration = 0.
        for _ in range(self.substeps):
            if self.physics_steps % self.robot_substeps == 0:
                before = c.velocity.copy()
                c.update(self.data, action, self.robot_dt)
                self.interval_command_speed = max(self.interval_command_speed, float(np.max(abs(c.velocity[:14]))))
                self.interval_command_acceleration = max(self.interval_command_acceleration, float(np.max(abs(c.velocity[:14]-before[:14])))/self.robot_dt)
            self.data.ctrl[:] = c.torque(self.data, c.target)
            self._physics_step()
            self.physics_steps += 1
            _, peak, other = self._contact_metrics()
            self.interval_peak = max(self.interval_peak, peak)
            self.interval_other = max(self.interval_other, other)
            self.interval_robot_peak = max(self.interval_robot_peak, self.interval_collision)
            if self.strict_reason:
                break
        # The inherited step adds one policy dt; preserve actual time on a
        # substep safety termination rather than pretending a full step elapsed.
        self.time -= self.dt-self.integrated_ticks*self.model.opt.timestep

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.strict_reason:
            if not terminated:
                reward -= 10.
            terminated = True
            info["termination_reason"] = self.strict_reason
        elapsed = self.integrated_ticks*self.model.opt.timestep
        info.update(physics_peak_error_deg=float(np.rad2deg(self.interval_error_peak)),
                    physics_min_arm_margin_deg=float(np.rad2deg(self.interval_min_margin)),
                    physics_collision_peak_n=self.interval_collision, elapsed_seconds=elapsed,
                    centering_torque_nm=self.load.centering_torque(self.wheel_angle))
        return obs, float(reward*.05), terminated, truncated, info
