"""Goal-conditioned hand/wrist RL, with a passive, contact-driven wheel."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .simulation import SimulationSupport
from .controller import ControlConfig, DifferentialController
from .observations import FRAME_DIM, PRIVILEGED_DIM, actor_frame



@dataclass
class DexTaskConfig:
    amplitude_deg: float = 10.0
    target_mode: str = "step"
    episode_seconds: float = 15.0
    frequency: float = .02
    policy_hz: int = 20
    history_steps: int = 8
    settle_seconds: float = 1.2
    target_speed_deg: float = 20.0
    randomization_strength: float = 0.0
    no_contact: bool = False
    controller: ControlConfig = field(default_factory=ControlConfig)

    def validate(self):
        if self.target_mode not in ("hold", "step", "sine", "sequence", "mixed"):
            raise ValueError("Unknown target mode")
        if self.policy_hz <= 0 or self.history_steps < 1 or self.episode_seconds <= 0:
            raise ValueError("Frequencies, history and episode duration must be positive")
        if self.settle_seconds < 0 or self.frequency < 0 or self.target_speed_deg <= 0:
            raise ValueError("Invalid task timing")
        if not 0 <= self.amplitude_deg <= 30:
            raise ValueError("Task target amplitude must be within [0, 30] degrees")
        if not 0 <= self.randomization_strength <= 1:
            raise ValueError("randomization_strength must be within [0, 1]")


class DexSteeringEnv(SimulationSupport):
    """Reuse visualization/contact readers, never the baseline control or reset."""
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}
    controller_class = None
    actor_frame_dim = FRAME_DIM

    def __init__(self, config=None, render_mode=None, scene_path=None,
                 camera="hands", width=960, height=540):
        gym.Env.__init__(self)
        self.config = config or DexTaskConfig()
        self.config.validate()
        if scene_path is None or self.controller_class is None:
            raise ValueError("Use the concrete contact-residual task with an explicit scene")
        self.scene_path = Path(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.sensor_data = mujoco.MjData(self.model)
        self.controller = self.controller_class(self.model, self.config.controller)
        self.wheel_q = int(self.model.joint("wheel_joint").qposadr[0])
        self.wheel_v = int(self.model.joint("wheel_joint").dofadr[0])
        self.wheel_body = self.model.body("steering_wheel").id
        self.wheel_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.wheel_body))
        if self.model.joint("wheel_joint").id in self.model.actuator_trnid[:, 0]:
            raise ValueError("Wheel must have no actuator")
        self.hand_body_groups = []
        for side in ("left", "right"):
            wrist = self.model.body(f"{side}_wrist_yaw_link").id
            bodies = {wrist}
            for i in range(wrist + 1, self.model.nbody):
                if self.model.body_parentid[i] in bodies:
                    bodies.add(i)
            self.hand_body_groups.append(bodies)
        self.pad_ids = [[self.model.site(f"{side}_{f}_pad").id
                         for f in ("thumb", "index", "middle", "ring", "pinky")]
                        for side in ("left", "right")]
        self.dependent_q = np.array([self.model.jnt_qposadr[j] for j in range(self.model.njnt)
                                    if self.model.jnt_qposadr[j] not in self.controller.motor_q
                                    and self.model.jnt_qposadr[j] != self.wheel_q])
        self.dt = 1 / self.config.policy_hz
        self.robot_dt = 1 / self.config.controller.robot_hz
        self.substeps = round(self.dt / self.model.opt.timestep)
        if abs(self.substeps * self.model.opt.timestep - self.dt) > 1e-9:
            raise ValueError("Policy frequency must divide the physics frequency")
        # 50 Hz robot and 20 Hz policy are scheduled on the common 500 Hz clock.
        self.robot_substeps = round(self.robot_dt / self.model.opt.timestep)
        if abs(self.robot_substeps * self.model.opt.timestep - self.robot_dt) > 1e-9:
            raise ValueError("Robot frequency must divide the physics frequency")
        self.render_mode, self.camera = render_mode, camera
        self.width, self.height = width, height
        self.renderer = self.viewer = self.viewer_thread = None
        self.render_options = mujoco.MjvOption()
        self.render_options.geomgroup[3] = 0
        self.action_space = spaces.Box(-1, 1, (18,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(-np.inf, np.inf, (self.config.history_steps, self.actor_frame_dim), dtype=np.float32),
            "privileged": spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), dtype=np.float32)})
        self.rim_radius = .14
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_friction = self.model.dof_frictionloss.copy()
        self.nominal_geom_friction = self.model.geom_friction.copy()
        self.nominal_geom_size = self.model.geom_size.copy()
        self.nominal_ctrlrange = self.model.actuator_ctrlrange.copy()
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_eq_data = self.model.eq_data.copy()
        if self.config.no_contact:
            for g in self.wheel_geoms:
                self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 0
        self.contact_forces = np.zeros(2)
        self.enclosure = self.slip = np.zeros(2)
        self.previous_action = self.controller.neutral_action()
        self.target_angle = self.target_velocity = self.time = 0.0
        self.external_target = None
        self.external_target_velocity = 0.0
        self.history = deque(maxlen=self.config.history_steps)
        self.actor_queue = deque(maxlen=3)
        self.action_queue = deque(maxlen=3)
        self.action_delay = self.observation_delay = 0
        self.physics_steps = 0
        self._latest_frame = np.zeros(self.actor_frame_dim, dtype=np.float32)

    def set_randomization_strength(self, strength):
        if not 0 <= strength <= 1:
            raise ValueError("Randomization strength must be within [0, 1]")
        self.config.randomization_strength = float(strength)

    def _randomize(self):
        m, c = self.model, self.controller
        m.dof_damping[:] = self.nominal_damping
        m.dof_frictionloss[:] = self.nominal_friction
        m.geom_friction[:] = self.nominal_geom_friction
        m.geom_size[:] = self.nominal_geom_size
        m.actuator_ctrlrange[:] = self.nominal_ctrlrange
        m.body_mass[:] = self.nominal_mass
        m.body_inertia[:] = self.nominal_inertia
        m.eq_data[:] = self.nominal_eq_data
        c.kp, c.kd = c.nominal_kp.copy(), c.nominal_kd.copy()
        strength = self.config.randomization_strength
        def scale(width):
            return self.np_random.uniform(1 - width * strength, 1 + width * strength)
        m.dof_damping[self.wheel_v] *= scale(.4)
        m.dof_frictionloss[self.wheel_v] *= scale(.5)
        m.dof_damping[c.motor_v] *= scale(.3)
        m.dof_frictionloss[c.motor_v] *= scale(.3)
        m.eq_data[:, 1] *= scale(.015)
        m.geom_friction[:, 0] *= scale(.3)
        for g in self.wheel_geoms:
            if m.geom(g).name.startswith("rim_"):
                m.geom_size[g, 0] *= scale(.05)
        # Model uncertainty includes the passive wheel inertia and robot response.
        mass_scale = scale(.2)
        m.body_mass[self.wheel_body] *= mass_scale
        m.body_inertia[self.wheel_body] *= mass_scale
        m.actuator_ctrlrange[:] *= scale(.2)
        c.kp *= scale(.2)
        c.kd *= scale(.2)
        self.action_delay = int(self.np_random.integers(0, round(2 * strength) + 1))
        self.observation_delay = int(self.np_random.integers(0, round(2 * strength) + 1))
        mujoco.mj_setConst(m, self.data)

    def _raw_target(self):
        amp = np.deg2rad(self.config.amplitude_deg)
        if self.external_target is not None:
            return self.external_target
        mode = self.episode_mode
        if mode == "hold":
            return 0.0
        if mode == "step":
            return self.step_target
        if mode == "sine":
            return amp * np.sin(2 * np.pi * self.frequency * self.time + self.phase)
        if mode == "sequence":
            section = int(self.time / 5) % 6
            return amp * [0, 1, 1, 0, -1, 0][section]
        index = np.searchsorted(self.target_times, self.time, side="right")
        return self.target_values[min(index, len(self.target_values) - 1)]

    def _update_target(self, dt=None):
        dt = self.dt if dt is None else dt
        old = self.target_angle
        maximum = np.deg2rad(self.config.target_speed_deg) * dt
        self.target_angle += float(np.clip(self._raw_target() - old, -maximum, maximum))
        self.target_velocity = (self.target_angle - old) / dt
        if self.external_target is not None and abs(self._raw_target() - old) <= maximum:
            self.target_velocity = float(np.clip(self.external_target_velocity,
                                                 -np.deg2rad(self.config.target_speed_deg),
                                                 np.deg2rad(self.config.target_speed_deg)))

    def set_target(self, angle, velocity=0.0):
        if not np.isfinite([angle, velocity]).all() or abs(angle) > np.deg2rad(30):
            raise ValueError("Task external target must be finite and within +/-30 degrees")
        # The angle is an endpoint command; rate limiting remains explicit.
        self.external_target = float(angle)
        self.external_target_velocity = float(velocity)
        return self._observation()

    def _grasp_metrics(self):
        c = self.controller
        for s in range(2):
            local = c.axis_frame.T @ (self.data.site_xpos[c.grip_ids[s]] - c.center)
            radius = max(np.linalg.norm(local[1:]), 1e-8)
            rim_point = c.center + c.axis_frame @ np.r_[0, self.rim_radius * local[1:] / radius]
            points = np.vstack([self.data.site_xpos[c.grip_ids[s]], self.data.site_xpos[self.pad_ids[s]]])
            try:
                hull = ConvexHull(points)
                outside = max(0., float(np.max(hull.equations[:, :3] @ rim_point + hull.equations[:, 3])))
                self.enclosure[s] = np.exp(-(outside / .015) ** 2)
            except QhullError:
                self.enclosure[s] = 0
        relative = c.measured_phase(self.data) - self.initial_hand_phase - self.wheel_angle + self.initial_wheel
        self.slip = np.arctan2(np.sin(relative), np.cos(relative))

    def _sensor_frame(self, initialize=False):
        c = self.controller
        strength = self.config.randomization_strength
        q = self.data.qpos[c.motor_q] + self.np_random.normal(0, .002 * strength, 26)
        angle = self.wheel_angle + self.np_random.normal(0, .003 * strength)
        if initialize:
            self.sensor_velocity = np.zeros(26)
            self.sensor_wheel_velocity = 0.0
        else:
            self.sensor_velocity = .5 * self.sensor_velocity + .5 * (q - self.last_sensor_q) / self.dt
            self.sensor_wheel_velocity = .5 * self.sensor_wheel_velocity + .5 * (angle - self.last_sensor_angle) / self.dt
        self.last_sensor_q, self.last_sensor_angle = q.copy(), angle
        self.sensor_data.qpos[:] = self.data.qpos
        self.sensor_data.qpos[c.motor_q] = q
        mujoco.mj_kinematics(self.model, self.sensor_data)
        position = (self.sensor_data.site_xpos[c.grip_ids] - c.center) @ c.axis_frame
        rotation = np.array([c.axis_frame.T @ self.sensor_data.site_xmat[site].reshape(3, 3) for site in c.grip_ids])
        return actor_frame(q, self.sensor_velocity, c.target, self.previous_action, position,
                           rotation, angle, self.sensor_wheel_velocity,
                           self.target_angle, self.target_velocity, self.rim_radius)

    def _privileged(self):
        c, m = self.controller, self.model
        values = np.r_[self.data.qfrc_actuator[c.motor_v] / self.nominal_ctrlrange[:, 1],
                       self.data.qpos[self.dependent_q], self.contact_forces / 100,
                       self.interval_peak / 100, self.interval_other / 100,
                       self.enclosure, self.slip,
                       m.dof_damping[self.wheel_v], m.dof_frictionloss[self.wheel_v], m.body_mass[self.wheel_body],
                       np.mean(c.kp / c.nominal_kp), np.mean(c.kd / c.nominal_kd),
                       self.action_delay / 2, self.observation_delay / 2,
                       np.mean(m.geom_friction[list(self.wheel_geoms), 0])]
        if values.shape != (PRIVILEGED_DIM,):
            raise RuntimeError(f"Privileged observation shape changed: {values.shape}")
        return values.astype(np.float32)

    def _observation(self):
        return {"actor": np.array(self.history, dtype=np.float32), "privileged": self._privileged()}

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize()
        c = self.controller
        self.data.qpos[c.arm_q] = c.initial_arm
        self.data.qpos[c.hand_q] = c.open_hand
        self._set_initial_mimic()
        mujoco.mj_forward(self.model, self.data)
        warmup = round(self.config.settle_seconds / self.robot_dt)
        for k in range(warmup):
            fraction = min(1., (k + 1) / max(1., .65 * warmup))
            target = np.r_[c.initial_arm, c.open_hand + fraction * (c.closed_hand - c.open_hand)]
            for _ in range(self.robot_substeps):
                self.data.ctrl[:] = c.torque(self.data, target)
                self._physics_step()
        mujoco.mj_forward(self.model, self.data)
        c.reset_reference(self.data)
        self.initial_wheel = self.wheel_angle
        self.initial_hand_phase = c.measured_phase(self.data)
        self.time = self.target_angle = self.target_velocity = 0.0
        self.physics_steps = 0
        self.data.time = 0.0
        self.external_target = None
        self.external_target_velocity = 0.0
        amp = np.deg2rad(self.config.amplitude_deg)
        self.step_target = float(self.np_random.uniform(-amp, amp))
        if options and "target_deg" in options:
            value = float(options["target_deg"])
            if abs(value) > self.config.amplitude_deg:
                raise ValueError("Reset target exceeds configured amplitude")
            self.step_target = np.deg2rad(value)
        self.phase = 0.0 if self.config.randomization_strength == 0 else float(self.np_random.uniform(-.3, .3))
        self.episode_mode = self.config.target_mode
        self.frequency = self.config.frequency
        if self.config.target_mode == "mixed":
            self.frequency = float(self.np_random.uniform(.01, .10))
            self.episode_mode = str(self.np_random.choice(["sine", "sequence", "random_steps"]))
        n = max(2, int(self.config.episode_seconds / 2) + 2)
        self.target_times = np.cumsum(self.np_random.uniform(3, 8, n))
        self.target_values = self.np_random.uniform(-amp, amp, n + 1)
        self.target_values[::3] = 0
        self.previous_action = c.neutral_action()
        self.action_queue.clear()
        self.actor_queue.clear()
        for _ in range(3):
            self.action_queue.append(self.previous_action.copy())
        self.enclosure, self.slip = np.zeros(2), np.zeros(2)
        _, peak, other = self._contact_metrics()
        self.interval_peak, self.interval_other = peak, other
        self.interval_robot_peak = 0.0
        self.interval_command_speed = self.interval_command_acceleration = 0.0
        self.hold_seconds = self.max_hold_seconds = self.lost_contact_seconds = 0.0
        self._grasp_metrics()
        frame = self._sensor_frame(initialize=True)
        for _ in range(3):
            self.actor_queue.append(frame.copy())
        self.history.clear()
        for _ in range(self.config.history_steps):
            self.history.append(frame.copy())
        self._latest_frame = frame.copy()
        return self._observation(), self._info(peak, other)

    def _physics_step(self):
        """Overridable physics hook; default preserves the original integrator."""
        mujoco.mj_step(self.model, self.data)

    def _advance_action(self, action):
        c = self.controller
        self.interval_peak = self.interval_other = self.interval_robot_peak = 0.0
        self.interval_command_speed = self.interval_command_acceleration = 0.0
        for _ in range(self.substeps):
            if self.physics_steps % self.robot_substeps == 0:
                before = c.velocity.copy()
                c.update(self.data, action, self.robot_dt)
                self.interval_command_speed = max(self.interval_command_speed, float(np.max(np.abs(c.velocity[:14]))))
                self.interval_command_acceleration = max(self.interval_command_acceleration,
                                                         float(np.max(np.abs(c.velocity[:14] - before[:14]))) / self.robot_dt)
            self.data.ctrl[:] = c.torque(self.data, c.target)
            self._physics_step()
            self.physics_steps += 1
            _, peak, other = self._contact_metrics()
            self.interval_peak = max(self.interval_peak, peak)
            self.interval_other = max(self.interval_other, other)
            vector = np.zeros(6)
            for k in range(self.data.ncon):
                con = self.data.contact[k]
                if self.model.geom_contype[con.geom1] & 2 and self.model.geom_contype[con.geom2] & (2 | 8):
                    mujoco.mj_contactForce(self.model, self.data, k, vector)
                    self.interval_robot_peak = max(self.interval_robot_peak, float(np.linalg.norm(vector[:3])))

    def _info(self, peak, other):
        error = self.target_angle - self.wheel_angle
        intended_position, _ = self.controller.poses()
        actual_error = np.linalg.norm(self.data.site_xpos[self.controller.grip_ids] - intended_position, axis=1)
        return {"time": self.time, "target_angle": self.target_angle, "wheel_angle": self.wheel_angle,
                "target_velocity": self.target_velocity,
                "error_deg": float(np.rad2deg(error)), "wheel_velocity": float(self.data.qvel[self.wheel_v]),
                "left_force": float(self.contact_forces[0]), "right_force": float(self.contact_forces[1]),
                "peak_force": float(peak), "non_hand_wheel_force": float(other),
                "robot_collision_peak": self.interval_robot_peak,
                "left_enclosure": float(self.enclosure[0]), "right_enclosure": float(self.enclosure[1]),
                "max_grasp_drift_deg": float(np.rad2deg(np.max(np.abs(self.slip)))),
                "command_speed_rad_s": self.interval_command_speed,
                "command_acceleration_rad_s2": self.interval_command_acceleration,
                "qp_position_error_mm": float(np.max(self.controller.qp_error) * 1000),
                "actual_palm_error_mm": float(np.max(actual_error) * 1000),
                "max_motor_target_gap_rad": float(np.max(np.abs(self.controller.target - self.data.qpos[self.controller.motor_q]))),
                "actual_arm_speed_rad_s": float(np.max(np.abs(self.data.qvel[self.controller.arm_v]))),
                "left_wrist_phase_deg": float(np.rad2deg(self.controller.phase[0])),
                "right_wrist_phase_deg": float(np.rad2deg(self.controller.phase[1])),
                "qp_failures": self.controller.qp_failures,
                "phase_constraint_failures": self.controller.phase_constraint_failures,
                "safety_overrides": self.controller.safety_overrides,
                "hold_seconds": self.hold_seconds, "max_hold_seconds": self.max_hold_seconds,
                "success": bool(self.max_hold_seconds >= 10),
                "finite": bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError("action must have 18 finite values")
        action = np.clip(action, -1, 1)
        self.action_queue.append(action.copy())
        applied = list(self.action_queue)[-1 - self.action_delay]
        self._advance_action(applied)
        self.time += self.dt
        self._update_target()
        self._grasp_metrics()
        error = self.target_angle - self.wheel_angle
        contact = self.contact_forces > .2
        self.lost_contact_seconds = 0.0 if np.any(contact) else self.lost_contact_seconds + self.dt
        stable = (abs(error) <= np.deg2rad(2) and abs(self.data.qvel[self.wheel_v]) <= np.deg2rad(2)
                  and abs(self.target_velocity) < 1e-6 and abs(self._raw_target() - self.target_angle) < 1e-6
                  and np.all(contact) and np.all(self.enclosure >= .5))
        self.hold_seconds = self.hold_seconds + self.dt if stable else 0.0
        self.max_hold_seconds = max(self.max_hold_seconds, self.hold_seconds)
        self.previous_action = action.copy()
        frame = self._sensor_frame()
        self.actor_queue.append(frame)
        observed = list(self.actor_queue)[-1 - self.observation_delay].copy()
        # The command source is local and current even when sensor data is delayed.
        # These fixed fields belong to the shared first 120 values; subclasses
        # may append execution-reference state without moving the goal fields.
        observed[116:118] = [self.target_angle, self.target_velocity]
        observed[118] = self.target_angle - observed[114]
        self.history.append(observed)
        self._latest_frame = frame
        info = self._info(self.interval_peak, self.interval_other)
        reason = ""
        if not info["finite"]:
            reason = "nonfinite_state"
        elif self.interval_peak > 300 or self.interval_other > 150 or self.interval_robot_peak > 150:
            reason = "simulated_load_limit"
        elif abs(self.wheel_angle) > np.deg2rad(50):
            reason = "wheel_workspace"
        elif self.lost_contact_seconds > 1.0 and not self.config.no_contact:
            reason = "lost_grasp"
        elif self.controller.qp_failures > 5:
            reason = "qp_failure"
        elif self.controller.phase_constraint_failures:
            reason = "fixed_grasp_constraint_infeasible"
        terminated = bool(reason)
        truncated = self.time + 1e-9 >= self.config.episode_seconds
        info["termination_reason"] = reason
        return self._observation(), 0.0, terminated, bool(truncated), info
