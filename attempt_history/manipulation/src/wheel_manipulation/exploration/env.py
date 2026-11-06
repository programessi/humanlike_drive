"""Stage-one centralized task: physical preparation and smooth internal commands."""
from dataclasses import dataclass, field

from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..env import ContactResidualEnv, ContactTaskConfig
from ..common.env import DexSteeringEnv
from . import ACTOR_DIM, ACTION_DIM
from .controller import IndependentWristController, WristControlConfig


@dataclass
class ExplorationConfig(ContactTaskConfig):
    history_steps: int = 1
    episode_seconds: float = 15.
    amplitude_deg: float = 3.
    target_mode: str = "mixed"
    target_speed_deg: float = 5.
    target_acceleration_deg_s2: float = 40.
    randomization_strength: float = .15
    closure_jitter_rad: float = .04
    settle_jitter_seconds: float = .12
    controller: WristControlConfig = field(default_factory=WristControlConfig)

    def validate(self):
        super().validate()
        if self.history_steps != 1 or self.target_acceleration_deg_s2 <= 0:
            raise ValueError("GRU task uses one frame and positive reference acceleration")
        if min(self.closure_jitter_rad, self.settle_jitter_seconds) < 0:
            raise ValueError("Preparation jitter must be nonnegative")


class CentralizedWheelEnv(ContactResidualEnv):
    controller_class = IndependentWristController
    actor_frame_dim = ACTOR_DIM

    def __init__(self, config=None, **kwargs):
        self.command_angles = self.command_velocities = np.zeros(2)
        super().__init__(config or ExplorationConfig(), **kwargs)
        self.action_space = spaces.Box(-1, 1, (ACTION_DIM,), dtype=np.float32)
        self.observation_space["actor"] = spaces.Box(-np.inf, np.inf, (ACTOR_DIM,), dtype=np.float32)
        self.nominal_settle_seconds = self.config.settle_seconds
        self.preparation_closed = self.controller.nominal_closed.copy()
        self.reward_terms = {}

    def _randomize(self):
        super()._randomize()
        c = self.controller
        self.preparation_closed = np.clip(c.nominal_closed + self.np_random.uniform(
            -self.config.closure_jitter_rad, self.config.closure_jitter_rad, 12),
            c.hand_limits[:, 0] + .01, c.hand_limits[:, 1] - .01)
        c.closed_hand = self.preparation_closed.copy()
        self.config.settle_seconds = max(.6, self.nominal_settle_seconds + self.np_random.uniform(
            -self.config.settle_jitter_seconds, self.config.settle_jitter_seconds))

    def _make_commands(self):
        """Minimum-jerk segments; previews are commands, never future measurements."""
        count = int(np.ceil((self.config.episode_seconds + .6) / self.dt)) + 2
        times = np.arange(count) * self.dt
        amplitude = np.deg2rad(self.config.amplitude_deg)
        mode = self.episode_mode
        if mode == "hold":
            return np.zeros(count), np.zeros(count)
        if mode == "sine":
            frequency = min(self.frequency, self.config.target_speed_deg / max(1e-9, 2 * np.pi * self.config.amplitude_deg),
                            np.sqrt(self.config.target_acceleration_deg_s2 / max(1e-9, self.config.amplitude_deg)) / (2 * np.pi))
            return amplitude * np.sin(2 * np.pi * frequency * times), amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * times)
        angles, velocities = np.zeros(count), np.zeros(count)
        cursor, value, index = .75, 0., 0
        sequence = [amplitude, 0., -amplitude, 0.]
        while cursor <= times[-1]:
            if mode == "step":
                endpoint = self.step_target if index == 0 else value
            elif mode == "sequence":
                endpoint = sequence[index % len(sequence)]
            else:
                endpoint = float(self.np_random.uniform(-amplitude, amplitude))
            delta = endpoint - value
            duration = max(1., 1.875 * abs(delta) / np.deg2rad(self.config.target_speed_deg),
                           np.sqrt(5.8 * abs(delta) / np.deg2rad(self.config.target_acceleration_deg_s2)))
            u = np.clip((times - cursor) / duration, 0., 1.)
            active = times >= cursor
            angles[active] = value + delta * (10 * u[active] ** 3 - 15 * u[active] ** 4 + 6 * u[active] ** 5)
            velocities[active] = delta / duration * (30 * u[active] ** 2 - 60 * u[active] ** 3 + 30 * u[active] ** 4)
            cursor += duration + 2.
            value, index = endpoint, index + 1
        return angles, velocities

    def _update_target(self, dt=None):
        index = min(round(self.time / self.dt), len(self.command_angles) - 1)
        self.target_angle = float(self.command_angles[index])
        self.target_velocity = float(self.command_velocities[index])

    def _preview(self):
        index = round(self.time / self.dt)
        offsets = np.rint(np.arange(6) * .1 / self.dt).astype(int)
        return self.command_angles[np.minimum(index + offsets, len(self.command_angles) - 1)]

    def _sensor_frame(self, initialize=False):
        c = self.controller
        strength = self.config.randomization_strength
        q = self.data.qpos[c.motor_q] + self.np_random.normal(0, .002 * strength, 26)
        angle = self.wheel_angle + self.np_random.normal(0, .003 * strength)
        if initialize:
            self.sensor_velocity, self.sensor_wheel_velocity = np.zeros(26), 0.
        else:
            self.sensor_velocity = .5 * self.sensor_velocity + .5 * (q - self.last_sensor_q) / self.dt
            self.sensor_wheel_velocity = .5 * self.sensor_wheel_velocity + .5 * (angle - self.last_sensor_angle) / self.dt
        self.last_sensor_q, self.last_sensor_angle = q.copy(), angle
        # FK uses only measured active joint positions and known fixed geometry.
        # No simulated dependent finger position enters the actor's FK.
        self.sensor_data.qpos[:] = self.model.qpos0
        self.sensor_data.qpos[c.motor_q] = q
        mujoco.mj_kinematics(self.model, self.sensor_data)
        positions = (self.sensor_data.site_xpos[c.grip_ids] - c.center) @ c.axis_frame
        rotations = np.array([c.axis_frame.T @ self.sensor_data.site_xmat[s].reshape(3, 3) for s in c.grip_ids])
        wrists = np.c_[positions / .2, rotations[:, :, :2].reshape(2, 6)].ravel()
        geometry = np.r_[c.center / .8, c.axis_frame[:, :2].ravel(), self.rim_radius / .14, .014 / .014]
        frame = np.r_[q, self.sensor_velocity / 5., (c.target - q) / .15,
                      self.previous_action, wrists, angle, self.sensor_wheel_velocity,
                      (self._preview() - angle) / np.deg2rad(10),
                      self.target_velocity / np.deg2rad(20), geometry]
        if frame.shape != (ACTOR_DIM,) or not np.isfinite(frame).all():
            raise ValueError("Invalid centralized actor frame")
        return frame.astype(np.float32)

    def _observation(self):
        return {"actor": np.array(self.history[-1], dtype=np.float32), "privileged": self._privileged()}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.command_angles, self.command_velocities = self._make_commands()
        self._update_target()
        self.geometric_origins = self.data.site_xpos[self.controller.grip_ids].copy()
        self.geometric_rotations = self.data.site_xmat[self.controller.grip_ids].reshape(2, 3, 3).copy()
        frame = self._sensor_frame(initialize=True)
        self.history.clear()
        self.history.append(frame.copy())
        self.actor_queue.clear()
        for _ in range(3):
            self.actor_queue.append(frame.copy())
        self.reward_terms = {}
        info = self._info(self.interval_peak, self.interval_other)
        info["preparation_closed"] = self.preparation_closed.tolist()
        info["preparation_contact_forces"] = self.contact_forces.tolist()
        info["preparation_settle_seconds"] = self.config.settle_seconds
        return self._observation(), info

    # Bypass ContactResidualEnv's set_goal call and its angle-feedback motion.
    _advance_action = DexSteeringEnv._advance_action

    def _info(self, peak, other):
        info = DexSteeringEnv._info(self, peak, other)
        info.update(workspace_clips=self.controller.workspace_clips,
                    collision_constraints=self.controller.collision_constraints,
                    collision_qp_failures=self.controller.collision_qp_failures,
                    **{f"reward_{key}": value for key, value in self.reward_terms.items()})
        return info

    def _reward(self, previous):
        c = self.controller
        error = self.target_angle - self.wheel_angle
        rotation = c.axis_frame @ Rotation.from_rotvec([self.target_angle - self.initial_wheel, 0, 0]).as_matrix() @ c.axis_frame.T
        expected_positions = c.center + (self.geometric_origins - c.center) @ rotation.T
        expected_rotations = rotation @ self.geometric_rotations
        position_error = np.linalg.norm(self.data.site_xpos[c.grip_ids] - expected_positions, axis=1)
        current = self.data.site_xmat[c.grip_ids].reshape(2, 3, 3)
        orientation_error = Rotation.from_matrix(expected_rotations @ current.transpose(0, 2, 1)).magnitude()
        # Stage-one geometric reward only: these expected poses never enter IK.
        self.reward_terms = {
            "angle": float(2 * np.exp(-(error / np.deg2rad(3)) ** 2) + np.exp(-(error / np.deg2rad(10)) ** 2)),
            "geometry": float(.5 * np.mean(np.exp(-(position_error / .03) ** 2)) + .25 * np.mean(np.exp(-(orientation_error / .25) ** 2))),
            "grasp": float(.15 * self.enclosure.mean() + .1 * (self.contact_forces > .2).mean()),
            "motion": float(-.01 * np.mean((self.previous_action - previous) ** 2) - .005 * np.mean(self.previous_action ** 2)),
            "load": float(-.005 * max(0., self.interval_peak - 80) - .02 * (self.interval_other + self.interval_robot_peak)),
        }
        return sum(self.reward_terms.values())

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Centralized action needs 24 finite values")
        action = np.clip(action, -1, 1)
        previous = self.previous_action.copy()
        self.action_queue.append(action.copy())
        self._advance_action(list(self.action_queue)[-1 - self.action_delay])
        self.time += self.dt
        self._update_target()
        self._grasp_metrics()
        contact = self.contact_forces > .2
        self.lost_contact_seconds = 0. if np.any(contact) else self.lost_contact_seconds + self.dt
        stable = (abs(self.target_angle - self.wheel_angle) < np.deg2rad(2)
                  and abs(self.data.qvel[self.wheel_v]) < np.deg2rad(2)
                  and abs(self.target_velocity) < np.deg2rad(.1) and np.all(contact))
        self.hold_seconds = self.hold_seconds + self.dt if stable else 0.
        self.max_hold_seconds = max(self.max_hold_seconds, self.hold_seconds)
        self.previous_action = action.copy()
        reward = self._reward(previous)
        frame = self._sensor_frame()
        self.actor_queue.append(frame.copy())
        observed = list(self.actor_queue)[-1 - self.observation_delay].copy()
        # Command preview is current even if the physical sensors are delayed.
        observed[122:128] = (self._preview() - observed[120]) / np.deg2rad(10)
        observed[128] = self.target_velocity / np.deg2rad(20)
        self.history.append(observed)
        self._latest_frame = frame.copy()
        info = self._info(self.interval_peak, self.interval_other)
        reason = ""
        if not info["finite"]:
            reason = "nonfinite_state"
        elif self.interval_peak > 300 or self.interval_other > 150 or self.interval_robot_peak > 150:
            reason = "simulated_load_limit"
        elif abs(self.wheel_angle) > np.deg2rad(35):
            reason = "stage1_wheel_workspace"
        elif self.lost_contact_seconds > 1. and not self.config.no_contact:
            reason = "lost_grasp"
        elif self.controller.qp_failures > 5:
            reason = "qp_failure"
        info["termination_reason"] = reason
        if reason:
            reward -= 10.
        return self._observation(), float(reward), bool(reason), bool(self.time + 1e-9 >= self.config.episode_seconds), info
