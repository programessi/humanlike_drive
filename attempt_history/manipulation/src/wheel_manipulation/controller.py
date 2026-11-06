"""Wheel feedback provides common motion; PPO provides bounded pose corrections.

No wheel actuator, direct wheel-state writes, or independent learned velocity
integrators exist. Only the deterministic common reference is integrated.
"""
from dataclasses import dataclass

import numpy as np
import mujoco

from .common.controller import ControlConfig, DifferentialController


@dataclass
class ResidualControlConfig(ControlConfig):
    max_phase_difference_deg: float = 1.0
    wheel_position_gain: float = 2.0
    wheel_velocity_gain: float = .15
    reference_acceleration_deg_s2: float = 80.0
    common_residual_deg: float = 1.0
    differential_residual_deg: float = .5
    radial_residual_m: float = .002
    axial_residual_m: float = .002
    finger_residual_rad: float = .15
    residual_filter_seconds: float = .10
    residual_angular_rate_deg_s: float = 5.0


class ContactResidualController(DifferentialController):
    def __init__(self, model, config=None):
        super().__init__(model, config or ResidualControlConfig())
        c = self.config
        values = [c.wheel_position_gain, c.reference_acceleration_deg_s2,
                  c.common_residual_deg, c.differential_residual_deg,
                  c.radial_residual_m, c.axial_residual_m, c.finger_residual_rad,
                  c.residual_filter_seconds, c.residual_angular_rate_deg_s,
                  c.angular_speed_deg, c.phase_lead_deg]
        if not np.isfinite(values).all() or min(values) <= 0 or c.wheel_velocity_gain < 0:
            raise ValueError("Invalid contact residual or wheel-reference bounds")
        if 2 * c.differential_residual_deg > c.max_phase_difference_deg + 1e-9:
            raise ValueError("Differential residual exceeds the fixed-grasp phase band")
        if not np.isfinite(c.wheel_velocity_gain) or c.radial_residual_m > c.offset_limit or c.axial_residual_m > c.offset_limit:
            raise ValueError("Invalid velocity gain or linear residual workspace")
        self.wheel_q = int(model.joint("wheel_joint").qposadr[0])
        self.wheel_v = int(model.joint("wheel_joint").dofadr[0])
        self.goal_angle = self.goal_velocity = 0.0
        self.nominal_phase = self.nominal_velocity = 0.0
        self.phase_residual = np.zeros(2)  # common and differential, in radians
        self.phase_reference_velocity = np.zeros(2)
        self.reference_clips = 0
        self.encoder_angle = self.encoder_velocity = 0.0
        if not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE):
            raise ValueError("v2 requires its force-limited implicit-PD scene")

    def torque(self, data, target, gravity_compensation=True):
        """Return actuator position controls, preserving the legacy call site.

        The affine actuator computes Kp*(control-q)-Kd*qvel and clips the TOTAL
        motor force. Gravity/bias feedforward enters as a position-control offset;
        no unlimited qfrc_applied or passive motor damping is added.
        """
        command = np.asarray(target, dtype=float).copy()
        if gravity_compensation:
            command += data.qfrc_bias[self.motor_v] / self.kp
        return command

    def neutral_action(self):
        return np.zeros(18)

    def reset_reference(self, data):
        super().reset_reference(data)
        self.goal_angle = self.goal_velocity = 0.0
        self.nominal_phase = self.nominal_velocity = 0.0
        self.phase_residual[:] = 0
        self.phase_reference_velocity[:] = 0
        self.reference_clips = 0
        self.encoder_angle = float(data.qpos[self.wheel_q])
        self.encoder_velocity = 0.0

    def set_goal(self, angle, velocity):
        if not np.isfinite([angle, velocity]).all():
            raise ValueError("Wheel goal must be finite")
        self.goal_angle, self.goal_velocity = float(angle), float(velocity)

    def _advance_common_reference(self, data, dt):
        c = self.config
        # Encoder velocity is estimated at robot update rate, rather than using
        # simulator-only instantaneous wheel qvel in the reference controller.
        measured_angle = float(data.qpos[self.wheel_q])
        alpha = 1 - np.exp(-dt / .04)
        self.encoder_velocity += alpha * ((measured_angle - self.encoder_angle) / dt - self.encoder_velocity)
        self.encoder_angle = measured_angle
        desired = (self.goal_velocity + c.wheel_position_gain * (self.goal_angle - measured_angle)
                   - c.wheel_velocity_gain * (self.encoder_velocity - self.goal_velocity))
        vmax, acceleration = np.deg2rad([c.angular_speed_deg, c.reference_acceleration_deg_s2])
        limit = np.deg2rad(c.phase_limit_deg - c.common_residual_deg - c.differential_residual_deg)
        if limit <= 0:
            raise ValueError("Residuals consume the entire phase workspace")
        # Discrete braking bounds keep the common reference continuous at its
        # workspace limit. Measurement-lead conflicts are explicitly reported.
        lower_distance, upper_distance = max(0., self.nominal_phase + limit), max(0., limit - self.nominal_phase)
        lo = max(-vmax, self.nominal_velocity - acceleration * dt,
                 acceleration * dt - np.sqrt((acceleration * dt) ** 2 + 2 * acceleration * lower_distance))
        hi = min(vmax, self.nominal_velocity + acceleration * dt,
                 np.sqrt((acceleration * dt) ** 2 + 2 * acceleration * upper_distance) - acceleration * dt)
        offsets = np.array([self.phase_residual.sum(), self.phase_residual[0] - self.phase_residual[1]])
        measured_phase = self.measured_phase(data)
        lead = np.deg2rad(c.phase_lead_deg)
        lead_lo = float(np.max(measured_phase - lead - offsets))
        lead_hi = float(np.min(measured_phase + lead - offsets))
        lo = max(lo, (lead_lo - self.nominal_phase) / dt)
        hi = min(hi, (lead_hi - self.nominal_phase) / dt)
        if lo > hi + 1e-10:
            self.phase_constraint_failures += 1
            return
        self.nominal_velocity = float(np.clip(desired, lo, max(lo, hi)))
        self.reference_clips += int(abs(self.nominal_velocity - desired) > 1e-6)
        self.nominal_phase += dt * self.nominal_velocity

    def update(self, data, action, dt):
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError("v2 action must have 18 finite values")
        c = self.config
        alpha = 1 - np.exp(-dt / c.residual_filter_seconds)
        self.filtered_action += alpha * (np.clip(action, -1, 1) - self.filtered_action)
        a = self.filtered_action
        desired_phase_residual = np.deg2rad([c.common_residual_deg, c.differential_residual_deg]) * a[12:14]
        self.phase_residual += np.clip(desired_phase_residual - self.phase_residual,
                                       -np.deg2rad(c.residual_angular_rate_deg_s) * dt,
                                       np.deg2rad(c.residual_angular_rate_deg_s) * dt)
        old_phase = self.phase.copy()
        self._advance_common_reference(data, dt)
        self.phase = self.nominal_phase + np.array([self.phase_residual.sum(),
                                                   self.phase_residual[0] - self.phase_residual[1]])
        self.phase_reference_velocity = (self.phase - old_phase) / dt
        wrist = a[14:].reshape(2, 2)
        desired_radial = c.radial_residual_m * wrist[:, 0]
        desired_axial = c.axial_residual_m * wrist[:, 1]
        self.radial += np.clip(desired_radial - self.radial, -c.linear_speed * dt, c.linear_speed * dt)
        self.axial += np.clip(desired_axial - self.axial, -c.linear_speed * dt, c.linear_speed * dt)
        self.solve_velocity(*self.poses(), data, dt)
        desired_hand = np.clip(self.closed_hand + c.finger_residual_rad * a[:12],
                               self.hand_limits[:, 0] + c.joint_margin,
                               self.hand_limits[:, 1] - c.joint_margin)
        inds = slice(14, 26)
        lo, hi = self._bounds(inds, c.hand_speed, c.hand_acceleration,
                              self.hand_limits, data.qpos[self.hand_q], dt)
        self.velocity[inds] = np.clip((desired_hand - self.target[inds]) / dt, lo, hi)
        self.target[inds] += dt * self.velocity[inds]
        return self.target.copy()
