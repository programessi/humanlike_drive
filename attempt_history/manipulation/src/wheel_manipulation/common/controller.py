"""Bound-constrained differential IK, with finite-impedance motor execution.

The least-squares problem is a convex QP with box constraints on joint velocity.
It controls a kinematic reference only. It never writes the physical state.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation

from .hardware import RobotHardware


@dataclass
class ControlConfig:
    robot_hz: int = 50
    angular_speed_deg: float = 20.0
    linear_speed: float = .01
    offset_limit: float = .005
    phase_limit_deg: float = 40.0
    phase_lead_deg: float = 8.0
    arm_speed: float = 1.0
    arm_acceleration: float = 4.0
    hand_speed: float = 1.5
    hand_acceleration: float = 8.0
    joint_lead: float = .15
    joint_margin: float = .005
    position_gain: float = 12.0
    orientation_weight: float = .12
    damping: float = .003
    posture_weight: float = .001
    initial_palm_pitch_deg: float = 0.0
    max_phase_difference_deg: float = 2.0


class DifferentialController(RobotHardware):
    def __init__(self, model, config=None):
        super().__init__(model)
        self.config = config or ControlConfig()
        if self.config.robot_hz <= 0:
            raise ValueError("robot_hz must be positive")
        if min(self.config.arm_speed, self.config.arm_acceleration,
               self.config.hand_speed, self.config.hand_acceleration,
               self.config.offset_limit, self.config.joint_lead) <= 0:
            raise ValueError("control bounds must be positive")
        if not 0 <= self.config.max_phase_difference_deg <= 80:
            raise ValueError("Relative wrist phase bound must be within [0, 80] degrees")
        self.hand_limits[[1, 7], 0] = 0.0  # Coupled thumb joints cannot extend below zero.
        self.initial_arm, self.initial_ik_error = self._initial_grasp_ik()
        if self.initial_ik_error > .002:
            raise ValueError(f"Initial grasp IK is infeasible: {self.initial_ik_error * 1000:.2f} mm")
        self.seed = self.initial_arm.copy()
        self.command_data = mujoco.MjData(model)
        self.target = np.r_[self.initial_arm, self.closed_hand]
        self.velocity = np.zeros(26)
        self.phase = np.zeros(2)
        self.radial = np.zeros(2)
        self.axial = np.zeros(2)
        self.filtered_action = self.neutral_action()
        self.base_local = np.array([[-.002, .163, 0], [-.002, -.163, 0]])
        self.origin_phase = np.array([0., np.pi])
        self.orientation_offsets = np.tile(np.eye(3), (2, 1, 1))
        self.qp_error = np.zeros(2)
        self.qp_failures = 0
        self.safety_overrides = 0
        self.phase_constraint_failures = 0
        self.nominal_kp, self.nominal_kd = self.kp.copy(), self.kd.copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def _initial_grasp_ik(self):
        """One static pregrasp solve; no wheel-angle lookup or motion reference."""
        rotation = self.axis_frame @ Rotation.from_euler("y", self.config.initial_palm_pitch_deg, degrees=True).as_matrix()
        result, errors = [], []
        for side, site in enumerate(self.grip_ids):
            inds = slice(side * 7, (side + 1) * 7)
            q0 = np.array([-.65, .3, .1, 1.15, 0, 0, 0])
            if side == 1:
                q0[[1, 2, 4, 6]] *= -1
            position = self.center + self.axis_frame @ np.array([-.002, .163 if side == 0 else -.163, 0])
            def objective(q):
                self.ik_data.qpos[self.arm_q[inds]] = q
                mujoco.mj_kinematics(self.model, self.ik_data)
                current = self.ik_data.site_xmat[site].reshape(3, 3)
                return np.r_[self.ik_data.site_xpos[site] - position,
                             .12 * Rotation.from_matrix(rotation @ current.T).as_rotvec(), .0002 * (q - q0)]
            lower, upper = self.limits[inds, 0] + .03, self.limits[inds, 1] - .03
            solution = least_squares(objective, np.clip(q0, lower, upper), bounds=(lower, upper),
                                     max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            result.extend(solution.x)
            errors.append(np.linalg.norm(objective(solution.x)[:3]))
        return np.array(result), float(max(errors))

    def neutral_action(self):
        return np.zeros(18)

    def reset_reference(self, data):
        limits = np.vstack([self.limits, self.hand_limits])
        self.target = np.clip(data.qpos[self.motor_q], limits[:, 0] + self.config.joint_margin,
                              limits[:, 1] - self.config.joint_margin)
        self.command_data.qpos[:] = data.qpos
        self.command_data.qpos[self.motor_q] = self.target
        mujoco.mj_forward(self.model, self.command_data)
        self.velocity[:] = 0
        self.phase[:] = self.radial[:] = self.axial[:] = 0
        self.filtered_action = self.neutral_action()
        self.qp_failures = self.safety_overrides = 0
        self.phase_constraint_failures = 0
        self.qp_error[:] = 0
        for s, site in enumerate(self.grip_ids):
            local = self.axis_frame.T @ (self.command_data.site_xpos[site] - self.center)
            self.base_local[s] = local
            self.origin_phase[s] = np.arctan2(local[2], local[1])
            self.orientation_offsets[s] = self.axis_frame.T @ self.command_data.site_xmat[site].reshape(3, 3)

    def measured_phase(self, data):
        local = (data.site_xpos[self.grip_ids] - self.center) @ self.axis_frame
        phases = np.arctan2(local[:, 2], local[:, 1]) - self.origin_phase
        return np.arctan2(np.sin(phases), np.cos(phases))

    def poses(self):
        positions, rotations = [], []
        for s in range(2):
            rotation = Rotation.from_rotvec([self.phase[s], 0, 0]).as_matrix()
            local = self.base_local[s].copy()
            radius = np.linalg.norm(local[1:])
            local[1:] *= (radius + self.radial[s]) / radius
            local[0] += self.axial[s]
            positions.append(self.center + self.axis_frame @ rotation @ local)
            rotations.append(self.axis_frame @ rotation @ self.orientation_offsets[s])
        return np.array(positions), np.array(rotations)

    def _bounds(self, previous, speed, acceleration, limits, measured, dt):
        margin = self.config.joint_margin
        lo = np.maximum(-speed, self.velocity[previous] - acceleration * dt)
        hi = np.minimum(speed, self.velocity[previous] + acceleration * dt)
        # Reserve braking distance before a joint limit. A simple one-step
        # position bound can demand instantaneous deceleration at the boundary.
        lower_distance = np.maximum(0., self.target[previous] - limits[:, 0] - margin)
        upper_distance = np.maximum(0., limits[:, 1] - margin - self.target[previous])
        braking_lo = acceleration * dt - np.sqrt((acceleration * dt) ** 2 + 2 * acceleration * lower_distance)
        braking_hi = np.sqrt((acceleration * dt) ** 2 + 2 * acceleration * upper_distance) - acceleration * dt
        lo = np.maximum(lo, braking_lo)
        hi = np.minimum(hi, braking_hi)
        lo = np.maximum(lo, (limits[:, 0] + margin - self.target[previous]) / dt)
        hi = np.minimum(hi, (limits[:, 1] - margin - self.target[previous]) / dt)
        # Bound commanded displacement from measurements; abrupt disturbances
        # may require a reported safety override of the acceleration constraint.
        lead = self.config.joint_lead
        lead_lo = (measured - lead - self.target[previous]) / dt
        lead_hi = (measured + lead - self.target[previous]) / dt
        bad = np.maximum(lo, lead_lo) >= np.minimum(hi, lead_hi)
        self.safety_overrides += int(np.any(bad))
        lo = np.where(bad, lo, np.maximum(lo, lead_lo))
        hi = np.where(bad, hi, np.minimum(hi, lead_hi))
        # At a limit the interval can collapse. BVLS requires strict bounds.
        return lo, np.maximum(hi, lo + 1e-10)

    def solve_velocity(self, positions, rotations, measured, dt):
        self.command_data.qpos[:] = measured.qpos
        self.command_data.qpos[self.arm_q] = self.target[:14]
        mujoco.mj_forward(self.model, self.command_data)
        c = self.config
        for s, site in enumerate(self.grip_ids):
            inds = slice(s * 7, (s + 1) * 7)
            dofs = self.arm_v[inds]
            mujoco.mj_jacSite(self.model, self.command_data, self.jacp, self.jacr, site)
            position_error = positions[s] - self.command_data.site_xpos[site]
            current = self.command_data.site_xmat[site].reshape(3, 3)
            orientation_error = Rotation.from_matrix(rotations[s] @ current.T).as_rotvec()
            a = np.vstack([self.jacp[:, dofs], c.orientation_weight * self.jacr[:, dofs],
                           c.damping * np.eye(7), c.posture_weight * np.eye(7)])
            b = np.r_[c.position_gain * position_error,
                      c.orientation_weight * c.position_gain * orientation_error,
                      np.zeros(7), c.posture_weight * (self.seed[inds] - self.target[inds])]
            lo, hi = self._bounds(inds, c.arm_speed, c.arm_acceleration,
                                  self.limits[inds], measured.qpos[self.arm_q[inds]], dt)
            solution = lsq_linear(a, b, bounds=(lo, hi), method="bvls", tol=1e-8, max_iter=30)
            if not solution.success or not np.isfinite(solution.x).all():
                self.qp_failures += 1
                velocity = np.clip(np.zeros(7), lo, hi)
            else:
                velocity = solution.x
            self.velocity[inds] = velocity
            self.target[inds] += dt * velocity
            self.qp_error[s] = np.linalg.norm(position_error)

    def update(self, data, action, dt):
        raise NotImplementedError("Use the contact-residual task controller")
