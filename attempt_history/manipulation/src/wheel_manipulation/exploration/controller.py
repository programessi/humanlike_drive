"""Policy-driven independent wrist targets, joint IK and force-limited PD.

There is deliberately no wheel-error controller or moving circle reference here.
Wheel geometry is used for coordinates, not to supply the task motion.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import lsq_linear, minimize, LinearConstraint
from scipy.spatial.transform import Rotation

from ..common.controller import ControlConfig, DifferentialController
from ..controller import ContactResidualController


@dataclass
class WristControlConfig(ControlConfig):
    wrist_linear_speed: float = .04
    wrist_angular_speed_deg: float = 25.
    wrist_workspace_m: float = .06
    wrist_orientation_workspace_deg: float = 40.
    wrist_position_lead_m: float = .012
    wrist_orientation_lead_deg: float = 10.
    action_filter_seconds: float = .10
    collision_margin_m: float = .012
    collision_activation_m: float = .035


class IndependentWristController(DifferentialController):
    # Identical affine motor law to the preserved task, without its wheel servo.
    torque = ContactResidualController.torque

    def __init__(self, model, config=None):
        super().__init__(model, config or WristControlConfig())
        if not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE):
            raise ValueError("Exploration needs force-limited affine PD actuators")
        self.nominal_closed = self.closed_hand.copy()
        self.wrist_positions = np.zeros((2, 3))
        self.wrist_rotations = np.tile(np.eye(3), (2, 1, 1))
        self.initial_positions = self.wrist_positions.copy()
        self.initial_rotations = self.wrist_rotations.copy()
        self.workspace_clips = self.collision_constraints = self.collision_qp_failures = 0
        # Conservative geometric constraints for elbow/palm versus opposite
        # elbow/palm and torso. Finger contacts remain in physical diagnostics;
        # these sparse guards do not constitute full-body collision guarantees.
        elbows = [model.geom(f"{s}_elbow_link_guard").id for s in ("left", "right")]
        palms = [int(np.flatnonzero((model.geom_bodyid == model.body(f"{s}_wrist_yaw_link").id)
                                    & (model.geom_contype != 0))[0]) for s in ("left", "right")]
        torso = model.geom("dex_torso_guard").id
        self.guard_pairs = [(a, b) for a in (elbows[0], palms[0]) for b in (elbows[1], palms[1])]
        self.guard_pairs += [(g, torso) for g in elbows + palms]
        self._j1 = np.zeros((3, model.nv))
        self._j2 = np.zeros((3, model.nv))

    def neutral_action(self):
        return np.zeros(24)

    def reset_reference(self, data):
        super().reset_reference(data)
        self.wrist_positions = data.site_xpos[self.grip_ids].copy()
        self.wrist_rotations = data.site_xmat[self.grip_ids].reshape(2, 3, 3).copy()
        self.initial_positions = self.wrist_positions.copy()
        self.initial_rotations = self.wrist_rotations.copy()
        self.workspace_clips = self.collision_constraints = self.collision_qp_failures = 0

    def poses(self):
        return self.wrist_positions.copy(), self.wrist_rotations.copy()

    def hand_targets(self, action):
        """Absolute, full-range drive targets; zero retains nominal closure."""
        lo = self.hand_limits[:, 0] + self.config.joint_margin
        hi = self.hand_limits[:, 1] - self.config.joint_margin
        center = np.clip(self.nominal_closed, lo, hi)
        return np.where(action < 0, center + action * (center - lo), center + action * (hi - center))

    @staticmethod
    def _limit_vector(vector, limit):
        return vector * min(1., limit / max(np.linalg.norm(vector), 1e-12))

    def _guard_constraints(self):
        rows, bounds = [], []
        for g1, g2 in self.guard_pairs:
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(self.model, self.command_data, g1, g2,
                                             self.config.collision_activation_m, segment)
            if distance >= self.config.collision_activation_m or distance < 0:
                continue
            normal = segment[3:] - segment[:3]
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            mujoco.mj_jac(self.model, self.command_data, self._j1, None, segment[:3],
                          int(self.model.geom_bodyid[g1]))
            mujoco.mj_jac(self.model, self.command_data, self._j2, None, segment[3:],
                          int(self.model.geom_bodyid[g2]))
            rows.append(normal @ (self._j2[:, self.arm_v] - self._j1[:, self.arm_v]))
            bounds.append(4. * (self.config.collision_margin_m - distance))
        return rows, bounds

    def solve_velocity(self, positions, rotations, measured, dt):
        """Joint 14-variable box QP, with active linearized guard constraints."""
        c = self.config
        self.command_data.qpos[:] = measured.qpos
        self.command_data.qpos[self.arm_q] = self.target[:14]
        mujoco.mj_forward(self.model, self.command_data)
        matrices, targets = [], []
        for side, site in enumerate(self.grip_ids):
            mujoco.mj_jacSite(self.model, self.command_data, self.jacp, self.jacr, site)
            error = positions[side] - self.command_data.site_xpos[site]
            current = self.command_data.site_xmat[site].reshape(3, 3)
            rot_error = Rotation.from_matrix(rotations[side] @ current.T).as_rotvec()
            matrices.extend([self.jacp[:, self.arm_v].copy(), c.orientation_weight * self.jacr[:, self.arm_v].copy()])
            targets.extend([c.position_gain * error, c.orientation_weight * c.position_gain * rot_error])
            self.qp_error[side] = np.linalg.norm(error)
        a = np.vstack(matrices + [c.damping * np.eye(14), c.posture_weight * np.eye(14)])
        b = np.concatenate(targets + [np.zeros(14), c.posture_weight * (self.seed - self.target[:14])])
        lo, hi = self._bounds(slice(0, 14), c.arm_speed, c.arm_acceleration,
                              self.limits, measured.qpos[self.arm_q], dt)
        solution = lsq_linear(a, b, bounds=(lo, hi), method="bvls", tol=1e-8, max_iter=30)
        velocity = solution.x
        rows, lower = self._guard_constraints()
        ok = solution.success and np.isfinite(velocity).all()
        if rows:
            self.collision_constraints += len(rows)
            mat, lower = np.asarray(rows), np.asarray(lower)
            constrained = minimize(lambda x: .5 * np.sum((a @ x - b) ** 2), velocity,
                                   jac=lambda x: a.T @ (a @ x - b), method="SLSQP",
                                   bounds=list(zip(lo, hi)), constraints=[LinearConstraint(mat, lower, np.inf)],
                                   options={"maxiter": 30, "ftol": 1e-9})
            ok = bool(constrained.success and np.isfinite(constrained.x).all()
                      and np.all(mat @ constrained.x >= lower - 1e-6))
            velocity = constrained.x
            if not ok:
                self.collision_qp_failures += 1
        if not ok:
            self.qp_failures += 1
            velocity = np.clip(np.zeros(14), lo, hi)
        self.velocity[:14] = velocity
        self.target[:14] += dt * velocity

    def update(self, data, action, dt):
        action = np.asarray(action, dtype=float)
        if action.shape != (24,) or not np.isfinite(action).all():
            raise ValueError("Centralized action must contain 24 finite values")
        c = self.config
        alpha = 1 - np.exp(-dt / c.action_filter_seconds)
        self.filtered_action += alpha * (np.clip(action, -1, 1) - self.filtered_action)
        wrists = self.filtered_action[:12].reshape(2, 6)
        for side, site in enumerate(self.grip_ids):
            position = self.wrist_positions[side] + self.axis_frame @ (wrists[side, :3] * c.wrist_linear_speed * dt)
            delta = position - self.initial_positions[side]
            local = np.clip(self.axis_frame.T @ delta, -c.wrist_workspace_m, c.wrist_workspace_m)
            bounded = self.initial_positions[side] + self.axis_frame @ local
            self.workspace_clips += int(np.linalg.norm(bounded - position) > 1e-9)
            self.wrist_positions[side] = data.site_xpos[site] + self._limit_vector(
                bounded - data.site_xpos[site], c.wrist_position_lead_m)
            step = self.axis_frame @ (wrists[side, 3:] * np.deg2rad(c.wrist_angular_speed_deg) * dt)
            rotation = Rotation.from_rotvec(step).as_matrix() @ self.wrist_rotations[side]
            initial_error = Rotation.from_matrix(rotation @ self.initial_rotations[side].T).as_rotvec()
            rotation = Rotation.from_rotvec(self._limit_vector(initial_error, np.deg2rad(c.wrist_orientation_workspace_deg))).as_matrix() @ self.initial_rotations[side]
            measured = data.site_xmat[site].reshape(3, 3)
            lead = Rotation.from_matrix(rotation @ measured.T).as_rotvec()
            self.wrist_rotations[side] = Rotation.from_rotvec(self._limit_vector(lead, np.deg2rad(c.wrist_orientation_lead_deg))).as_matrix() @ measured
        self.solve_velocity(*self.poses(), data, dt)
        desired_hand = self.hand_targets(self.filtered_action[12:])
        lo, hi = self._bounds(slice(14, 26), c.hand_speed, c.hand_acceleration,
                              self.hand_limits, data.qpos[self.hand_q], dt)
        self.velocity[14:] = np.clip((desired_hand - self.target[14:]) / dt, lo, hi)
        self.target[14:] += dt * self.velocity[14:]
        local = (self.wrist_positions - self.center) @ self.axis_frame
        raw_phase = np.arctan2(local[:, 2], local[:, 1]) - self.origin_phase
        self.phase = np.arctan2(np.sin(raw_phase), np.cos(raw_phase))
