"""Cylindrical wrist motion coordinates; no reference-angle feedback law."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize, LinearConstraint
from scipy.spatial.transform import Rotation

from ..exploration.controller import IndependentWristController, WristControlConfig


@dataclass
class RegraspControlConfig(WristControlConfig):
    posture_weight: float = .01
    wrist_linear_speed: float = .05
    wrist_angular_speed_deg: float = 40.
    wrist_workspace_m: float = .36
    wrist_orientation_workspace_deg: float = 170.
    radial_min_m: float = .085
    radial_max_m: float = .28
    axial_min_m: float = -.14
    axial_max_m: float = .10
    guard_max_iterations: int = 150


class RegraspController(IndependentWristController):
    def __init__(self, model, config=None):
        super().__init__(model, config or RegraspControlConfig())
        self.nominal_initial_arm = self.initial_arm.copy()
        self.preparation_cache = {}

    def preparation_arm(self, phase_deg=0., free_side=-1, free_gap_m=.12):
        """Static initial-state IK only. Never called during policy execution."""
        key = (float(phase_deg), int(free_side), float(free_gap_m))
        if key not in self.preparation_cache:
            result = self.nominal_initial_arm.copy()
            for side, site in enumerate(self.grip_ids):
                inds = slice(7 * side, 7 * (side + 1))
                phase = 0. if side == free_side else np.deg2rad(phase_deg)
                rot = Rotation.from_rotvec([phase, 0, 0]).as_matrix()
                local = rot @ np.array([-.002, .163 * (1 if side == 0 else -1), 0])
                if side == free_side:
                    local[0] -= free_gap_m
                pos, orientation = self.center + self.axis_frame @ local, self.axis_frame @ rot
                q0 = result[inds].copy()
                def fun(q):
                    self.ik_data.qpos[self.arm_q[inds]] = q
                    mujoco.mj_kinematics(self.model, self.ik_data)
                    actual = self.ik_data.site_xmat[site].reshape(3, 3)
                    return np.r_[self.ik_data.site_xpos[site] - pos,
                                 .12 * Rotation.from_matrix(orientation @ actual.T).as_rotvec(), .0002 * (q - q0)]
                lo, hi = self.limits[inds, 0] + .03, self.limits[inds, 1] - .03
                sol = least_squares(fun, np.clip(q0, lo, hi), bounds=(lo, hi), max_nfev=200,
                                    ftol=1e-8, xtol=1e-8, gtol=1e-8)
                if np.linalg.norm(fun(sol.x)[:3]) > .003:
                    raise ValueError(f"Preparation IK infeasible: {key}, side={side}")
                result[inds] = sol.x
            self.preparation_cache[key] = result
        return self.preparation_cache[key].copy()

    def update(self, data, action, dt):
        action = np.asarray(action, dtype=float)
        if action.shape != (24,) or not np.isfinite(action).all():
            raise ValueError("Expected 24 finite wrist/hand commands")
        c = self.config
        self.filtered_action += (1 - np.exp(-dt / c.action_filter_seconds)) * (np.clip(action, -1, 1) - self.filtered_action)
        wrists = self.filtered_action[:12].reshape(2, 6)
        for side, site in enumerate(self.grip_ids):
            local = self.axis_frame.T @ (self.wrist_positions[side] - self.center)
            rho = max(np.linalg.norm(local[1:]), c.radial_min_m)
            phi = np.arctan2(local[2], local[1])
            vt, vr, vz = wrists[side, :3] * c.wrist_linear_speed
            phi += dt * vt / rho
            new_rho = np.clip(rho + dt * vr, c.radial_min_m, c.radial_max_m)
            z = np.clip(local[0] + dt * vz, c.axial_min_m, c.axial_max_m)
            self.workspace_clips += int(new_rho != rho + dt * vr or z != local[0] + dt * vz)
            desired = self.center + self.axis_frame @ np.r_[z, new_rho * np.cos(phi), new_rho * np.sin(phi)]
            self.wrist_positions[side] = data.site_xpos[site] + self._limit_vector(desired - data.site_xpos[site], c.wrist_position_lead_m)
            step = self.axis_frame @ (wrists[side, 3:] * np.deg2rad(c.wrist_angular_speed_deg) * dt)
            rotation = Rotation.from_rotvec(step).as_matrix() @ self.wrist_rotations[side]
            actual = data.site_xmat[site].reshape(3, 3)
            error = Rotation.from_matrix(rotation @ actual.T).as_rotvec()
            self.wrist_rotations[side] = Rotation.from_rotvec(self._limit_vector(error, np.deg2rad(c.wrist_orientation_lead_deg))).as_matrix() @ actual
        self.solve_velocity(*self.poses(), data, dt)
        desired_hand = self.hand_targets(self.filtered_action[12:])
        lo, hi = self._bounds(slice(14, 26), c.hand_speed, c.hand_acceleration,
                              self.hand_limits, data.qpos[self.hand_q], dt)
        self.velocity[14:] = np.clip((desired_hand - self.target[14:]) / dt, lo, hi)
        self.target[14:] += dt * self.velocity[14:]
        local = (self.wrist_positions - self.center) @ self.axis_frame
        raw = np.arctan2(local[:, 2], local[:, 1]) - self.origin_phase
        self.phase = np.arctan2(np.sin(raw), np.cos(raw))

    def solve_velocity(self, positions, rotations, measured, dt):
        """Same joint QP as exploration, with audited guard-solver budget.

        Preserved controllers are unchanged. In the release/regrasp route the
        old 30-iteration SLSQP budget produced status 9 despite a feasible
        problem; 150 iterations completed both directions. Bounds and guard
        margins are not relaxed to obtain a successful solver status.
        """
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
        lo, hi = self._bounds(slice(0, 14), c.arm_speed, c.arm_acceleration, self.limits, measured.qpos[self.arm_q], dt)
        solution = lsq_linear(a, b, bounds=(lo, hi), method="bvls", tol=1e-8, max_iter=30)
        velocity = solution.x
        rows, lower = self._guard_constraints()
        ok = bool(solution.success and np.isfinite(velocity).all())
        if rows:
            self.collision_constraints += len(rows)
            mat, lower = np.asarray(rows), np.asarray(lower)
            constrained = minimize(lambda x: .5 * np.sum((a @ x - b) ** 2), velocity,
                                   jac=lambda x: a.T @ (a @ x - b), method="SLSQP",
                                   bounds=list(zip(lo, hi)), constraints=[LinearConstraint(mat, lower, np.inf)],
                                   options={"maxiter": c.guard_max_iterations, "ftol": 1e-9})
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
