"""Kinematic feasibility guard, independent of the physical wrist executor.

The shadow H1 never steps dynamics and its joints never drive simulated hands.
Local continuation IK certifies a target; FK of the accepted solution supplies
the wrist pose. Invalid requests are shortened or rejected, not teleported to
an unrelated IK branch. This is an online guard, not an offline workspace map.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..common.hardware import RobotHardware
from ..regrasp.scene import SCENE_PATH, build_scene


@dataclass
class ReachabilityConfig:
    enabled: bool = True
    joint_margin_deg: float = 5.
    joint_speed_rad_s: float = 1.
    collision_margin_m: float = .012
    minimum_singular_value: float = .005
    position_tolerance_m: float = .00025
    orientation_tolerance_deg: float = .15
    path_resolution_rad: float = .005
    max_evaluations: int = 35

    def validate(self):
        values = [self.joint_margin_deg, self.joint_speed_rad_s,
                  self.collision_margin_m, self.minimum_singular_value,
                  self.position_tolerance_m, self.orientation_tolerance_deg,
                  self.path_resolution_rad, self.max_evaluations]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Invalid reachability parameters')


class ArmReachabilityGuard:
    """Track one continuous paired-arm branch, using scratch MuJoCo data only.

    Collision checks use the source model's eight elbow/palm/torso guard pairs.
    Sampling joint-space segments is conservative engineering screening, not
    continuous collision detection or a full mesh/actuator safety certificate.
    """
    def __init__(self, config=None):
        self.config = config or ReachabilityConfig()
        self.config.validate()
        if not SCENE_PATH.exists():
            build_scene()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        hardware = RobotHardware(self.model)
        self.data = hardware.ik_data
        self.arm_q, self.arm_v = hardware.arm_q, hardware.arm_v
        self.sites, self.limits = hardware.grip_ids, hardware.limits
        self.center, self.frame = hardware.center, hardware.axis_frame
        self.seed = hardware.seed.copy()
        self.lower = self.limits[:, 0] + np.deg2rad(self.config.joint_margin_deg)
        self.upper = self.limits[:, 1] - np.deg2rad(self.config.joint_margin_deg)
        if np.any(self.lower >= self.upper):
            raise ValueError('Joint margin removes the arm workspace')
        elbows = [self.model.geom(f'{s}_elbow_link_guard').id for s in ('left', 'right')]
        palms = [int(np.flatnonzero((self.model.geom_bodyid == self.model.body(f'{s}_wrist_yaw_link').id)
                                   & (self.model.geom_contype != 0))[0]) for s in ('left', 'right')]
        torso = self.model.geom('dex_torso_guard').id
        self.guard_pairs = [(a, b) for a in (elbows[0], palms[0]) for b in (elbows[1], palms[1])]
        self.guard_pairs += [(g, torso) for g in elbows + palms]
        self.jacp, self.jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.q = None

    def _forward(self, q):
        self.data.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.model, self.data)
        return (self.data.site_xpos[self.sites].copy(),
                self.data.site_xmat[self.sites].reshape(2, 3, 3).copy())

    def _solve(self, positions, rotations, start, lower, upper, budget):
        """No wheel state/goal, hand force or old controller enters this solve."""
        result = start.copy()
        for side, site in enumerate(self.sites):
            inds = slice(7*side, 7*(side+1))
            self._forward(start)
            if (np.linalg.norm(self.data.site_xpos[site]-positions[side]) < 1e-10
                    and np.linalg.norm(Rotation.from_matrix(
                        self.data.site_xmat[site].reshape(3, 3) @ rotations[side].T).as_rotvec()) < 1e-10):
                continue
            def residual(q):
                self.data.qpos[self.arm_q[inds]] = q
                mujoco.mj_kinematics(self.model, self.data)
                actual = self.data.site_xmat[site].reshape(3, 3)
                return np.r_[self.data.site_xpos[site]-positions[side],
                             .12*Rotation.from_matrix(actual @ rotations[side].T).as_rotvec(),
                             .00001*(q-start[inds])]
            solution = least_squares(residual, np.clip(start[inds], lower[inds], upper[inds]),
                                     bounds=(lower[inds], upper[inds]), max_nfev=budget,
                                     ftol=1e-8, xtol=1e-8, gtol=1e-10)
            # Hitting the iteration budget is acceptable only if the explicit
            # pose, joint, singularity and path checks below all pass. Optimizer
            # termination alone is neither proof nor disproof of feasibility.
            if solution.status < 0 or not np.isfinite(solution.x).all():
                return None, 'solver'
            result[inds] = solution.x
        actual_p, actual_r = self._forward(result)
        pe = np.linalg.norm(actual_p-positions, axis=1).max()
        re = max(np.linalg.norm(Rotation.from_matrix(a @ b.T).as_rotvec())
                 for a, b in zip(actual_r, rotations))
        if pe > self.config.position_tolerance_m or re > np.deg2rad(self.config.orientation_tolerance_deg):
            return None, 'pose_or_step_bound'
        return result, ''

    def diagnostics(self, q):
        self._forward(q)
        # Jacobians require mj_comPos after mj_kinematics; mj_forward also
        # updates all collision geometry transforms on this scratch data.
        mujoco.mj_forward(self.model, self.data)
        singular = []
        for side, site in enumerate(self.sites):
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, site)
            dofs = self.arm_v[7*side:7*(side+1)]
            jac = np.vstack([self.jacp[:, dofs], .12*self.jacr[:, dofs]])
            singular.append(float(np.linalg.svd(jac, compute_uv=False)[-1]))
        clearance = min(mujoco.mj_geomDistance(self.model, self.data, a, b, 1., np.zeros(6))
                        for a, b in self.guard_pairs)
        margin = float(np.min(np.minimum(q-self.limits[:, 0], self.limits[:, 1]-q)))
        return dict(min_joint_margin_deg=float(np.rad2deg(margin)),
                    min_guard_clearance_m=float(clearance), min_singular_value=min(singular))

    def _valid(self, q):
        if not np.isfinite(q).all() or np.any(q < self.lower-1e-10) or np.any(q > self.upper+1e-10):
            return False, 'joint_limit', None
        metrics = self.diagnostics(q)
        if metrics['min_guard_clearance_m'] < self.config.collision_margin_m:
            return False, 'collision_guard', metrics
        if metrics['min_singular_value'] < self.config.minimum_singular_value:
            return False, 'singularity', metrics
        return True, '', metrics

    def check_segment(self, start, end):
        count = max(1, int(np.ceil(np.max(np.abs(end-start))/self.config.path_resolution_rad)))
        for fraction in np.linspace(0., 1., count+1):
            valid, reason, _ = self._valid(start + fraction*(end-start))
            if not valid:
                return False, reason
        return True, ''

    def reset(self, positions, rotations):
        q, reason = self._solve(positions, rotations, self.seed, self.lower, self.upper, 250)
        if q is None:
            raise ValueError(f'Initial wrist grasp is not H1-reachable: {reason}')
        valid, reason, self.metrics = self._valid(q)
        if not valid:
            raise ValueError(f'Initial paired-arm configuration is invalid: {reason}')
        self.q = q
        self.positions, self.rotations = self._forward(q)
        self.last_fraction, self.last_reason = 1., ''
        self.clipped_steps, self.rejected_steps = 0, 0
        self.actual_valid, self.actual_reason = True, ''
        return self.positions.copy(), self.rotations.copy()

    def filter(self, positions, rotations, dt):
        if self.q is None:
            raise RuntimeError('Reset reachability guard before filtering')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid reachability clock')
        old_p, old_r, old_q = self.positions.copy(), self.rotations.copy(), self.q.copy()
        # An unchanged full pose must never trigger nullspace/posture motion.
        increments = np.array([Rotation.from_matrix(b @ a.T).as_rotvec() for a, b in zip(old_r, rotations)])
        if np.max(np.abs(positions-old_p)) < 1e-10 and np.max(np.abs(increments)) < 1e-10:
            self.last_fraction, self.last_reason = 1., ''
            return old_p, old_r
        bound = self.config.joint_speed_rad_s*dt
        lower, upper = np.maximum(self.lower, old_q-bound), np.minimum(self.upper, old_q+bound)
        reason = ''
        for fraction in (1., .5, .25, .125, .0625):
            p = old_p + fraction*(positions-old_p)
            r = np.array([Rotation.from_rotvec(fraction*e).as_matrix() @ a for e, a in zip(increments, old_r)])
            q, failure = self._solve(p, r, old_q, lower, upper, self.config.max_evaluations)
            if q is not None:
                valid, failure = self.check_segment(old_q, q)
                if valid:
                    self.q = q
                    self.positions, self.rotations = self._forward(q)
                    self.metrics = self.diagnostics(q)
                    self.last_fraction, self.last_reason = fraction, reason
                    self.clipped_steps += int(fraction < 1.)
                    return self.positions.copy(), self.rotations.copy()
            if not reason:
                reason = failure
        # Roll back scratch state as well as the public certificate.
        self._forward(old_q)
        self.last_fraction, self.last_reason = 0., reason
        self.clipped_steps += 1
        self.rejected_steps += 1
        return old_p, old_r

    def observation(self):
        normalized_q = 2*(self.q-self.limits[:, 0])/(self.limits[:, 1]-self.limits[:, 0])-1
        return np.r_[normalized_q, self.last_fraction]

    def assess_actual(self, positions, rotations):
        """Reject episodes exploiting compliance beyond the kinematic envelope.

        Actual floating palms are not teleported back. This is a local branch
        check at policy ticks, not a physical arm constraint at every substep.
        """
        q, reason = self._solve(positions, rotations, self.q,
                                np.maximum(self.lower, self.q-.25),
                                np.minimum(self.upper, self.q+.25), self.config.max_evaluations)
        valid = q is not None
        if valid:
            valid, reason = self.check_segment(self.q, q)
        self.actual_valid, self.actual_reason = bool(valid), reason
        self._forward(self.q)
        return self.actual_valid

    def info(self):
        return dict(arm_reachability_enabled=True, arm_accepted_fraction=self.last_fraction,
                    arm_filter_reason=self.last_reason, arm_clipped_steps=self.clipped_steps,
                    arm_rejected_steps=self.rejected_steps,
                    arm_min_joint_margin_deg=self.metrics['min_joint_margin_deg'],
                    arm_min_guard_clearance_mm=1000*self.metrics['min_guard_clearance_m'],
                    arm_min_singular_value=self.metrics['min_singular_value'],
                    arm_actual_pose_valid=self.actual_valid, arm_actual_pose_reason=self.actual_reason)
