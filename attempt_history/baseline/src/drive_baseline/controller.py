"""IK references and bounded motor control; no wheel actuation or state writes."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .scene import ARM_NAMES, HAND_NAMES


class SteeringController:
    def __init__(self, model):
        self.model = model
        self.ik_data = mujoco.MjData(model)
        self.arm_jids = np.array([model.joint(n).id for n in ARM_NAMES])
        self.arm_q = model.jnt_qposadr[self.arm_jids]
        self.arm_v = model.jnt_dofadr[self.arm_jids]
        self.hand_jids = np.array([model.joint(n).id for n in HAND_NAMES])
        self.hand_q = model.jnt_qposadr[self.hand_jids]
        self.hand_v = model.jnt_dofadr[self.hand_jids]
        self.motor_q = np.r_[self.arm_q, self.hand_q]
        self.motor_v = np.r_[self.arm_v, self.hand_v]
        self.limits = model.jnt_range[self.arm_jids].copy()
        self.hand_limits = model.jnt_range[self.hand_jids].copy()
        # Thumb pitch is restricted by the coupled distal joint limits.
        self.hand_limits[[1, 7], 1] = .49
        self.center_id = model.site("wheel_center").id
        self.grip_ids = [model.site(f"{s}_grip").id for s in ("left", "right")]
        mujoco.mj_forward(model, self.ik_data)
        self.center = self.ik_data.site_xpos[self.center_id].copy()
        self.axis_frame = self.ik_data.site_xmat[self.center_id].reshape(3, 3).copy()
        self.seed = np.array([-.65, .10, .1, 1.15, 0, 0, 0, -.65, -.1, -.1, 1.15, 0, 0, 0])
        self.open_hand = np.tile([.3, .1, .15, .15, .15, .15], 2)
        self.closed_hand = np.tile([.6, .42, 1.0, 1.0, 1.0, 1.0], 2)
        self.table_angles = np.linspace(-np.deg2rad(65), np.deg2rad(65), 53)
        self.table = None
        self.max_ik_error = 0.0
        self.kp = np.r_[np.tile([140, 140, 100, 100, 35, 35, 35], 2), np.full(12, 1.5)]
        self.kd = np.r_[np.tile([12, 12, 8, 8, 3, 3, 3], 2), np.full(12, .06)]

    def solve_ik(self, angle, seed=None):
        seed = np.array(self.seed if seed is None else seed, copy=True)
        rot = self.axis_frame @ Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        result = seed.copy()
        errors = []
        for side, site in enumerate(self.grip_ids):
            inds = slice(side * 7, side * 7 + 7)
            # Palm/finger bases are just outside the tube at 9 and 3 o'clock.
            target = self.center + rot @ np.array([-.002, (.163 if side == 0 else -.163), 0])
            q0 = np.clip(seed[inds], self.limits[inds, 0] + .001, self.limits[inds, 1] - .001)

            def objective(q):
                self.ik_data.qpos[self.arm_q[inds]] = q
                mujoco.mj_kinematics(self.model, self.ik_data)
                xyz = self.ik_data.site_xpos[site] - target
                current = self.ik_data.site_xmat[site].reshape(3, 3)
                angular = Rotation.from_matrix(rot @ current.T).as_rotvec()
                return np.r_[xyz, .12 * angular, .003 * (q - q0)]

            opt = least_squares(objective, q0, bounds=(self.limits[inds, 0] + .001, self.limits[inds, 1] - .001), max_nfev=100, ftol=1e-7)
            result[inds] = opt.x
            errors.append(float(np.linalg.norm(objective(opt.x)[:3])))
        return result, max(errors)

    def build_table(self):
        middle = len(self.table_angles) // 2
        self.table = np.zeros((len(self.table_angles), 14))
        self.table[middle], error = self.solve_ik(0)
        self.max_ik_error = error
        for direction in (-1, 1):
            prev = middle
            for i in range(middle + direction, len(self.table_angles) if direction > 0 else -1, direction):
                self.table[i], error = self.solve_ik(self.table_angles[i], self.table[prev])
                self.max_ik_error = max(self.max_ik_error, error)
                prev = i
        return self.table

    def reference(self, angle):
        if self.table is None:
            self.build_table()
        return np.array([np.interp(angle, self.table_angles, self.table[:, i]) for i in range(14)])

    def torque(self, data, target, gravity_compensation=True):
        tau = self.kp * (target - data.qpos[self.motor_q]) - self.kd * data.qvel[self.motor_v]
        if gravity_compensation:
            tau += data.qfrc_bias[self.motor_v]
        return np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
