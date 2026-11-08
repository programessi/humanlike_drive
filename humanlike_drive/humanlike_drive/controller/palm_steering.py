"""Non-learning Cartesian steering with bounded palm-pressure regulation.

All IK states are scratch data. Runtime acts exclusively through the original
26 motors; privileged simulation contacts are an explicit engineering input.
"""

from __future__ import annotations
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from humanlike_drive.controller.basic_grasps import joint_addresses
from humanlike_drive.env.logitech_scene import ARM_NAMES


class PalmIK:
    def __init__(self, model, data, position_weight=1.0, rotation_weight=0.14):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        mujoco.mj_forward(model, self.data)
        self.qids = joint_addresses(model, ARM_NAMES)
        self.jids = np.array([model.joint(n).id for n in ARM_NAMES])
        self.vids = model.jnt_dofadr[self.jids]
        self.bounds = model.jnt_range[self.jids].copy() + np.deg2rad([2.2, -2.2])
        self.seed = data.qpos[self.qids].copy()
        self.center = data.site("wheel_frame").xpos.copy()
        self.frame = data.site("wheel_frame").xmat.reshape(3, 3).copy()
        self.axis = self.frame[:, 0]
        self.anchors = [
            (
                data.site(s + "_grip").xpos.copy(),
                data.site(s + "_grip").xmat.reshape(3, 3).copy(),
            )
            for s in ("left", "right")
        ]
        self.rotation_weight = rotation_weight
        self.position_weight = position_weight
        self.torso = model.geom("torso_link_geom_1").id

    def solve(
        self, side, phase=0.0, warm=None, radial_offset=0.0, axial_offset=0.0, twist=0.0
    ):
        m, d = self.model, self.data
        idx = slice(7 * side, 7 * side + 7)
        qids = self.qids[idx]
        seed = self.seed[idx]
        warm = seed if warm is None else warm
        name = ("left", "right")[side]
        site = m.site(name + "_grip").id
        rotation = Rotation.from_rotvec(self.axis * phase).as_matrix()
        p0, r0 = self.anchors[side]
        p = self.center + rotation @ (p0 - self.center)
        radial = p - self.center
        radial -= self.axis * (radial @ self.axis)
        radial /= np.linalg.norm(radial)
        p += radial_offset * radial + axial_offset * self.axis
        target_r = Rotation.from_rotvec(radial * twist).as_matrix() @ rotation @ r0
        moving = [
            m.geom(name + "_" + s + "_link_geom_1").id
            for s in ("shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
        ]

        def error(q):
            d.qpos[qids] = q
            mujoco.mj_kinematics(m, d)
            current = d.site_xmat[site].reshape(3, 3)
            distances = np.array(
                [mujoco.mj_geomDistance(m, d, self.torso, g, 0.1, None) for g in moving]
            )
            return np.r_[
                self.position_weight * (d.site_xpos[site] - p),
                self.rotation_weight
                * Rotation.from_matrix(target_r @ current.T).as_rotvec(),
                0.0005 * (q - warm),
                0.0001 * (q - seed),
                0.002 * q[5:7],
                10 * np.maximum(0.006 - distances, 0),
            ]

        starts = [np.clip(warm, *self.bounds[idx].T)]
        if abs(phase) < 1e-8:
            for elbow in [0.4, 0.8, 1.2]:
                q = starts[0].copy()
                q[1] = 0.1 if side == 0 else -0.1
                q[3] = elbow
                starts.append(q)
        best = None
        for start in starts:
            r = least_squares(
                error,
                np.clip(start, self.bounds[idx, 0] + 1e-8, self.bounds[idx, 1] - 1e-8),
                bounds=tuple(self.bounds[idx].T),
                max_nfev=130,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            if best is None or np.linalg.norm(r.fun) < np.linalg.norm(best.fun):
                best = r
        e = error(best.x)
        ranges = m.jnt_range[self.jids[idx]]
        fa = d.body(name + "_wrist_roll_link").xmat.reshape(3, 3)[:, 0]
        ha = d.body(name + "_wrist_yaw_link").xmat.reshape(3, 3)[:, 0]
        report = dict(
            side=name,
            phase_deg=float(np.rad2deg(phase)),
            position_error_mm=float(np.linalg.norm(e[:3]) * 1000),
            rotation_error_deg=float(
                np.rad2deg(np.linalg.norm(e[3:6]) / self.rotation_weight)
            ),
            wrist_bend_deg=float(np.rad2deg(np.arccos(np.clip(fa @ ha, -1, 1)))),
            wrist_yaw_deg=float(np.rad2deg(best.x[-1])),
            min_margin_deg=float(
                np.rad2deg(
                    np.min(np.minimum(best.x - ranges[:, 0], ranges[:, 1] - best.x))
                )
            ),
            torso_clearance_mm=float(
                min(
                    mujoco.mj_geomDistance(m, d, self.torso, g, 0.1, None)
                    for g in moving
                )
                * 1000
            ),
            q=best.x.tolist(),
        )
        return best.x.copy(), report

    def cartesian_correction(
        self, side, q, radial_m=0.0, tangential_m=0.0, axial_m=0.0
    ):
        m, d = self.model, self.data
        idx = slice(side * 7, side * 7 + 7)
        d.qpos[self.qids[idx]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        site = m.site(("left", "right")[side] + "_grip").id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, site)
        radial = d.site_xpos[site] - self.center
        radial -= self.axis * (radial @ self.axis)
        radial /= np.linalg.norm(radial)
        delta = (
            radial_m * radial
            + tangential_m * np.cross(self.axis, radial)
            + axial_m * self.axis
        )
        jac = np.r_[jp[:, self.vids[idx]], 0.14 * jr[:, self.vids[idx]]]
        dq = np.linalg.lstsq(jac, np.r_[delta, np.zeros(3)], rcond=1e-4)[0]
        return np.clip(q + dq, *self.bounds[idx].T)


class PalmPointIK:
    """Constrain a vendor-palm surface point and inward normal, free palm twist.

    Free twist uses the arm redundancy to keep wrists natural. The surface point
    is a geometric task coordinate, not an added contact pad or force actuator.
    """

    def __init__(self, model, data, point_x=0.10, axial=-0.015, radius=0.14):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        mujoco.mj_forward(model, self.data)
        self.qids = joint_addresses(model, ARM_NAMES)
        self.jids = np.array([model.joint(n).id for n in ARM_NAMES])
        self.vids = model.jnt_dofadr[self.jids]
        self.bounds = model.jnt_range[self.jids].copy() + np.deg2rad([2.2, -2.2])
        self.seed = data.qpos[self.qids].copy()
        self.center = data.site("wheel_frame").xpos.copy()
        self.frame = data.site("wheel_frame").xmat.reshape(3, 3).copy()
        self.axis = self.frame[:, 0].copy()
        self.frame = (
            Rotation.from_rotvec(
                -self.axis * float(data.joint("wheel_joint").qpos[0])
            ).as_matrix()
            @ self.frame
        )
        self.local = [np.array([point_x, -0.025, 0.0]), np.array([point_x, 0.025, 0.0])]
        self.axial = axial
        self.radius = radius
        self.torso = model.geom("torso_link_geom_1").id

    def solve(
        self,
        side,
        phase=0.0,
        warm=None,
        radial_offset=0.0,
        axial_offset=0.0,
        multi_start=False,
    ):
        m, d = self.model, self.data
        idx = slice(7 * side, 7 * side + 7)
        qids = self.qids[idx]
        name = ("left", "right")[side]
        bid = m.body(name + "_wrist_yaw_link").id
        warm = self.seed[idx] if warm is None else warm
        radial = Rotation.from_rotvec(self.axis * phase).apply(
            self.frame[:, 1] * (1 if side == 0 else -1)
        )
        target = (
            self.center
            + (self.axial + axial_offset) * self.axis
            + (self.radius + radial_offset) * radial
        )
        moving = [
            m.geom(name + "_" + s + "_link_geom_1").id
            for s in ("shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
        ]
        short = ("L", "R")[side]
        hs = [
            i
            for i in range(m.ngeom)
            if m.geom_contype[i] == 2 and m.geom(i).name.startswith(short + "_")
        ]
        house = m.geom("housing_collision").id

        def error(q):
            d.qpos[qids] = q
            mujoco.mj_kinematics(m, d)
            r = d.xmat[bid].reshape(3, 3)
            p = d.xpos[bid] + r @ self.local[side]
            normal = r[:, 1] * (-1 if side == 0 else 1)
            dis = np.array(
                [mujoco.mj_geomDistance(m, d, self.torso, g, 0.1, None) for g in moving]
            )
            hd = np.array(
                [mujoco.mj_geomDistance(m, d, house, g, 0.1, None) for g in hs]
            )
            return np.r_[
                p - target,
                0.08 * (normal + radial),
                0.04 * q[5:7],
                0.0005 * (q - warm),
                10 * np.maximum(0.006 - dis, 0),
                5 * np.maximum(0.005 - hd, 0),
            ]

        starts = [warm.copy()]
        if multi_start:
            for e in [0.5, 1.0, 1.5]:
                q = warm.copy()
                q[1] = 0.1 if side == 0 else -0.1
                q[3] = e
                starts.append(q)
        best = None
        for start in starts:
            result = least_squares(
                error,
                np.clip(start, self.bounds[idx, 0] + 1e-8, self.bounds[idx, 1] - 1e-8),
                bounds=tuple(self.bounds[idx].T),
                max_nfev=160,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
        e = error(best.x)
        r = d.xmat[bid].reshape(3, 3)
        fa = d.body(name + "_wrist_roll_link").xmat.reshape(3, 3)[:, 0]
        ranges = m.jnt_range[self.jids[idx]]
        report = dict(
            side=name,
            phase_deg=float(np.rad2deg(phase)),
            radial_offset_m=radial_offset,
            axial_offset_m=axial_offset,
            position_error_mm=float(np.linalg.norm(e[:3]) * 1000),
            normal_error_deg=float(
                np.rad2deg(2 * np.arcsin(min(np.linalg.norm(e[3:6]) / 0.16, 1.0)))
            ),
            wrist_bend_deg=float(np.rad2deg(np.arccos(np.clip(fa @ r[:, 0], -1, 1)))),
            wrist_yaw_deg=float(np.rad2deg(best.x[-1])),
            min_margin_deg=float(
                np.rad2deg(
                    np.min(np.minimum(best.x - ranges[:, 0], ranges[:, 1] - best.x))
                )
            ),
            q=best.x.tolist(),
        )
        return best.x.copy(), report

    def correction(self, side, q, phase, inset=0.0, lead=0.0, axial=0.0):
        """Small Cartesian correction about the planned palm surface point."""
        m, d = self.model, self.data
        idx = slice(side * 7, side * 7 + 7)
        d.qpos[self.qids[idx]] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        bid = m.body(("left", "right")[side] + "_wrist_yaw_link").id
        r = d.xmat[bid].reshape(3, 3)
        point = d.xpos[bid] + r @ self.local[side]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, point, bid)
        radial = Rotation.from_rotvec(self.axis * phase).apply(
            self.frame[:, 1] * (1 if side == 0 else -1)
        )
        delta = (
            -inset * radial
            + lead * self.radius * np.cross(self.axis, radial)
            + axial * self.axis
        )
        jac = np.r_[jp[:, self.vids[idx]], 0.10 * jr[:, self.vids[idx]]]
        dq = np.linalg.lstsq(jac, np.r_[delta, np.zeros(3)], rcond=1e-4)[0]
        return np.clip(q + dq, *self.bounds[idx].T)
