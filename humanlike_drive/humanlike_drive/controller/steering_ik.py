"""Offline steering pose search with wrist posture and free-hand retreat.

Only scratch MjData are changed. These paths must subsequently be replayed
through the original finite motors and contact physics.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from humanlike_drive.controller.basic_grasps import joint_addresses
from humanlike_drive.controller.p0_control import smooth_fraction
from humanlike_drive.env.logitech_scene import ARM_NAMES


class RimIK:
    def __init__(self, model, initial, wrist_weight=.025, margin_deg=3.):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial.qpos
        mujoco.mj_forward(model, self.data)
        self.armq = joint_addresses(model, ARM_NAMES)
        self.jids = np.array([model.joint(n).id for n in ARM_NAMES])
        self.seed = initial.qpos[self.armq].copy()
        self.preload = initial.ctrl[:14]-self.seed
        self.lower = model.jnt_range[self.jids, 0]+np.deg2rad(margin_deg)
        self.upper = model.jnt_range[self.jids, 1]-np.deg2rad(margin_deg)
        self.center = initial.site('wheel_frame').xpos.copy()
        self.axis = initial.site('wheel_frame').xmat.reshape(3, 3)[:, 0].copy()
        self.anchors = [(initial.site(f'{s}_grip').xpos.copy(),
                         initial.site(f'{s}_grip').xmat.reshape(3, 3).copy()) for s in ('left', 'right')]
        self.wrist_weight = wrist_weight
        self.margin_deg = margin_deg
        self.records = []

    def solve(self, side, phase, warm=None, retreat=0., retreat_distance=.07):
        """phase is the hand's world travel about the axis, not wheel state."""
        m, d = self.model, self.data
        qids = self.armq[7*side:7*side+7]
        lower, upper = self.lower[7*side:7*side+7], self.upper[7*side:7*side+7]
        seed = self.seed[7*side:7*side+7]
        warm = seed if warm is None else warm
        pos0, rot0 = self.anchors[side]
        rotation = Rotation.from_rotvec(self.axis*phase).as_matrix()
        target_pos = self.center+rotation@(pos0-self.center)-self.axis*retreat*retreat_distance
        # Untwist while withdrawing. Additional free-hand wrist regularization
        # prevents a redundant elbow branch from snapping at adjacent rim phases.
        target_rot = Rotation.from_rotvec(self.axis*phase*(1.-retreat)).as_matrix()@rot0
        orientation_weight = .16
        name = ('left', 'right')[side]
        site = m.site(f'{name}_grip').id
        torso = m.geom('torso_link_geom_1').id
        moving = [m.geom(f'{name}_{s}_link_geom_1').id for s in
                  ('shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')]
        wrist_weight = self.wrist_weight*min(abs(phase)/np.deg2rad(25.)+retreat, 1.)

        wrist_weight += .035*retreat*smooth_fraction(abs(phase)/np.deg2rad(30.))

        def distances():
            return np.array([mujoco.mj_geomDistance(m, d, torso, g, .1, None) for g in moving])

        def residual(q):
            d.qpos[qids] = q
            mujoco.mj_kinematics(m, d)
            rot = d.site_xmat[site].reshape(3, 3)
            return np.r_[d.site_xpos[site]-target_pos,
                         orientation_weight*Rotation.from_matrix(target_rot@rot.T).as_rotvec(),
                         .0015*(q-warm), .0005*(q-seed),
                         wrist_weight*q[5:7],
                         5.*np.maximum(.006-distances(), 0.)]

        result = least_squares(residual, np.clip(warm, lower+1e-7, upper-1e-7),
                               bounds=(lower, upper), max_nfev=120,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = residual(result.x)
        forearm = d.body(f'{name}_wrist_roll_link').xmat.reshape(3, 3)[:, 0]
        hand = d.body(f'{name}_wrist_yaw_link').xmat.reshape(3, 3)[:, 0]
        ranges = m.jnt_range[self.jids[7*side:7*side+7]]
        margin = np.min(np.minimum(result.x-ranges[:, 0], ranges[:, 1]-result.x))
        report = dict(side=name, phase_deg=float(np.rad2deg(phase)), retreat=float(retreat),
                      position_error_mm=float(np.linalg.norm(error[:3])*1000),
                      rotation_error_deg=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(
                          target_rot@d.site_xmat[site].reshape(3, 3).T).as_rotvec()))),
                      orientation_constrained=bool(orientation_weight > .001),
                      wrist_bend_deg=float(np.rad2deg(np.arccos(np.clip(forearm@hand, -1., 1.)))),
                      min_margin_deg=float(np.rad2deg(margin)),
                      torso_clearance_mm=float(np.min(distances())*1000), q=result.x.tolist())
        self.records.append(report)
        return result.x.copy(), report
