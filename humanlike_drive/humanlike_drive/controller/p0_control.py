"""Finite 26-drive executor and an explicit contact-driven P0 baseline.

This is an engineering feasibility controller, not the proposed learned policy.
The wheel has no actuator. IK modifies scratch data only, before rollouts.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from humanlike_drive.env.logitech_scene import ARM_NAMES, HAND_NAMES
from humanlike_drive.env.inspire_hand import effective_joint_ranges
from humanlike_drive.controller.basic_grasps import joint_addresses


def smooth_fraction(x):
    z = np.clip(x, 0., 1.)
    return z*z*(3.-2.*z)


def release_motion(seconds, closed, *, regrasp=False):
    """Open fingers away from the housing; reverse this route on approach.

    A straight interpolation to fully extended fingers sweeps the G29 housing
    near +/-30 deg. Keep the four fingers curled until the wrist withdraws.
    Returns wrist retreat fraction and six independent motor targets. Mimic
    descendants remain controlled by the original model equalities.
    """
    half = np.array([.05, .02, .65, .65, .65, .65])
    opened = np.array([.05, .02, .10, .10, .10, .10])
    if regrasp:
        fingers = smooth_fraction(seconds/1.)
        hand = (1.-fingers)*opened+fingers*half
        retreat = 1.-smooth_fraction((seconds-1.)/1.8)
        closure = smooth_fraction((seconds-2.8)/1.4)
        hand = (1.-closure)*hand+closure*closed
    else:
        aperture = smooth_fraction(seconds/.8)
        hand = (1.-aperture)*closed+aperture*half
        retreat = smooth_fraction((seconds-.6)/2.2)
        extension = smooth_fraction((seconds-1.5)/1.3)
        hand = (1.-extension)*hand+extension*opened
    return float(retreat), hand


class RateExecutor:
    """Normalized target rates at 20 Hz, finite position motors at 500 Hz."""

    def __init__(self, model, data, hz=20.):
        names = ARM_NAMES + HAND_NAMES
        self.qids = joint_addresses(model, names)
        self.jids = np.array([model.joint(n).id for n in names])
        if list(model.actuator_trnid[:, 0]) != list(self.jids):
            raise ValueError("26-drive actuator ordering differs from declared action interface")
        self.dt = 1. / hz
        self.vmax = np.r_[np.full(14, .6), np.full(12, 1.)]
        self.preload = np.r_[np.full(14, .20), np.full(12, .45)]
        self.bounds = effective_joint_ranges(model)[self.jids].copy()
        self.bounds[:14] += np.array([np.deg2rad(2.1), -np.deg2rad(2.1)])
        limited = model.actuator_ctrllimited.astype(bool)
        self.bounds[limited, 0] = np.maximum(self.bounds[limited, 0], model.actuator_ctrlrange[limited, 0])
        self.bounds[limited, 1] = np.minimum(self.bounds[limited, 1], model.actuator_ctrlrange[limited, 1])
        self.target = np.clip(data.ctrl.copy(), *self.bounds.T)
        self.previous_action = np.zeros(26)
        self.action = np.zeros(26)

    def step(self, action, actual):
        self.previous_action = self.action.copy()
        self.action = np.clip(np.asarray(action), -1., 1.)
        candidate = self.target + self.action * self.vmax * self.dt
        # Stop target accumulation against a blocked drive. This safety clamp
        # is logged through requested action vs executed position targets.
        lower = np.maximum(self.bounds[:, 0], actual - self.preload)
        upper = np.minimum(self.bounds[:, 1], actual + self.preload)
        candidate = np.clip(candidate, lower, upper)
        # A moving feedback window must never bypass the target rate bound.
        self.target += np.clip(candidate - self.target, -self.vmax * self.dt, self.vmax * self.dt)
        self.target = np.clip(self.target, *self.bounds.T)
        return self.target.copy()

    def follow(self, desired, actual):
        return self.step((desired - self.target) / (self.dt * self.vmax), actual)


class HandPath:
    """Offline warm-started IK of rigidly rotated initial hand anchors.

    Preserves the settled grasp's motor preload. Reachability is reported
    separately; it is never evidence that a physical steering trial succeeded.
    """

    def __init__(self, model, initial, maximum_deg=180., step_deg=5.):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial.qpos
        mujoco.mj_forward(model, self.data)
        self.armq = joint_addresses(model, ARM_NAMES)
        self.jids = [model.joint(n).id for n in ARM_NAMES]
        self.lower = model.jnt_range[self.jids, 0] + np.deg2rad(2.1)
        self.upper = model.jnt_range[self.jids, 1] - np.deg2rad(2.1)
        self.seed = initial.qpos[self.armq].copy()
        self.preload = initial.ctrl[:14] - self.seed
        self.hand_ctrl = initial.ctrl[14:].copy()
        self.frame = initial.site("wheel_frame").xmat.reshape(3, 3).copy()
        self.center = initial.site("wheel_frame").xpos.copy()
        self.axis = self.frame[:, 0]
        self.q0 = float(initial.joint("wheel_joint").qpos[0])
        self.sites = [model.site(f"{s}_grip").id for s in ("left", "right")]
        self.anchors = [(initial.site(s).xpos.copy(), initial.site(s).xmat.reshape(3, 3).copy())
                        for s in self.sites]
        self.records = []
        points = [(self.q0, self.seed.copy())]
        for sign in (-1, 1):
            warm = self.seed.copy()
            for degree in np.arange(step_deg, maximum_deg + 1e-5, step_deg):
                theta = self.q0 + sign * np.deg2rad(degree)
                warm, record = self.solve(theta, warm)
                self.records.append(record)
                points.append((theta, warm.copy()))
        points.sort(key=lambda p: p[0])
        self.interpolator = PchipInterpolator([p[0] for p in points], [p[1] for p in points], axis=0)
        self.minimum, self.maximum = points[0][0], points[-1][0]

    def solve(self, theta, warm, retreat=None):
        self.data.joint("wheel_joint").qpos[0] = theta  # offline scratch state
        rotation = Rotation.from_rotvec(self.axis * (theta - self.q0)).as_matrix()
        reports, solution = [], warm.copy()
        for i, site in enumerate(self.sites):
            qids = self.armq[7*i:7*i+7]
            lo, hi = self.lower[7*i:7*i+7], self.upper[7*i:7*i+7]
            anchor_p, anchor_r = self.anchors[i]
            p = self.center + rotation @ (anchor_p - self.center)
            if retreat is not None and i == retreat[0]:
                radial = p-self.center
                radial -= self.axis*(radial@self.axis)
                radial /= np.linalg.norm(radial)
                p += (-.6*self.axis+.8*radial)*retreat[1]
            r = rotation @ anchor_r
            side = ("left", "right")[i]
            torso = self.model.geom("torso_link_geom_1").id
            clearance_geoms = [self.model.geom(f"{side}_{link}_link_geom_1").id
                               for link in ("shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]

            def clearances():
                return np.array([mujoco.mj_geomDistance(self.model, self.data, torso, g, .1, None)
                                 for g in clearance_geoms])

            def error(q):
                self.data.qpos[qids] = q
                mujoco.mj_kinematics(self.model, self.data)
                current_r = self.data.site_xmat[site].reshape(3, 3)
                return np.r_[self.data.site_xpos[site] - p,
                             .16 * Rotation.from_matrix(r @ current_r.T).as_rotvec(),
                             .0008 * (q - self.seed[7*i:7*i+7]),
                             5.*np.maximum(.004-clearances(), 0.),
                             .002*np.maximum(np.deg2rad(4.)-np.minimum(q-lo, hi-q), 0.)]

            result = least_squares(error, np.clip(warm[7*i:7*i+7], lo + 1e-7, hi - 1e-7),
                                   bounds=(lo, hi), max_nfev=90,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            solution[7*i:7*i+7] = result.x
            e = error(result.x)
            reports.append({"side": ("left", "right")[i],
                            "position_error_mm": float(np.linalg.norm(e[:3])*1000),
                            "rotation_error_deg": float(np.rad2deg(np.linalg.norm(e[3:6])/.16)),
                            "torso_clearance_mm": float(1000*np.min(clearances())),
                            "joint_margin_deg": float(np.rad2deg(np.min(np.minimum(
                                result.x-self.model.jnt_range[self.jids[7*i:7*i+7], 0],
                                self.model.jnt_range[self.jids[7*i:7*i+7], 1]-result.x))))})
        return solution, {"angle_deg": float(np.rad2deg(theta)), "arms": reports}

    def command(self, theta):
        return np.r_[self.interpolator(np.clip(theta, self.minimum, self.maximum)) + self.preload,
                     self.hand_ctrl]

    def retreat_commands(self, theta, side, distance=.06):
        """Offline release toward the driver/outside rim, keeping the other anchor."""
        warm = self.interpolator(theta).copy()
        samples = np.linspace(0., distance, 13)
        points, reports = [], []
        for displacement in samples:
            warm, report = self.solve(theta, warm, retreat=(side, displacement))
            points.append(warm[7*side:7*side+7].copy()+self.preload[7*side:7*side+7])
            reports.append({"retreat_m": float(displacement), **report})
        return PchipInterpolator(samples/distance, points, axis=0), reports


class RoundTrip:
    """5 deg/s trapezoidal velocity, cosine acceleration ramps, 3 s holds."""

    def __init__(self, target_deg, start_deg=0., speed_deg_s=5., ramp_s=.5,
                 initial_hold_s=2., hold_s=3., target_hold_s=None):
        self.start, self.target = np.deg2rad([start_deg, target_deg])
        self.speed = np.deg2rad(speed_deg_s)
        self.ramp = ramp_s
        self.move = abs(self.target-self.start)/self.speed + ramp_s if self.target != self.start else 0.
        if 0. < self.move < 2*ramp_s:
            raise ValueError("Angle too small for the configured speed ramps")
        self.initial_hold, self.hold = initial_hold_s, hold_s if target_hold_s is None else target_hold_s
        self.duration = initial_hold_s + 2*self.move + self.hold + hold_s

    def _fraction(self, t):
        t = np.clip(t, 0., self.move)
        r, v = self.ramp, self.speed
        if t < r:
            distance = v/2*(t-r/np.pi*np.sin(np.pi*t/r))
        elif t <= self.move-r:
            distance = v*(t-r/2)
        else:
            remain = self.move-t
            distance = abs(self.target-self.start)-v/2*(remain-r/np.pi*np.sin(np.pi*remain/r))
        return distance/abs(self.target-self.start)

    def sample(self, t):
        if t < self.initial_hold:
            return self.start, 0, self.target
        t -= self.initial_hold
        if t < self.move:
            return self.start + (self.target-self.start)*self._fraction(t), 1, self.target
        t -= self.move
        if t < self.hold:
            return self.target, 2, self.target
        t -= self.hold
        if t < self.move:
            return self.target + (self.start-self.target)*self._fraction(t), 3, self.start
        return self.start, 4, self.start
