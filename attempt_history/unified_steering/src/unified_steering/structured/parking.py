"""Reset-only, explicitly bounded free-wrist poses and their physical audit.

This module adds no execution-time wrist constraint or hand-role observation.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .env import SteeringTeacherEnv, StructuredController, TaskConfig

# Axial/y/z translation in the wheel frame (m), then rotation vector (deg).
# A finite screened set is intentional; it does not imply a continuous volume.
PROFILES = {
    "nominal": (0, 0, 0, 0, 0, 0),
    "axial_near": (.01, 0, 0, 0, 0, 0),
    "axial_far": (-.01, 0, 0, 0, 0, 0),
    "lateral_plus": (0, .01, 0, 0, 0, 0),
    "lateral_minus": (0, -.01, 0, 0, 0, 0),
    "height_plus": (0, 0, .01, 0, 0, 0),
    "height_minus": (0, 0, -.01, 0, 0, 0),
    "tilt_plus": (0, 0, 0, 0, 5, 0),
    "tilt_minus": (0, 0, 0, 0, -5, 0),
}

SCREENED_RESETS = {
    "near_negative": dict(parking_profile="axial_near", phase_deg=-5., free_opening_delta_rad=-.02),
    "tilt_positive": dict(parking_profile="tilt_plus", phase_deg=5., free_opening_delta_rad=.02),
}


class ParkedController(StructuredController):
    parking_profile = "nominal"

    def preparation_arm(self, phase_deg=0., free_side=-1, free_gap_m=.12):
        result = super().preparation_arm(phase_deg, free_side, free_gap_m)
        if free_side < 0 or self.parking_profile == "nominal":
            return result
        delta = np.asarray(PROFILES[self.parking_profile])
        indices = slice(7*free_side, 7*(free_side+1))
        site = self.grip_ids[free_side]
        self.ik_data.qpos[:] = self.model.qpos0
        self.ik_data.qpos[self.arm_q] = result
        mujoco.mj_kinematics(self.model, self.ik_data)
        position = self.ik_data.site_xpos[site].copy()+self.axis_frame @ delta[:3]
        orientation = (self.axis_frame @ Rotation.from_rotvec(np.deg2rad(delta[3:])).as_matrix()
                       @ self.axis_frame.T @ self.ik_data.site_xmat[site].reshape(3, 3))
        q0 = result[indices].copy()

        def residual(q):
            self.ik_data.qpos[self.arm_q[indices]] = q
            mujoco.mj_kinematics(self.model, self.ik_data)
            actual = self.ik_data.site_xmat[site].reshape(3, 3)
            return np.r_[self.ik_data.site_xpos[site]-position,
                         .12*Rotation.from_matrix(orientation @ actual.T).as_rotvec(), .0002*(q-q0)]

        solution = least_squares(residual, q0,
                                 bounds=(self.limits[indices, 0]+.055, self.limits[indices, 1]-.055),
                                 max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = residual(solution.x)
        if np.linalg.norm(error[:3]) > .003 or np.linalg.norm(error[3:6])/.12 > .04:
            raise ValueError(f"Infeasible parked pose: {self.parking_profile}")
        result[indices] = solution.x
        return result


class SupportTeacherEnv(SteeringTeacherEnv):
    controller_class = ParkedController

    def _randomize(self):
        profile = self.reset_options.get("parking_profile", "nominal")
        if profile not in PROFILES:
            raise ValueError(f"Unknown parking profile: {profile}")
        self.controller.parking_profile = profile
        super()._randomize()
        opening = float(self.reset_options.get("free_opening_delta_rad", 0.))
        if abs(opening) > .03:
            raise ValueError("Opening reset perturbation exceeds screened range")
        if self.free_side >= 0:
            inds = slice(self.free_side*6, (self.free_side+1)*6)
            self.controller.closed_hand[inds] = np.clip(
                self.controller.closed_hand[inds]+opening,
                self.controller.hand_limits[inds, 0]+.01, self.controller.hand_limits[inds, 1]-.01)

    def _advance_action(self, action):
        self.free_force_peak = 0.
        super()._advance_action(action)

    def _contact_metrics(self):
        forces, peak, other = super()._contact_metrics()
        if self.measuring and self.free_side >= 0:
            self.free_force_peak = max(getattr(self, "free_force_peak", 0.), forces[self.free_side])
        return forces, peak, other

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["physics_free_hand_force_peak_n"] = self.free_force_peak
        return obs, reward, terminated, truncated, info

    def free_clearance(self):
        if self.free_side < 0:
            return None
        m = self.model
        free = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.hand_body_groups[self.free_side]
                and (m.geom_contype[g] or m.geom_conaffinity[g])]
        return min(mujoco.mj_geomDistance(m, self.data, a, b, 1., None)
                   for a in free for b in self.wheel_geoms)


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for support, free in (("left", 1), ("right", 0)):
        for profile in PROFILES:
            for phase, opening in ((-5., -.02), (5., .02)):
                env = SupportTeacherEnv(TaskConfig(target_mode="hold", episode_seconds=3.))
                try:
                    _, prep = env.reset(seed=1007, options=dict(free_side=free, phase_deg=phase,
                                           parking_profile=profile, free_opening_delta_rad=opening))
                    clearance = env.free_clearance()
                    env.controller.update = lambda *args: None
                    peak = force = 0.
                    for _ in range(60):
                        _, _, terminated, truncated, info = env.step(np.zeros(24))
                        peak = max(peak, info["physics_peak_error_deg"])
                        force = max(force, info["physics_free_hand_force_peak_n"])
                        if terminated or truncated:
                            break
                    record = dict(support=support, profile=profile, phase_deg=phase, opening_delta_rad=opening,
                                  initial_wheel_deg=float(np.rad2deg(prep["wheel_angle"])),
                                  initial_free_clearance_m=clearance, peak_error_deg=peak,
                                  free_hand_force_peak_n=force, termination_reason=info["termination_reason"],
                                  passed=bool(not terminated and truncated and clearance>=.01 and force<=.2 and peak<=1.))
                except ValueError as error:
                    record = dict(support=support, profile=profile, phase_deg=phase, passed=False, error=str(error))
                finally:
                    env.close()
                records.append(record)
                (output/"progress.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
    (output/"summary.json").write_text(json.dumps(dict(cases=records, passed=all(r["passed"] for r in records)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
