"""Matched initial-state PD integration check and wheel-contact causality."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wheel_manipulation.common.hardware import RobotHardware
from wheel_manipulation.env import ContactResidualEnv, ContactTaskConfig
from wheel_manipulation.common.kinematics import kinematic_sweep


class IntegrationProbeEnv(ContactResidualEnv):
    def __init__(self, config, explicit=False):
        self.explicit = explicit
        super().__init__(config)

    def _randomize(self):
        super()._randomize()
        if self.explicit:
            m = self.model
            m.actuator_gainprm[:, 0] = 1
            m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_biasprm[:] = 0
            m.actuator_ctrllimited[:] = True
            m.actuator_forcelimited[:] = False

    def _set_initial_mimic(self):
        super()._set_initial_mimic()
        # Record the actual physical initialization before the grasp warmup.
        self.pregrasp_state = np.r_[self.data.qpos, self.data.qvel].tolist()


def motor_case(explicit=False, seconds=10):
    env = IntegrationProbeEnv(ContactTaskConfig(target_mode="hold", amplitude_deg=0, episode_seconds=seconds), explicit)
    c, m = env.controller, env.model
    if explicit:
        c.torque = lambda data, target: RobotHardware.torque(c, data, target)
    original = c.torque
    samples = []
    grasp_phase = True
    def capture(data, target):
        samples.append({"time": len(samples) * m.opt.timestep, "episode_time": float(data.time),
                        "phase": "grasp_initialization" if grasp_phase else "rollout",
                        "wheel_angle_deg": float(np.rad2deg(env.wheel_angle)),
                        "wheel_velocity_deg_s": float(np.rad2deg(data.qvel[env.wheel_v])),
                        "right_wrist_roll_velocity_rad_s": float(data.qvel[c.arm_v[11]]),
                        "max_arm_velocity_rad_s": float(np.max(np.abs(data.qvel[c.arm_v]))),
                        "max_motor_force_fraction": float(np.max(np.abs(data.actuator_force) / env.nominal_ctrlrange[:, 1]))})
        return original(data, target)
    c.torque = capture
    _, initial = env.reset(seed=1001)
    grasp_phase = False
    error = []
    for _ in range(round(seconds / env.dt)):
        _, _, terminated, truncated, info = env.step(np.zeros(18))
        error.append(info["error_deg"])
        if terminated or truncated:
            break
    velocity = np.array([s["right_wrist_roll_velocity_rad_s"] for s in samples])
    result = {"execution": "explicit" if explicit else "implicit", "postgrasp_state": initial,
              "pregrasp_state": env.pregrasp_state,
              "physics_sampling_hz": 1 / m.opt.timestep, "seconds": info["time"],
              "early_failure": terminated, "wheel_rmse_deg": float(np.sqrt(np.mean(np.array(error) ** 2))),
              "max_arm_velocity_rad_s": max(s["max_arm_velocity_rad_s"] for s in samples),
              "right_wrist_adjacent_velocity_correlation": float(np.corrcoef(velocity[1:], velocity[:-1])[0, 1]),
              "max_motor_force_fraction": max(s["max_motor_force_fraction"] for s in samples),
              "max_grasp_drift_deg": info["max_grasp_drift_deg"],
              "max_hold_seconds": info["max_hold_seconds"]}
    env.close()
    return result, samples


def contact_case(no_contact):
    env = ContactResidualEnv(ContactTaskConfig(episode_seconds=5, no_contact=no_contact))
    _, initial = env.reset(seed=1001, options={"target_deg": 10})
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(np.zeros(18))
        if terminated or truncated:
            break
    change = float(np.rad2deg(env.wheel_angle - initial["wheel_angle"]))
    result = {"no_contact": no_contact, "wheel_change_deg": change, "early_failure": terminated,
              "final_error_deg": info["error_deg"], "reference_phase_deg": info["common_reference_phase_deg"],
              "passed": not terminated and (abs(change) < 1e-8 if no_contact else change > 3)}
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for explicit in (True, False):
        result, samples = motor_case(explicit)
        cases.append(result)
        with (args.output / f"{result['execution']}_500hz.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        print(json.dumps(result, indent=2), flush=True)
    contacts = [contact_case(False), contact_case(True)]
    report = {"motor_integration": cases, "causality": contacts, "kinematics": kinematic_sweep(),
              "identical_pregrasp_state": cases[0]["pregrasp_state"] == cases[1]["pregrasp_state"],
              "passed": all(c["passed"] for c in contacts) and cases[1]["max_motor_force_fraction"] <= 1 + 1e-8,
              "implementation_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path, (
                  "src/wheel_manipulation/common/hardware.py", "src/wheel_manipulation/common/simulation.py",
                  "src/wheel_manipulation/controller.py", "src/wheel_manipulation/env.py", "assets/wheel.xml"))}}
    report["passed"] &= report["identical_pregrasp_state"] and report["kinematics"]["passed"]
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
