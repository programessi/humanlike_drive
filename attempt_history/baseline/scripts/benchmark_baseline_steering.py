"""Offline evaluation of the preserved driving baseline; never trains or actuates a wheel.

Run from the repository root with the existing mujoco Python environment.
The observer copies SteeringEnv._advance verbatim and only reads diagnostics.
An equivalence check against the original environment runs before the suite.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/humanlike-drive-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from drive_baseline.carla_bridge import residual_scale
from drive_baseline.env import SteeringEnv, TaskConfig

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "runs/ppo_track/best/best_model.zip"
REPLAY = ROOT / "outputs/carla_drive_clockwise/drive_metrics.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ObservedEnv(SteeringEnv):
    def _advance(self, target):
        self.interval_peak_N = 0.0
        self.interval_other_N = 0.0
        self.interval_max_joint_speed = 0.0
        self.interval_saturated_substeps = 0
        for _ in range(self.substeps):
            self.data.ctrl[:] = self.controller.torque(self.data, target)
            mujoco.mj_step(self.model, self.data)
            _, peak, other = self._contact_metrics()
            self.interval_peak_N = max(self.interval_peak_N, peak)
            self.interval_other_N = max(self.interval_other_N, other)
            self.interval_max_joint_speed = max(
                self.interval_max_joint_speed,
                float(np.abs(self.data.qvel[self.controller.motor_v]).max()))
            self.interval_saturated_substeps += int(np.any(
                np.abs(self.data.ctrl) >= .98 * self.model.actuator_ctrlrange[:, 1]))


def validate_observer():
    a = SteeringEnv(TaskConfig(episode_seconds=10, integral_gain=1))
    b = ObservedEnv(TaskConfig(episode_seconds=10, integral_gain=1))
    try:
        oa, _ = a.reset(seed=7)
        ob, _ = b.reset(seed=7)
        differences = []
        for k in range(100):
            if k % 5 == 0:
                angle = np.deg2rad(15 * np.sin(k * .02))
                oa = a.set_target(angle, .1)
                ob = b.set_target(angle, .1)
            action = .05 * np.sin(np.arange(26) + k * .02)
            oa, ra, ta, _, _ = a.step(action)
            ob, rb, tb, _, _ = b.step(action)
            differences += [float(np.max(np.abs(a.data.qpos-b.data.qpos))),
                            float(np.max(np.abs(a.data.qvel-b.data.qvel))),
                            float(np.max(np.abs(oa-ob))), abs(ra-rb)]
            assert ta == tb
        maximum = max(differences)
        assert maximum == 0.0, f"Observer changed simulation: {maximum}"
        assert a.model.joint("wheel_joint").id not in a.model.actuator_trnid[:, 0]
        return {"steps": 100, "max_state_observation_reward_difference": maximum,
                "wheel_actuator_present": False}
    finally:
        a.close()
        b.close()


def make_cases(smoke=False):
    cases = []
    if smoke:
        definitions = [("hold_pos15", "hold", 15, 0, 10),
                       ("replay_clockwise", "replay", 0, 0, 10)]
    else:
        definitions = [("hold_center", "hold", 0, 0, 600),
                       ("hold_pos15", "hold", 15, 0, 300),
                       ("hold_neg15", "hold", -15, 0, 300),
                       ("hold_pos30", "hold", 30, 0, 300),
                       ("hold_neg30", "hold", -30, 0, 300),
                       ("sine15_slow", "sine", 15, .02, 600),
                       ("sine30_slow", "sine", 30, .02, 600),
                       ("sine30_fast", "sine", 30, .10, 600),
                       ("stop_reverse", "sequence", 30, 0, 600),
                       ("replay_clockwise", "replay", 0, 0, 300),
                       ("no_contact", "sine", 15, .10, 10)]
    for name, mode, amplitude, frequency, seconds in definitions:
        for policy in ("ppo", "no_ppo"):
            cases.append(dict(name=name, mode=mode, amplitude_deg=amplitude,
                              frequency_hz=frequency, seconds=seconds, seed=7,
                              randomize=False, policy=policy,
                              no_contact=name == "no_contact"))
    if not smoke:
        for seed in (101, 102, 103):
            for name, mode, amplitude, frequency in (
                ("sine30_fast", "sine", 30, .10),
                ("replay_clockwise", "replay", 0, 0)):
                for policy in ("ppo", "no_ppo"):
                    cases.append(dict(name=name, mode=mode, amplitude_deg=amplitude,
                                      frequency_hz=frequency, seconds=300, seed=seed,
                                      randomize=True, policy=policy, no_contact=False))
    return cases


def case_id(case):
    domain = "randomized" if case["randomize"] else "nominal"
    return f"{case['name']}_{domain}_s{case['seed']}_{case['policy']}"


def raw_targets(case):
    times = np.arange(round(case["seconds"] * 10)) * .1
    if case["mode"] == "replay":
        with REPLAY.open() as handle:
            source = list(csv.DictReader(handle))[:len(times)]
        assert len(source) == len(times)
        timestamps = np.array([float(row["time"]) for row in source])
        assert np.max(np.abs(np.diff(timestamps)-.1)) < 1e-9
        # Replay the already processed target exactly, not through a second limiter.
        return (np.array([float(row["requested_wheel_rad"]) for row in source]),
                np.array([float(row["target_wheel_rad"]) for row in source]))
    amplitude = np.deg2rad(case["amplitude_deg"])
    if case["mode"] == "hold":
        raw = np.full(len(times), amplitude)
    elif case["mode"] == "sine":
        raw = amplitude * np.sin(2 * np.pi * case["frequency_hz"] * times)
    else:
        # 120 s cycle: centre, +15, centre, -15, centre, +30,
        # centre, -30, centre. Transitions retain production 20 deg/s slew.
        edges = np.array([10, 25, 35, 50, 60, 80, 90, 110])
        levels = np.deg2rad([0, 15, 0, -15, 0, 30, 0, -30, 0])
        raw = levels[np.searchsorted(edges, times % 120, side="right")]
    return raw, None


def contact_wrench(env):
    """Contact resultant on the wheel, expressed in its fixed axis frame.

    MuJoCo's contact frame normal points geom1 -> geom2; the positive
    contact wrench acts on geom2. Opposing hand resultants are a proxy,
    not a measured wrist wrench or a complete internal-force decomposition.
    """
    forces = np.zeros((2, 3))
    torques = np.zeros((2, 3))
    vector = np.zeros(6)
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        wheel1 = contact.geom1 in env.wheel_geoms
        wheel2 = contact.geom2 in env.wheel_geoms
        if not (wheel1 or wheel2):
            continue
        other = contact.geom2 if wheel1 else contact.geom1
        body = env.model.geom_bodyid[other]
        side = next((s for s, group in enumerate(env.hand_body_groups)
                     if body in group), None)
        if side is None:
            continue
        mujoco.mj_contactForce(env.model, env.data, i, vector)
        sign = -1 if wheel1 else 1
        frame = contact.frame.reshape(3, 3)
        force = sign * (frame.T @ vector[:3])
        torque = sign * (frame.T @ vector[3:])
        forces[side] += force
        torques[side] += torque + np.cross(contact.pos-env.controller.center, force)
    local_forces = forces @ env.controller.axis_frame
    local_torques = torques @ env.controller.axis_frame
    total = local_forces.sum(axis=0)
    opposing = (np.linalg.norm(forces[0]) + np.linalg.norm(forces[1])
                - np.linalg.norm(forces.sum(axis=0))) / 2
    return {"net_axial_contact_force_N": float(total[0]),
            "net_inplane_contact_force_N": float(np.linalg.norm(total[1:])),
            "opposing_hand_force_proxy_N": float(opposing),
            "wheel_axis_contact_torque_Nm": float(local_torques[:, 0].sum())}


def longest_true(mask, dt=.02):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * dt


def summarize(rows, case, initial, parameters, wall_seconds):
    data = {name: np.array([row[name] for row in rows]) for name in rows[0]}
    error = data["error_deg"]
    both = (data["left_force"] > .2) & (data["right_force"] > .2)
    any_hand = (data["left_force"] > .2) | (data["right_force"] > .2)
    windows = []
    for start, end in ((0, 8), (0, 30), (0, 60), (10, case["seconds"])):
        mask = (data["time"] > start-1e-8) & (data["time"] <= end+1e-8)
        if mask.any():
            windows.append(dict(start_s=start, end_s=end, samples=int(mask.sum()),
                                rmse_deg=float(np.sqrt(np.mean(error[mask]**2)))))
    minute_windows = []
    for start in range(0, int(case["seconds"]), 60):
        mask = (data["time"] > start+1e-8) & (data["time"] <= start+60+1e-8)
        if mask.any():
            minute_windows.append(dict(start_s=start, end_s=start+60,
                                      rmse_deg=float(np.sqrt(np.mean(error[mask]**2))),
                                      both_contact_fraction=float(both[mask].mean()),
                                      any_contact_fraction=float(any_hand[mask].mean())))
    end_mask = data["time"] > data["time"][-1]-60+1e-8
    summary = dict(case=case, simulated_seconds=float(data["time"][-1]),
                   requested_seconds=case["seconds"], early_failure=bool(rows[-1]["terminated"]),
                   rmse_deg=float(np.sqrt(np.mean(error**2))),
                   mae_deg=float(np.abs(error).mean()), max_abs_error_deg=float(np.abs(error).max()),
                   final_error_deg=float(error[-1]),
                   last60_rmse_deg=float(np.sqrt(np.mean(error[end_mask]**2))),
                   fraction_within_3deg=float((np.abs(error) < 3).mean()),
                   both_hands_contact_fraction=float(both.mean()),
                   any_hand_contact_fraction=float(any_hand.mean()),
                   longest_no_hand_contact_s=longest_true(~any_hand),
                   longest_not_both_contact_s=longest_true(~both),
                   peak_single_contact_force_N_500hz=float(data["peak_force_500hz"].max()),
                   peak_single_contact_force_N_50hz=float(data["peak_force"].max()),
                   peak_non_hand_wheel_force_N_500hz=float(data["other_force_500hz"].max()),
                   force_over_300N_intervals=int((data["peak_force_500hz"] > 300).sum()),
                   max_reference_abs_deg=float(np.abs(data["reference_deg"]).max()),
                   reference_at_limit_fraction=float((np.abs(data["reference_deg"])
                                                       >= np.rad2deg(.95)-.1).mean()),
                   integral_at_limit_fraction=float((np.abs(data["integral_offset_rad"]) >= .199).mean()),
                   mean_policy_scale=float(data["policy_scale"].mean()),
                   policy_faded_fraction=float((data["policy_scale"] < .999).mean()),
                   peak_arm_reference_speed_rad_s=float(data["arm_reference_speed_rad_s"].max()),
                   peak_actual_motor_speed_rad_s=float(data["motor_speed_500hz"].max()),
                   peak_opposing_hand_force_proxy_N=float(data["opposing_hand_force_proxy_N"].max()),
                   peak_net_axial_contact_force_N=float(np.abs(data["net_axial_contact_force_N"]).max()),
                   max_left_palm_wheel_relative_change_deg=float(np.abs(data["left_palm_wheel_change_deg"]).max()),
                   max_right_palm_wheel_relative_change_deg=float(np.abs(data["right_palm_wheel_change_deg"]).max()),
                   windows=windows, minute_windows=minute_windows,
                   initial_state=initial, physics_parameters=parameters, wall_seconds=wall_seconds)
    if summary["early_failure"]:
        summary["failure_info"] = {key: rows[-1][key] for key in (
            "finite", "peak_force", "non_hand_wheel_force", "wheel_angle")}
    return summary


def plot_trial(directory, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["time"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout="constrained")
    axes[0].plot(t, [r["requested_deg"] for r in rows], label="Raw request", alpha=.4)
    axes[0].plot(t, [r["command_deg"] for r in rows], label="Processed command")
    axes[0].plot(t, np.rad2deg([r["wheel_angle"] for r in rows]), label="Physical wheel")
    axes[0].plot(t, [r["reference_deg"] for r in rows], label="Hand arc reference", alpha=.6)
    axes[0].set_ylabel("Angle (deg)")
    axes[1].plot(t, [r["error_deg"] for r in rows], label="Tracking error")
    axes[1].plot(t, [r["left_palm_wheel_change_deg"] for r in rows], label="Left palm/wheel change")
    axes[1].plot(t, [r["right_palm_wheel_change_deg"] for r in rows], label="Right palm/wheel change")
    axes[1].set_ylabel("Error / relative change (deg)")
    axes[2].plot(t, [r["left_force"] for r in rows], label="Left sum of contact norms")
    axes[2].plot(t, [r["right_force"] for r in rows], label="Right sum of contact norms")
    axes[2].plot(t, [r["peak_force_500hz"] for r in rows], label="Single contact peak (500 Hz)", alpha=.6)
    axes[2].set_ylabel("Force (N)")
    axes[3].plot(t, [r["policy_scale"] for r in rows], label="PPO guard scale")
    axes[3].plot(t, [r["integral_offset_rad"] / .2 for r in rows], label="Integral / bound")
    axes[3].set_ylabel("Controller state")
    for axis in axes:
        axis.legend(loc="upper right", fontsize=8, ncol=2)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(title)
    fig.savefig(directory / "tracking.png", dpi=130)
    plt.close(fig)


def run_case(case, output):
    torch.set_num_threads(1)
    start = time.monotonic()
    directory = Path(output) / case_id(case)
    directory.mkdir(parents=True, exist_ok=False)
    raw, replay_commands = raw_targets(case)
    env = ObservedEnv(TaskConfig(episode_seconds=case["seconds"]+1, integral_gain=1,
                                 randomize=case["randomize"]))
    if case["no_contact"]:
        for geom in env.wheel_geoms:
            env.model.geom_contype[geom] = 0
            env.model.geom_conaffinity[geom] = 0
    policy = PPO.load(POLICY, device="cpu") if case["policy"] == "ppo" else None
    obs, initial = env.reset(seed=case["seed"])
    parameters = {"wheel_damping": float(env.model.dof_damping[env.wheel_v]),
                  "wheel_frictionloss": float(env.model.dof_frictionloss[env.wheel_v]),
                  "rim_friction": env.model.geom_friction[next(iter(env.wheel_geoms))].tolist()}
    previous = env.wheel_angle
    previous_reference = env.controller.reference(env.ref_angle)
    initial_relative = []
    for side, site in enumerate(env.controller.grip_ids):
        local = env.controller.axis_frame.T @ (env.data.site_xpos[site]-env.controller.center)
        sign = 1 if side == 0 else -1
        initial_relative.append(np.arctan2(sign*local[2], sign*local[1])-env.wheel_angle)
    rows = []
    try:
        for block, request in enumerate(raw):
            clipped = float(np.clip(request, -.95, .95))
            command = float(replay_commands[block]) if replay_commands is not None else float(
                previous + np.clip(clipped-previous, -np.deg2rad(2), np.deg2rad(2)))
            velocity = (command-previous) / .1
            previous = command
            obs = env.set_target(command, velocity)
            for substep in range(5):
                scale = float(residual_scale(command, env.wheel_angle)) if policy else 0.0
                action = policy.predict(obs, deterministic=True)[0] * scale if policy else np.zeros(26)
                obs, reward, terminated, _, info = env.step(action)
                assert not np.any(env.data.xfrc_applied), "Unexpected external body force"
                assert not np.any(env.data.qfrc_applied), "Unexpected direct generalized force"
                reference = env.controller.reference(env.ref_angle)
                palm_changes = []
                for side, site in enumerate(env.controller.grip_ids):
                    local = env.controller.axis_frame.T @ (env.data.site_xpos[site]-env.controller.center)
                    sign = 1 if side == 0 else -1
                    relative = np.arctan2(sign*local[2], sign*local[1])-env.wheel_angle
                    palm_changes.append(float(np.rad2deg(relative-initial_relative[side])))
                row = dict(info, requested_deg=float(np.rad2deg(request)), command_deg=float(np.rad2deg(command)),
                           target_velocity_rad_s=velocity, reference_deg=float(np.rad2deg(env.ref_angle)),
                           integral_offset_rad=env.integral_offset, policy_scale=scale,
                           arm_reference_speed_rad_s=float(np.abs(reference-previous_reference).max()/env.dt),
                           motor_speed_500hz=env.interval_max_joint_speed,
                           peak_force_500hz=env.interval_peak_N, other_force_500hz=env.interval_other_N,
                           saturated_substep_fraction=env.interval_saturated_substeps/env.substeps,
                           left_palm_wheel_change_deg=palm_changes[0],
                           right_palm_wheel_change_deg=palm_changes[1],
                           terminated=terminated, reward=reward, **contact_wrench(env))
                rows.append(row)
                previous_reference = reference
                if terminated:
                    break
            if terminated:
                break
        with (directory / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = summarize(rows, case, initial, parameters, time.monotonic()-start)
        if case["no_contact"]:
            assert max(abs(r["wheel_angle"]) for r in rows) < 1e-8
            assert summary["any_hand_contact_fraction"] == 0
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        plot_trial(directory, rows, case_id(case))
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (
        POLICY, REPLAY, ROOT / "assets/cockpit.xml", ROOT / "assets/cockpit.ik.npz",
        ROOT / "src/drive_baseline/env.py", ROOT / "src/drive_baseline/controller.py",
        ROOT / "src/drive_baseline/carla_bridge.py")}
    cases = make_cases(args.smoke)
    protocol = dict(baseline_git_commit="9b4c65fcfe21d9c85e9065c47295a5811c3b5de9",
                    policy="track PPO, deterministic inference",
                    target_hz=10, control_hz=50, physics_hz=500, force_peak_hz=500,
                    target_rate_deg_s=20, integral_gain=1, guard=True,
                    nominal_seed=7, randomized_seeds=[101, 102, 103],
                    observer_validation=validate_observer(), hashes=hashes, cases=cases,
                    versions=dict(mujoco=mujoco.__version__, torch=torch.__version__, numpy=np.__version__),
                    note="No training, regrasp, state writes, wheel actuator, CARLA, or real hardware. "
                         "Production termination remains at 50 Hz; 500 Hz force telemetry does not change it.")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("Observer validated;", len(cases), "cases;", sum(c["seconds"] for c in cases),
          "requested simulation seconds", flush=True)
    summaries = []
    import multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                                mp_context=multiprocessing.get_context("spawn")) as executor:
        tasks = {executor.submit(run_case, case, args.output): case for case in cases}
        for task in concurrent.futures.as_completed(tasks):
            summary = task.result()
            summaries.append(summary)
            (args.output / "partial_summary.json").write_text(json.dumps(summaries, indent=2))
            print(f"{len(summaries)}/{len(cases)} {case_id(summary['case'])}: "
                  f"{summary['simulated_seconds']:.1f}s RMSE={summary['rmse_deg']:.3f}deg "
                  f"both_contact={summary['both_hands_contact_fraction']:.1%} "
                  f"peak={summary['peak_single_contact_force_N_500hz']:.1f}N "
                  f"failure={summary['early_failure']}", flush=True)
    assert all(sha256(ROOT / path) == value for path, value in hashes.items()), "Baseline files changed"
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    fields = ["id", "simulated_seconds", "early_failure", "rmse_deg", "mae_deg", "max_abs_error_deg",
              "last60_rmse_deg", "both_hands_contact_fraction", "any_hand_contact_fraction",
              "longest_no_hand_contact_s", "peak_single_contact_force_N_500hz",
              "reference_at_limit_fraction", "integral_at_limit_fraction"]
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in sorted(summaries, key=lambda s: case_id(s["case"])):
            writer.writerow(dict(id=case_id(summary["case"]),
                                 **{key: summary[key] for key in fields if key != "id"}))
    print("Completed. Baseline hashes unchanged.", flush=True)


if __name__ == "__main__":
    main()
