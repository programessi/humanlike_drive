"""Open-loop feasibility probes, explicitly NOT a learned policy evaluation.

Use predetermined wrist velocities, never a wheel-angle error servo. The
separate audit_regrasp_route.py script checks a waypoint regrasp route.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wheel_manipulation.regrasp.env import RegraspEnv, RegraspConfig


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for support in ("both", "left", "right"):
        free = {"both": -1, "left": 1, "right": 0}[support]
        for phase in (0., 20., -20., 40., -40.):
            cfg = RegraspConfig(curriculum="A", target_mode="hold", episode_seconds=14,
                                randomization_strength=0, closure_jitter_rad=0, settle_jitter_seconds=0)
            env = RegraspEnv(cfg)
            try:
                _, prep = env.reset(seed=1007, options={"free_side": free, "phase_deg": phase})
                rows = []
                start = env.wheel_angle
                expected = start
                for step in range(280):
                    t = step * env.dt
                    speed = 0. if t < 1 or t >= 13 else 2. if t < 5 else -2. if t < 9 else 2.
                    action = np.zeros(24)
                    local = (env.controller.wrist_positions - env.controller.center) @ env.controller.axis_frame
                    for side in range(2):
                        if side == free:
                            action[12 + 6 * side:18 + 6 * side] = -1
                            continue
                        radius = np.linalg.norm(local[side, 1:])
                        action[6 * side] = radius * np.deg2rad(speed) / cfg.controller.wrist_linear_speed
                        action[6 * side + 3] = speed / cfg.controller.wrist_angular_speed_deg
                    expected += np.deg2rad(speed) * env.dt
                    _, _, done, truncated, info = env.step(action)
                    info.update(open_loop_expected_deg=float(np.rad2deg(expected)),
                                open_loop_error_deg=float(np.rad2deg(env.wheel_angle - expected)))
                    rows.append(info)
                    if done or truncated:
                        break
                with (output / f"{support}_{phase:+.0f}.csv").open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
                torque = np.array([[r[f"{side}_wheel_contact_torque_nm"] for side in ("left", "right")] for r in rows])
                item = {"support": support, "initial_phase_deg": phase, "steps": len(rows),
                        "wheel_range_deg": [min(np.rad2deg(r["wheel_angle"]) for r in rows), max(np.rad2deg(r["wheel_angle"]) for r in rows)],
                        "open_loop_rmse_deg": float(np.sqrt(np.mean([r["open_loop_error_deg"]**2 for r in rows]))),
                        "mean_abs_hand_torque_nm": np.abs(torque).mean(0).tolist(),
                        "mean_hand_contact_force_norm_sum_n": [float(np.mean([r[f"{s}_force"] for r in rows])) for s in ("left", "right")],
                        "peak_robot_collision_n": max(r["robot_collision_peak"] for r in rows),
                        "termination": rows[-1]["termination_reason"], "preparation": prep}
                results.append(item)
                print(json.dumps({k: v for k, v in item.items() if k != "preparation"}), flush=True)
                (output / "progress.json").write_text(json.dumps(results, indent=2))
            except ValueError as error:
                results.append({"support": support, "initial_phase_deg": phase, "initialization_error": str(error)})
            finally:
                env.close()
    (output / "summary.json").write_text(json.dumps({"kind": "scripted_open_loop_physical_diagnostic_not_RL",
        "cases": results, "stage1_passed": False,
        "limits": "No learned handover. Rotation under preset wrist motion only tests local physical controllability."}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
