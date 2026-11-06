"""Scripted physical route audit; never report this as learned handover.

Start at +/-40 degrees, open the disadvantaged hand, withdraw, relocate,
approach and close. Then open the former support and probe the new grasp with
predetermined positive/negative wrist motion. No wheel-angle servo is used.
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wheel_manipulation.regrasp.env import RegraspConfig, RegraspEnv
from wheel_manipulation.regrasp.controller import RegraspControlConfig


def smooth(u):
    u = np.clip(u, 0., 1.)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def wrist_action(controller, side, position, rotation, config):
    c = controller
    local = c.axis_frame.T @ (c.wrist_positions[side] - c.center)
    phi = np.arctan2(local[2], local[1])
    tangent = c.axis_frame @ np.array([0., -np.sin(phi), np.cos(phi)])
    radial = c.axis_frame @ np.array([0., np.cos(phi), np.sin(phi)])
    velocity = 8 * (position - c.wrist_positions[side])
    translation = np.array([tangent @ velocity, radial @ velocity, c.axis_frame[:, 0] @ velocity]) / config.wrist_linear_speed
    omega = 8 * Rotation.from_matrix(rotation @ c.wrist_rotations[side].T).as_rotvec()
    return np.r_[translation, (c.axis_frame.T @ omega) / np.deg2rad(config.wrist_angular_speed_deg)]


def run(output, posture_weight=.01):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for direction in (1, -1):
        free = 1 if direction > 0 else 0
        support = 1 - free
        cfg = RegraspConfig(episode_seconds=28, target_mode="hold", randomization_strength=0,
                            closure_jitter_rad=0, settle_jitter_seconds=0,
                            controller=RegraspControlConfig(posture_weight=posture_weight))
        env = RegraspEnv(cfg)
        rows = []
        try:
            _, prep = env.reset(seed=1007, options={"phase_deg": 40. * direction})
            c = env.controller
            initial_pos, initial_rot = c.poses()
            local_initial = c.axis_frame.T @ (initial_pos[free] - c.center)
            withdrawn = initial_pos[free] - .065 * c.axis_frame[:, 0]
            relocated = c.center + c.axis_frame @ np.array([local_initial[0] - .065, .163 if free == 0 else -.163, 0])
            goal = relocated + .065 * c.axis_frame[:, 0]
            goal_rot = c.axis_frame.copy()
            rotvec = Rotation.from_matrix(goal_rot @ initial_rot[free].T).as_rotvec()
            support_goal = c.center + c.axis_frame @ np.array([-.122, .163 if support == 0 else -.163, 0])
            support_rotation = Rotation.from_matrix(goal_rot @ initial_rot[support].T).as_rotvec()
            for k in range(round(27 / env.dt)):
                t = k * env.dt
                action = np.zeros(24)
                if t < 2:
                    name, position, rotation = "open", initial_pos[free], initial_rot[free]
                elif t < 4:
                    name = "withdraw"
                    position = initial_pos[free] + smooth((t - 2) / 2) * (withdrawn - initial_pos[free])
                    rotation = initial_rot[free]
                elif t < 9:
                    name = "relocate"
                    # This off-wheel path is a straight waypoint segment, not
                    # constrained steering under contact.
                    position = withdrawn + smooth((t - 4) / 5) * (relocated - withdrawn)
                    rotation = Rotation.from_rotvec(smooth((t - 4) / 5) * rotvec).as_matrix() @ initial_rot[free]
                elif t < 11:
                    name = "approach"
                    position = relocated + smooth((t - 9) / 2) * (goal - relocated)
                    rotation = goal_rot
                elif t < 14:
                    name, position, rotation = "close_and_settle", goal, goal_rot
                elif t < 19:
                    name, position, rotation = "release_former_support", goal, goal_rot
                else:
                    name, position, rotation = "new_grasp_motion_probe", None, None
                if position is not None:
                    action[6 * free:6 * free + 6] = wrist_action(c, free, position, rotation, cfg.controller)
                else:
                    speed = 2. if t < 22 else -2. if t < 25 else 0.
                    local = c.axis_frame.T @ (c.wrist_positions[free] - c.center)
                    action[6 * free] = np.linalg.norm(local[1:]) * np.deg2rad(speed) / cfg.controller.wrist_linear_speed
                    action[6 * free + 3] = speed / cfg.controller.wrist_angular_speed_deg
                if t < 11:
                    action[12 + 6 * free:18 + 6 * free] = -1
                if t >= 14:
                    action[12 + 6 * support:18 + 6 * support] = -1
                if t >= 15:
                    u = smooth((t - 15) / 4)
                    position = initial_pos[support] + u * (support_goal - initial_pos[support])
                    rotation = Rotation.from_rotvec(u * support_rotation).as_matrix() @ initial_rot[support]
                    action[6 * support:6 * support + 6] = wrist_action(c, support, position, rotation, cfg.controller)
                _, _, done, truncated, info = env.step(np.clip(action, -1, 1))
                info["audit_phase"] = name
                rows.append(info)
                if done or truncated:
                    break
            with (output / f"route_{direction:+d}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            probe = [r for r in rows if r["time"] >= 19]
            side, peer = ("left", "right") if free == 0 else ("right", "left")
            result = {"direction": direction, "regrasp_hand": side, "steps": len(rows),
                      "termination": rows[-1]["termination_reason"], "preparation": prep, "task": asdict(cfg),
                      "wheel_min_deg": float(np.rad2deg(min(r["wheel_angle"] for r in rows))),
                      "wheel_max_deg": float(np.rad2deg(max(r["wheel_angle"] for r in rows))),
                      "peak_robot_collision_n": max(r["robot_collision_peak"] for r in rows),
                      "new_hand_material_shift_deg": float((rows[-1][f"{side}_material_phase_deg"] - rows[0][f"{side}_material_phase_deg"] + 180) % 360 - 180),
                      "new_hand_joint_margin_gain_deg": rows[-1][f"{side}_joint_margin_deg"] - rows[0][f"{side}_joint_margin_deg"]}
            if probe:
                result.update(probe_contact_fraction=float(np.mean([r[f"{side}_force"] > .5 for r in probe])),
                              probe_peer_contact_fraction=float(np.mean([r[f"{peer}_force"] > .5 for r in probe])),
                              probe_mean_abs_torque_nm=float(np.mean([abs(r[f"{side}_wheel_contact_torque_nm"]) for r in probe])),
                              probe_wheel_range_deg=float(np.rad2deg(max(r["wheel_angle"] for r in probe) - min(r["wheel_angle"] for r in probe))))
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("preparation", "task")}), flush=True)
        finally:
            env.close()
    (output / "summary.json").write_text(json.dumps({"kind": "scripted_route_feasibility_not_policy",
        "cases": results, "stage1_passed": False,
        "limits": "Scripted waypoint motion is not used by the RL execution path. No learned handover evidence."}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--posture-weight", type=float, default=.01)
    a = p.parse_args()
    run(a.output, a.posture_weight)
