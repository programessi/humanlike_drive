"""Separate kinematic reachability from physically settled contact evidence."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from wheel_manipulation.scene import build_scene
from wheel_manipulation.exploration.env import CentralizedWheelEnv, ExplorationConfig


def finger_branch(name):
    for prefix in ("L_", "R_"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            if name.startswith(prefix + finger + "_"):
                return prefix + finger
    return None


def build_audited_scene(path):
    build_scene(path)
    tree = ET.parse(path)
    root = tree.getroot()
    removed = []
    for item in list(root.find("contact")):
        if item.tag != "exclude":
            continue
        a, b = finger_branch(item.get("body1", "")), finger_branch(item.get("body2", ""))
        if a and b and a != b:
            removed.append(dict(item.attrib))
            root.find("contact").remove(item)
    root.set("model", "h1_2_regrasp_interfinger_collision_audit")
    ET.indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return removed


def collision_rows(env):
    rows = []
    for con in env.data.contact:
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in env.wheel_geoms or g2 in env.wheel_geoms:
            continue
        if con.dist < -.001:
            rows.append({"body1": env.model.body(int(env.model.geom_bodyid[g1])).name,
                         "body2": env.model.body(int(env.model.geom_bodyid[g2])).name,
                         "penetration_mm": float(-1000 * con.dist)})
    return rows


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    scene = output / "wheel.xml"
    removed = build_audited_scene(scene)
    config = ExplorationConfig(randomization_strength=0, closure_jitter_rad=0,
                               settle_jitter_seconds=0, target_mode="hold")
    env = CentralizedWheelEnv(config, scene_path=scene)
    try:
        _, info = env.reset(seed=7)
        initial = {"wheel_deg": float(np.rad2deg(env.wheel_angle)),
                   "contact_force_norm_sum_n": env.contact_forces.tolist(),
                   "nonwheel_penetrations": collision_rows(env),
                   "active_positions": env.data.qpos[env.controller.motor_q].tolist()}
        for _ in range(20):
            _, _, done, _, info = env.step(np.zeros(24))
            if done:
                break
        initial["after_1s"] = {"wheel_deg": float(np.rad2deg(env.wheel_angle)),
                               "contact_force_norm_sum_n": env.contact_forces.tolist(),
                               "robot_collision_peak_n": info["robot_collision_peak"],
                               "nonwheel_penetrations": collision_rows(env)}
        c = env.controller
        env.reset(seed=7)
        origin = env.data.site_xpos[c.grip_ids].copy()
        rotations = env.data.site_xmat[c.grip_ids].reshape(2, 3, 3).copy()
        physical_q = env.data.qpos.copy()
        rows = []
        for direction in (1, -1):
            previous_q = env.data.qpos[c.arm_q].copy()
            for degrees in np.arange(0, 101, 5) * direction:
                rot = c.axis_frame @ Rotation.from_rotvec([np.deg2rad(degrees), 0, 0]).as_matrix() @ c.axis_frame.T
                positions = c.center + (origin - c.center) @ rot.T
                targets = rot @ rotations
                env.data.qpos[:] = physical_q
                errors = []
                for side, site in enumerate(c.grip_ids):
                    inds = slice(side * 7, (side + 1) * 7)
                    q0 = previous_q[inds].copy()
                    def objective(q):
                        env.data.qpos[c.arm_q[inds]] = q
                        mujoco.mj_kinematics(env.model, env.data)
                        r = env.data.site_xmat[site].reshape(3, 3)
                        return np.r_[env.data.site_xpos[site] - positions[side],
                                     .12 * Rotation.from_matrix(targets[side] @ r.T).as_rotvec(),
                                     .0002 * (q - q0)]
                    lower, upper = c.limits[inds, 0] + .03, c.limits[inds, 1] - .03
                    sol = least_squares(objective, np.clip(q0, lower, upper), bounds=(lower, upper),
                                        max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                    residual = objective(sol.x)
                    errors.append((float(np.linalg.norm(residual[:3]) * 1000),
                                   float(np.rad2deg(np.linalg.norm(residual[3:6]) / .12))))
                    previous_q[inds] = sol.x
                env.data.qpos[env.wheel_q] = np.deg2rad(degrees)
                mujoco.mj_forward(env.model, env.data)
                margins = np.minimum(previous_q - c.limits[:, 0], c.limits[:, 1] - previous_q)
                collisions = collision_rows(env)
                rows.append({"wheel_deg": int(degrees), "left_position_error_mm": errors[0][0],
                             "right_position_error_mm": errors[1][0], "left_orientation_error_deg": errors[0][1],
                             "right_orientation_error_deg": errors[1][1],
                             "min_joint_margin_deg": float(np.rad2deg(margins.min())),
                             "nonwheel_penetration_count": len(collisions),
                             "max_nonwheel_penetration_mm": max([x["penetration_mm"] for x in collisions], default=0),
                             "collision_pairs": json.dumps(collisions)})
        with (output / "fixed_grasp_scan.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = {"kind": "kinematic_audit_not_policy_or_dynamic_regrasp_success",
                   "scene": str(scene), "restored_interfinger_exclusions": len(removed),
                   "restored_pairs": removed, "physical_nominal_preparation": initial,
                   "scan": rows, "stage1_passed": False}
        (output / "audit.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"restored_pairs": len(removed), "physical_preparation": initial,
                          "kinematic_scan_rows": len(rows)}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    audit(p.parse_args().output)
