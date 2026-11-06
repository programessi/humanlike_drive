"""Fresh static IK presets for inspecting the new G29 bench (not learned skills)."""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from humanlike_drive.env.logitech_scene import ARM_NAMES, HAND_NAMES, SCENE, PROJECT, vec
from humanlike_drive.env.inspire_hand import apply_mimic, effective_joint_ranges, mimic_errors
from humanlike_drive.controller.grasp_contacts import contact_distribution

PRESETS = {
    "nine_three": {"angles": [0., 0.], "active": [True, True]},
    "ten_two": {"angles": [30., -30.], "active": [True, True]},
    "left_support": {"angles": [0., 0.], "active": [True, False]},
    "right_support": {"angles": [0., 0.], "active": [False, True]},
}
TARGET_CONFIG = PROJECT / "humanlike_drive/config/controller/grasp_targets.json"
_targets = json.loads(TARGET_CONFIG.read_text())
for _name, _preset in PRESETS.items():
    _preset.update(copy.deepcopy(_targets["active_grasp"]))
    _preset.update(copy.deepcopy(_targets.get("presets", {}).get(_name, {})))
    _preset["label"] = _targets["labels"][_name]
PRESET_FILE = PROJECT / "assets/scenes/basic_grasps.json"


def joint_addresses(model, names):
    return np.array([model.jnt_qposadr[model.joint(n).id] for n in names])


def coupled_hand(model, data, side, active, values=None):
    if values is None:
        values = _targets["active_grasp"]["hand_drives"] if active else [.05, .02, .10, .10, .10, .10]
    ranges = effective_joint_ranges(model)
    for suffix, value in zip(("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                              "middle_proximal", "ring_proximal", "pinky_proximal"), values):
        joint = model.joint(f"{side}_{suffix}_joint").id
        if not ranges[joint, 0] <= value <= ranges[joint, 1]:
            raise ValueError(f"Drive target violates coupled joint limits: {side}_{suffix}={value}")
        data.qpos[model.jnt_qposadr[joint]] = value
    apply_mimic(model, data, side)


def solve_pose(model, name, preset=None):
    preset = PRESETS[name] if preset is None else preset
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame_site = model.site("wheel_frame").id
    center = data.site_xpos[frame_site].copy()
    frame = data.site_xmat[frame_site].reshape(3, 3).copy()
    armq = joint_addresses(model, ARM_NAMES)
    reports = []
    for i, (side, short) in enumerate((("left", "L"), ("right", "R"))):
        coupled_hand(model, data, short, preset["active"][i],
                     preset.get("hand_drives") if preset["active"][i] else None)
        phi = np.deg2rad(preset["angles"][i])
        radial_frame = frame @ Rotation.from_rotvec([phi, 0, 0]).as_matrix()
        rot = radial_frame @ Rotation.from_euler("y", preset.get("wrist_tilt_deg", -30), degrees=True).as_matrix()
        if preset["active"][i]:
            target = center + radial_frame @ np.array([preset.get("grip_axial_m", -.025),
                                                       preset.get("grip_radius_m", .159) * (1 if i == 0 else -1),
                                                       preset.get("grip_height_m", 0.)])
            if "grip_pose" in preset:
                mirror = np.diag([1., 1. if i == 0 else -1., 1.])
                target = center + radial_frame @ mirror @ np.array(preset["grip_pose"]["position_m"])
                rot = radial_frame @ mirror @ np.array(preset["grip_pose"]["rotation_matrix"]) @ mirror
        else:
            target = center + np.array([-.18, .34 if i == 0 else -.34, -.12])
            rot = frame @ Rotation.from_euler("y", -30, degrees=True).as_matrix()
        qids = armq[i * 7:(i + 1) * 7]
        jids = [model.joint(n).id for n in ARM_NAMES[i * 7:(i + 1) * 7]]
        lower, upper = model.jnt_range[jids, 0] + .035, model.jnt_range[jids, 1] - .035
        # Pose search preference, not a change to physical joint limits.
        upper[3] = min(upper[3], 2.2)
        site = model.site(f"{side}_grip").id
        elbow = model.body(f"{side}_elbow_link").id
        elbow_target = np.array([.13, .30 if i == 0 else -.30, model.body("pelvis").pos[2] + .14])
        posture = preset.get("arm_posture") if preset["active"][i] else None
        if posture:
            elbow_target = np.array(posture["left_elbow_preference_m"], dtype=float)
            elbow_target[1] *= 1 if i == 0 else -1
            wrist_yaw_target = np.deg2rad(posture["wrist_yaw_target_deg"]) * (1 if i == 0 else -1)
        seed = np.array([-.65, .3, .1, 1.15, 0, 0, 0])
        if i:
            seed[[1, 2, 4, 6]] *= -1
        def residual(q):
            data.qpos[qids] = q
            mujoco.mj_kinematics(model, data)
            current = data.site_xmat[site].reshape(3, 3)
            task_error = np.r_[data.site_xpos[site] - target,
                               .16 * Rotation.from_matrix(rot @ current.T).as_rotvec()]
            if posture:
                # Prioritize the configured hand pose, then use arm redundancy
                # to reduce wrist deviation and keep a relaxed elbow position.
                return np.r_[task_error, .0015 * (data.xpos[elbow] - elbow_target),
                             .0001 * (q - seed), .00005 * q[4:6],
                             .004 * (q[6] - wrist_yaw_target)]
            return np.r_[task_error,
                         .08 * (data.xpos[elbow] - elbow_target),
                         .005 * (q - seed), .01 * q[4:]]
        best = None
        for k in range(4):
            start = seed.copy()
            start[3] += k * .25
            result = least_squares(residual, np.clip(start, lower + 1e-5, upper - 1e-5),
                                   bounds=(lower, upper), max_nfev=220,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
        data.qpos[qids] = best.x
        error = residual(best.x)
        reports.append({"side": side, "active": preset["active"][i],
                        "position_error_mm": float(np.linalg.norm(error[:3]) * 1000),
                        "rotation_error_deg": float(np.rad2deg(np.linalg.norm(error[3:6]) / .16)),
                        "wrist_angles_deg": {suffix: float(np.rad2deg(best.x[4 + j]))
                                             for j, suffix in enumerate(("roll", "pitch", "yaw"))},
                        "joint_margin_deg": float(np.rad2deg(np.min(np.minimum(best.x - model.jnt_range[jids, 0],
                                                                               model.jnt_range[jids, 1] - best.x))))})
    mujoco.mj_forward(model, data)
    ctrl = data.qpos[joint_addresses(model, ARM_NAMES + HAND_NAMES)].copy()
    return data, ctrl, reports


def settle(model, data, target, seconds=4., approach_ctrl=None, approach_seconds=0.):
    """Close from open fingers under finite motors, then remove setup fixture.

    A wheel-centering fixture is removed 1.5 seconds after the approach phase.
    It is absent from the exported scene/viewer. Finger and thumb drives remain
    mechanically coupled during the entire approach and closure.
    """
    armv = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES])
    wheel_v = model.jnt_dofadr[model.joint("wheel_joint").id]
    wheel_q = model.jnt_qposadr[model.joint("wheel_joint").id]
    start = target.copy()
    if approach_ctrl is not None:
        start[:14] = approach_ctrl[:14]
    start[14:] = [.05, .02, .10, .10, .10, .10] * 2
    for short in ("L", "R"):
        coupled_hand(model, data, short, False)
    mujoco.mj_forward(model, data)
    initial_penetration = max([0.] + [-c.dist for c in data.contact])
    preparation_penetration = initial_penetration
    preparation_nonrim_force = 0.
    preparation_nonrim_peak = None
    for i in range(int(seconds / model.opt.timestep)):
        t = i * model.opt.timestep
        # Preshape the fingers while still outside the rim. Closing only after
        # the wrist arrived made the open ring finger sweep over the paddle.
        preshape_seconds = .6 if approach_seconds else 1.2
        alpha = np.clip(t / preshape_seconds, 0., 1.)
        data.ctrl[:] = (1 - alpha) * start + alpha * target
        if approach_seconds:
            arm_alpha = min(1., t / approach_seconds)
            arm_alpha = arm_alpha * arm_alpha * (3 - 2 * arm_alpha)
            data.ctrl[:14] = (1 - arm_alpha) * start[:14] + arm_alpha * target[:14]
        # Establish four-finger support before bringing the thumb into contact.
        thumb_alpha = np.clip((t - approach_seconds - .7) / 1.2, 0., 1.)
        for index in (14, 15, 20, 21):
            data.ctrl[index] = (1 - thumb_alpha) * start[index] + thumb_alpha * target[index]
        data.qfrc_applied[:] = 0
        data.qfrc_applied[armv] = data.qfrc_bias[armv]
        if t < approach_seconds + 1.5:
            data.qfrc_applied[wheel_v] = -8. * data.qpos[wheel_q] - .4 * data.qvel[wheel_v]
        mujoco.mj_step(model, data)
        preparation_penetration = max(preparation_penetration, max([0.] + [-c.dist for c in data.contact]))
        if i % 10 == 0:
            sample = contact_distribution(model, data)
            peak = max(p["non_rim_normal_force_N"] for p in sample.values())
            if peak > preparation_nonrim_force:
                preparation_nonrim_force = peak
                preparation_nonrim_peak = {"time_seconds": t, "contacts": [point for p in sample.values()
                    for point in p["points"] if point["wheel_part"] != "rim"]}
    data.qfrc_applied[:] = 0
    data.qfrc_applied[armv] = data.qfrc_bias[armv]
    mujoco.mj_forward(model, data)
    contacts = {"left": 0, "right": 0}
    max_penetration = 0.
    for contact in data.contact:
        a, b = model.geom(contact.geom1).name, model.geom(contact.geom2).name
        if not (a.startswith("collision_") or b.startswith("collision_")):
            continue
        hand = b if a.startswith("collision_") else a
        side = "left" if hand.startswith(("L_", "left_")) else "right"
        contacts[side] += 1
        max_penetration = max(max_penetration, -contact.dist)
    distribution = contact_distribution(model, data)
    stats = {"preparation_seconds": seconds, "approach_seconds": approach_seconds,
             "setup_fixture_removed_at_seconds": approach_seconds + 1.5,
             "wheel_angle_deg": float(np.rad2deg(data.qpos[wheel_q])),
             "initial_penetration_mm": float(1000 * initial_penetration),
             "preparation_max_penetration_mm": float(1000 * preparation_penetration),
             "preparation_max_nonrim_normal_force_N_sampled_at_50Hz": preparation_nonrim_force,
             "preparation_peak_nonrim_contact": preparation_nonrim_peak,
             "wheel_contacts": contacts, "rim_contact_fingers": {k: p["rim_fingers"] for k, p in distribution.items()},
             "max_wheel_penetration_mm": float(max_penetration * 1000),
             "max_joint_speed_rad_s": float(np.max(np.abs(data.qvel))),
             "loaded_contact_distribution": distribution,
             "max_mimic_error_deg": float(np.rad2deg(max(abs(v) for v in mimic_errors(model, data).values())))}
    data.qfrc_applied[:] = 0
    return stats


def generate():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    poses = {}
    for name in PRESETS:
        data, ctrl, reports = solve_pose(model, name)
        approach = PRESETS[name].get("approach_seconds", 0.)
        prectrl = None
        if approach:
            pregrasp = copy.deepcopy(PRESETS[name])
            pregrasp["grip_pose"]["position_m"][1] += .05
            pregrasp["hand_drives"] = [.05, .02, .10, .10, .10, .10]
            data, prectrl, _ = solve_pose(model, name, pregrasp)
        settled = settle(model, data, ctrl, seconds=6. if approach else 4.,
                         approach_ctrl=prectrl, approach_seconds=approach)
        poses[name] = {**PRESETS[name], "qpos": data.qpos.tolist(), "ctrl": ctrl.tolist(), "ik": reports,
                       "settling": settled}
    PRESET_FILE.write_text(json.dumps(poses, indent=2) + "\n")
    tree = ET.parse(SCENE)
    root = tree.getroot()
    if root.find("keyframe") is not None:
        root.remove(root.find("keyframe"))
    keys = ET.SubElement(root, "keyframe")
    for name, pose in poses.items():
        ET.SubElement(keys, "key", name=name, qpos=vec(pose["qpos"]), ctrl=vec(pose["ctrl"]))
    ET.indent(tree, space="  ")
    tree.write(SCENE, encoding="utf-8", xml_declaration=True)
    return poses


def apply(model, data, name, poses=None):
    if poses is None:
        poses = json.loads(PRESET_FILE.read_text())
    mujoco.mj_resetData(model, data)
    data.qpos[:] = poses[name]["qpos"]
    data.ctrl[:] = poses[name]["ctrl"]
    mujoco.mj_forward(model, data)
    armv = [model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES]
    data.qfrc_applied[armv] = data.qfrc_bias[armv]
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    result = generate()
    print(json.dumps({n: {"ik": p["ik"], "settling": p["settling"]} for n, p in result.items()}, indent=2))
