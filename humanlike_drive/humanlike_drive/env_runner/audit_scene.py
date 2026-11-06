"""Geometry and initial-contact checks for the new CAD scene."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from humanlike_drive.env.logitech_scene import SCENE, PROJECT, ARM_NAMES, HAND_NAMES
from humanlike_drive.controller.basic_grasps import PRESETS, PRESET_FILE, TARGET_CONFIG, apply
from humanlike_drive.controller.grasp_contacts import contact_distribution
from humanlike_drive.env.inspire_hand import HAND_URDF, mimic_errors


def wrist_posture(model, data):
    result = {}
    for side in ("left", "right"):
        angles = {axis: float(np.rad2deg(data.qpos[model.jnt_qposadr[
            model.joint(f"{side}_wrist_{axis}_joint").id]])) for axis in ("roll", "pitch", "yaw")}
        forearm = data.body(f"{side}_wrist_roll_link").xmat.reshape(3, 3)[:, 0]
        hand = data.body(f"{side}_wrist_yaw_link").xmat.reshape(3, 3)[:, 0]
        result[side] = {"joint_angles_deg": angles,
                        "forearm_hand_angle_deg": float(np.rad2deg(np.arccos(np.clip(forearm @ hand, -1., 1.))))}
    return result


def wheel_surface_check(model, data, mesh):
    """Compare loaded rim points to the original triangle mesh, not hull labels."""
    points = [p for side in contact_distribution(model, data).values()
              for p in side["points"] if p["wheel_part"] == "rim"]
    rotor = model.body("g29_wheel").id
    rotation = data.xmat[rotor].reshape(3, 3)
    local = np.array([rotation.T @ (np.array(p["position_world_m"]) - data.xpos[rotor]) for p in points])
    if len(local):
        _, distances, _ = trimesh.proximity.closest_point_naive(mesh, local)
    else:
        distances = []
    hand_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 3
                  and (model.geom(i).name.startswith(("L_", "R_")) or "wrist_yaw_link" in model.geom(i).name)]
    paddle_geoms = [i for i in range(model.ngeom) if model.geom(i).name.startswith("collision_paddle_")]
    return {"rim_contacts_checked": len(points),
            "max_loaded_rim_point_to_original_CAD_mm": float(max(distances, default=float("inf")) * 1000),
            "minimum_hand_to_paddle_clearance_mm": float(1000 * min(
                mujoco.mj_geomDistance(model, data, a, b, .10, None) for a in hand_geoms for b in paddle_geoms)),
            "scope": "exact closest triangle distance at sampled loaded rim points; paddle clearance uses CAD convex sectors"}


def arm_clearance(model, data):
    """Independent static distances where self-contact response is disabled.

    Check the two arm chains against each other and forearms/hands against the
    torso/pelvis. Adjacent shoulder assembly and intrahand pairs are excluded.
    """
    def descendant(body, root):
        while body:
            if body == root:
                return True
            body = model.body_parentid[body]
        return False
    arms = {side: [i for i in range(model.ngeom) if model.geom_group[i] == 3
                   and descendant(model.geom_bodyid[i], model.body(side + "_shoulder_pitch_link").id)]
            for side in ("left", "right")}
    torso = [i for i in range(model.ngeom) if model.geom_group[i] == 3
             and model.body(model.geom_bodyid[i]).name in ("pelvis", "torso_link")]
    pairs = [(a, b) for a in arms["left"] for b in arms["right"]]
    pairs += [(a, b) for side in arms for a in arms[side] for b in torso
              if "shoulder" not in model.geom(a).name]
    distances = [(float(mujoco.mj_geomDistance(model, data, a, b, .15, None)), a, b) for a, b in pairs]
    closest, a, b = min(distances)
    return {"pairs_checked": len(pairs), "minimum_or_lower_bound_mm": min(.15, closest) * 1000,
            "closest_pair": [model.geom(a).name, model.geom(b).name] if closest < .15 else None,
            "overlaps_deeper_than_1mm": [[model.geom(a).name, model.geom(b).name]
                                         for distance, a, b in distances if distance < -.001],
            "scope": "static cross-arm and forearm/hand-to-torso convex CAD distances; not full self-collision validation"}


def hand_clearance(model, data):
    distances = []
    for short, side in (("L", "left"), ("R", "right")):
        fingers = {f: [model.geom(f"{short}_{f}_{segment}_geom_1").id for segment in ("proximal", "intermediate")]
                   for f in ("index", "middle", "ring", "pinky")}
        thumbs = [model.geom(f"{short}_thumb_{segment}_geom_1").id for segment in ("proximal", "intermediate", "distal")]
        palm = model.geom(side + "_wrist_yaw_link_geom_2").id
        pairs = [(a, b) for x, y in itertools.combinations(fingers, 2) for a in fingers[x] for b in fingers[y]]
        pairs += [(a, b) for a in thumbs for finger in fingers.values() for b in finger]
        pairs += [(a, palm) for a in thumbs[1:] + [ids[1] for ids in fingers.values()]]
        distances.extend((float(mujoco.mj_geomDistance(model, data, a, b, .02, None)), a, b) for a, b in pairs)
    closest, a, b = min(distances)
    return {"pairs_checked": len(distances), "minimum_or_lower_bound_mm": min(.02, closest) * 1000,
            "closest_pair": [model.geom(a).name, model.geom(b).name] if closest < .02 else None,
            "scope": "static distances between different fingers, thumb and fingers, and distal segments and palm; adjacent hinge assemblies excluded"}


def audit(output):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rotor = model.body("g29_wheel").id
    # Raycast just the original CAD vs just the compound collision meshes.
    model.geom_group[:] = 0
    model.geom_group[model.geom("wheel_visual").id] = 4
    collision_ids = [i for i in range(model.ngeom) if model.geom(i).name.startswith("collision_")]
    model.geom_group[collision_ids] = 5
    visual_mask = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
    collision_mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    center = data.site("wheel_frame").xpos.copy()
    frame = data.site("wheel_frame").xmat.reshape(3, 3).copy()
    geomid = np.zeros(1, dtype=np.int32)
    errors = []
    missing = 0
    visual_hits = 0
    for axial in [-.019, -.014, -.007, 0., .005]:
        for angle in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            radial = frame[:, 1] * np.cos(angle) + frame[:, 2] * np.sin(angle)
            start = center + axial * frame[:, 0] + .18 * radial
            a = mujoco.mj_ray(model, data, start, -radial, visual_mask, True, -1, geomid)
            b = mujoco.mj_ray(model, data, start, -radial, collision_mask, True, -1, geomid)
            if 0 <= a < .09:
                visual_hits += 1
                if b < 0 or b > .09:
                    missing += 1
                else:
                    errors.append(abs(a - b) * 1000)
    openings = []
    for side in (-1, 1):
        start = center + .075 * frame[:, 2] + side * .025 * frame[:, 1] - .15 * frame[:, 0]
        a = mujoco.mj_ray(model, data, start, frame[:, 0], visual_mask, True, -1, geomid)
        b = mujoco.mj_ray(model, data, start, frame[:, 0], collision_mask, True, -1, geomid)
        openings.append({"visual_hit": float(a), "collision_hit": float(b), "both_open": bool(a < 0 and b < 0)})
    report = {"scene": str(SCENE), "nq": model.nq, "nu": model.nu, "collision_parts": len(collision_ids),
              "rim_radial_raycast": {"rays": 1800, "visual_rim_hits": visual_hits, "missed_collision_hits": missing,
                                      "median_error_mm": float(np.median(errors)),
                                      "p95_error_mm": float(np.quantile(errors, .95)),
                                      "max_error_mm": float(np.max(errors)),
                                      "scope": "sampled outer rim silhouettes at five axial depths; not a complete mesh Hausdorff bound"},
              "upper_finger_openings": openings, "presets": {}}
    report["assembly"] = {"housing_support_normal_world": data.xmat[model.body("g29_housing").id].reshape(3, 3)[:, 2].tolist(),
                          "shaft_inclination_deg": float(np.rad2deg(np.arcsin(abs(frame[2, 0])))),
                          "wheel_actuated": bool(any(model.actuator_trnid[i, 0] == model.joint("wheel_joint").id
                                                     for i in range(model.nu)))}
    report["self_collision_response_enabled"] = False
    report["mimic_source"] = str(HAND_URDF)
    report["sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in [("scene", SCENE), ("grasp_targets", TARGET_CONFIG), ("hand_urdf", HAND_URDF)]}
    # Reload original groups before contact/physics measurements.
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    armv = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES])
    armq = np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM_NAMES])
    armids = [model.joint(n).id for n in ARM_NAMES]
    poses = json.loads(PRESET_FILE.read_text())
    visual_mesh = trimesh.load(PROJECT / "assets/objects/logitech_g29/meshes/Link1.STL", force="mesh")
    report["hand_driver_ranges_rad"] = {name: model.actuator(name).ctrlrange.tolist() for name in HAND_NAMES}
    for name in PRESETS:
        apply(model, data, name)
        clearance = arm_clearance(model, data)
        finger_clearance = hand_clearance(model, data)
        original = data.qpos.copy()
        initial_distribution = contact_distribution(model, data)
        initial_surface = wheel_surface_check(model, data, visual_mesh)
        initial_wrist = wrist_posture(model, data)
        max_wrist_yaw = {side: 0. for side in ("left", "right")}
        max_wrist_bend = {side: 0. for side in ("left", "right")}
        contact_min = {"left": 999, "right": 999}
        contact_max = {"left": 0, "right": 0}
        penetrating = 0.
        max_wheel_change = 0.
        wheelq = model.jnt_qposadr[model.joint("wheel_joint").id]
        forbidden = set()
        min_margin = float("inf")
        max_wheel_applied = 0.
        duration = 5.
        max_mimic_error = 0.
        wrapping = {side: {"palm": 0, "thumb": 0, "min_fingers": 5,
                           "finger_samples": {f: 0 for f in ("index", "middle", "ring", "pinky")},
                           "minimum_finger_force_N": {f: float("inf") for f in ("index", "middle", "ring", "pinky")},
                           "maximum_nonrim_normal_force_N": 0.} for side in ("left", "right")}
        samples = 0
        for step in range(round(duration / model.opt.timestep)):
            data.qfrc_applied[:] = 0
            data.qfrc_applied[armv] = data.qfrc_bias[armv]
            mujoco.mj_step(model, data)
            for side, wrist in wrist_posture(model, data).items():
                max_wrist_yaw[side] = max(max_wrist_yaw[side], abs(wrist["joint_angles_deg"]["yaw"]))
                max_wrist_bend[side] = max(max_wrist_bend[side], wrist["forearm_hand_angle_deg"])
            max_mimic_error = max(max_mimic_error, max(abs(v) for v in mimic_errors(model, data).values()))
            if step % 10 == 0:
                distribution = contact_distribution(model, data)
                for side, p in distribution.items():
                    wrapping[side]["palm"] += p["palm_rim_contact"]
                    wrapping[side]["thumb"] += p["thumb_rim_contact"]
                    wrapping[side]["min_fingers"] = min(wrapping[side]["min_fingers"], len(p["rim_fingers"]))
                    for finger in wrapping[side]["finger_samples"]:
                        wrapping[side]["finger_samples"][finger] += finger in p["rim_fingers"]
                        wrapping[side]["minimum_finger_force_N"][finger] = min(
                            wrapping[side]["minimum_finger_force_N"][finger], p["rim_force_by_finger_N"][finger])
                    wrapping[side]["maximum_nonrim_normal_force_N"] = max(
                        wrapping[side]["maximum_nonrim_normal_force_N"], p["non_rim_normal_force_N"])
                samples += 1
            margin = np.minimum(data.qpos[armq] - model.jnt_range[armids, 0],
                                model.jnt_range[armids, 1] - data.qpos[armq])
            min_margin = min(min_margin, float(np.min(margin)))
            max_wheel_applied = max(max_wheel_applied, abs(float(data.qfrc_applied[model.jnt_dofadr[model.joint("wheel_joint").id]])))
            counts = {"left": 0, "right": 0}
            for contact_index, c in enumerate(data.contact):
                a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
                penetrating = max(penetrating, -c.dist)
                if a.startswith("collision_") or b.startswith("collision_"):
                    wheel_geom = a if a.startswith("collision_") else b
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    if not wheel_geom.startswith("collision_rim_") and force[0] > .05:
                        forbidden.add((a, b))
                    hand = b if a.startswith("collision_") else a
                    if hand.startswith(("L_", "R_")) or "wrist_yaw" in hand:
                        counts["left" if hand.startswith(("L_", "left_")) else "right"] += 1
                    else:
                        forbidden.add((a, b))
                else:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    # Positive-distance contacts inside the margin may carry load.
                    if c.dist < -.0005 or force[0] > .05:
                        forbidden.add((a, b))
            for side in counts:
                contact_min[side] = min(contact_min[side], counts[side])
                contact_max[side] = max(contact_max[side], counts[side])
            max_wheel_change = max(max_wheel_change, abs(data.qpos[wheelq] - original[wheelq]))
        margin = np.minimum(data.qpos[armq] - model.jnt_range[armids, 0],
                            model.jnt_range[armids, 1] - data.qpos[armq])
        report["presets"][name] = {"physics_seconds_without_setup_fixture": duration,
                                    "preparation": {key: value for key, value in poses[name]["settling"].items()
                                                    if key in ("preparation_seconds", "approach_seconds", "setup_fixture_removed_at_seconds",
                                                               "initial_penetration_mm", "preparation_max_penetration_mm",
                                                               "preparation_max_nonrim_normal_force_N_sampled_at_50Hz")},
                                    "minimum_wheel_contacts": contact_min,
                                    "maximum_wheel_contacts": contact_max,
                                    "max_contact_penetration_mm": float(penetrating * 1000),
                                    "max_wheel_drift_deg": float(np.rad2deg(max_wheel_change)),
                                    "arm_joint_margin_deg_at_end": float(np.rad2deg(np.min(margin))),
                                    "minimum_arm_joint_margin_deg": float(np.rad2deg(min_margin)),
                                    "max_direct_wheel_applied_torque_Nm": max_wheel_applied,
                                    "max_mimic_error_deg": float(np.rad2deg(max_mimic_error)),
                                    "initial_contact_distribution": initial_distribution,
                                    "final_contact_distribution": contact_distribution(model, data),
                                    "wrist_posture_initial": initial_wrist,
                                    "wrist_posture_final": wrist_posture(model, data),
                                    "maximum_abs_wrist_yaw_deg": max_wrist_yaw,
                                    "maximum_forearm_hand_angle_deg": max_wrist_bend,
                                    "wheel_surface_initial": initial_surface,
                                    "wheel_surface_final": wheel_surface_check(model, data, visual_mesh),
                                    "wrapping_samples_at_50Hz": samples,
                                    "wrapping_contact": {side: {"palm_rim_fraction": p["palm"] / samples,
                                                                 "thumb_rim_fraction": p["thumb"] / samples,
                                                                 "minimum_loaded_rim_fingers": p["min_fingers"],
                                                                 "rim_fraction_by_finger": {f: n / samples for f, n in p["finger_samples"].items()},
                                                                 "minimum_rim_force_by_finger_N": p["minimum_finger_force_N"],
                                                                 "maximum_nonrim_normal_force_N": p["maximum_nonrim_normal_force_N"]}
                                                         for side, p in wrapping.items()},
                                    "static_arm_clearance": clearance,
                                    "static_hand_clearance_initial": finger_clearance,
                                    "static_hand_clearance_final": hand_clearance(model, data),
                                    "unexpected_collisions": sorted(forbidden),
                                    "finite_state": bool(np.all(np.isfinite(data.qpos)))}
    checks = {"housing_horizontal": abs(report["assembly"]["housing_support_normal_world"][2] - 1.) < 1e-6,
              "natural_inclined_shaft": 20 < report["assembly"]["shaft_inclination_deg"] < 30,
              "passive_wheel": not report["assembly"]["wheel_actuated"],
              "sampled_rim_fits": missing == 0 and np.max(errors) < 2. and np.quantile(errors, .95) < .5,
              "sampled_finger_openings_preserved": all(o["both_open"] for o in openings)}
    for name, p in report["presets"].items():
        posture = PRESETS[name].get("arm_posture", {})
        if "maximum_abs_yaw_deg" in posture:
            checks[name + "_driverlike_wrist"] = all(
                p["maximum_abs_wrist_yaw_deg"][side] <= posture["maximum_abs_yaw_deg"]
                and p["maximum_forearm_hand_angle_deg"][side] <= posture["maximum_forearm_hand_angle_deg"]
                for side, active in zip(("left", "right"), PRESETS[name]["active"]) if active)
        checks[name + "_actual_rim_and_paddle_clearance"] = all(
            p[key]["max_loaded_rim_point_to_original_CAD_mm"] < 1.
            and p[key]["minimum_hand_to_paddle_clearance_mm"] > 1.
            for key in ("wheel_surface_initial", "wheel_surface_final"))
        checks[name + "_nonpenetrating_start_and_closure"] = (p["preparation"]["initial_penetration_mm"] < .01
                                                                and p["preparation"]["preparation_max_penetration_mm"] < .5
                                                                and p["preparation"]["preparation_max_nonrim_normal_force_N_sampled_at_50Hz"] == 0.)
        checks[name + "_mimic_under_contact"] = p["max_mimic_error_deg"] < .3
        checks[name + "_wrapping_contact"] = all(
            p["wrapping_contact"][side]["palm_rim_fraction"] >= .95
            and p["wrapping_contact"][side]["thumb_rim_fraction"] >= .95
            and p["wrapping_contact"][side]["minimum_loaded_rim_fingers"] == 4
            and p["wrapping_contact"][side]["maximum_nonrim_normal_force_N"] == 0.
            for side, active in zip(("left", "right"), PRESETS[name]["active"]) if active)
        checks[name + "_initial_contact"] = all(p["minimum_wheel_contacts"][side] >= 1 if active
                                                 else p["maximum_wheel_contacts"][side] == 0
                                                 for side, active in zip(("left", "right"), PRESETS[name]["active"]))
        checks[name + "_physical_settling"] = (p["finite_state"] and p["minimum_arm_joint_margin_deg"] > 2.
                                                 and p["max_contact_penetration_mm"] < .5
                                                 and p["max_wheel_drift_deg"] < 1.
                                                 and p["max_direct_wheel_applied_torque_Nm"] == 0.
                                                 and not p["unexpected_collisions"]
                                                 and p["static_hand_clearance_initial"]["minimum_or_lower_bound_mm"] > 0.
                                                 and p["static_hand_clearance_final"]["minimum_or_lower_bound_mm"] > 0.
                                                 and not p["static_arm_clearance"]["overlaps_deeper_than_1mm"])
    report["checks"] = {k: bool(v) for k, v in checks.items()}
    report["passed"] = all(checks.values())
    report["limits"] = "Initial-contact and geometry inspection only; no certified grasp wrench, trajectory tracking, or sim-to-real result."
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/logitech_scene_v1/audit.json")
    result = audit(parser.parse_args().output)
    raise SystemExit(0 if result["passed"] else 1)
