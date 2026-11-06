"""Contact distribution for grasp inspection, including actual normal forces."""
from __future__ import annotations

import mujoco
import numpy as np


def contact_distribution(model, data):
    frame = data.site("wheel_frame").xmat.reshape(3, 3)
    center = data.site("wheel_frame").xpos
    result = {side: {"loaded_links": [], "proximal_fingers": [], "distal_fingers": [],
                     "palm_contact": False, "thumb_contact": False,
                     "palm_rim_contact": False, "thumb_rim_contact": False,
                     "rim_fingers": [], "non_rim_fingers": [],
                     "rim_force_by_finger_N": {f: 0. for f in ("index", "middle", "ring", "pinky")},
                     "non_rim_normal_force_N": 0., "points": []}
              for side in ("left", "right")}
    normal_angles = {side: [] for side in result}
    for index, c in enumerate(data.contact):
        a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
        if not (a.startswith("collision_") or b.startswith("collision_")):
            continue
        hand = b if a.startswith("collision_") else a
        if hand.startswith(("L_", "left_")):
            side = "left"
        elif hand.startswith(("R_", "right_")):
            side = "right"
        else:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] < .05:
            continue
        p = result[side]
        p["loaded_links"].append(hand)
        p["thumb_contact"] |= "thumb" in hand
        p["palm_contact"] |= "wrist_yaw_link" in hand
        rim_contact = a.startswith("collision_rim_") or b.startswith("collision_rim_")
        wheel_geom = a if a.startswith("collision_") else b
        # Old scenes called the rear paddle sectors inner_2. Do not count
        # these as rim support when inspecting historical saved grasps.
        wheel_part = ("rim" if rim_contact else "paddle" if wheel_geom.startswith(
            ("collision_paddle_", "collision_inner_2_")) else "spoke_or_hub")
        if not rim_contact:
            p["non_rim_normal_force_N"] += float(force[0])
        p["palm_rim_contact"] |= rim_contact and "wrist_yaw_link" in hand
        p["thumb_rim_contact"] |= rim_contact and "thumb" in hand
        for finger in ("index", "middle", "ring", "pinky"):
            if finger + "_" in hand:
                p["rim_fingers" if rim_contact else "non_rim_fingers"].append(finger)
                if rim_contact:
                    p["rim_force_by_finger_N"][finger] += float(force[0])
            if finger + "_proximal" in hand:
                p["proximal_fingers"].append(finger)
            if finger + "_intermediate" in hand:
                p["distal_fingers"].append(finger)
        p["points"].append({"link": hand, "wheel_geom": wheel_geom, "wheel_part": wheel_part,
                            "position_wheel_frame_m": (frame.T @ (c.pos - center)).tolist(),
                            "position_world_m": c.pos.tolist(),
                            "normal_force_N": float(force[0]), "distance_mm": float(1000 * c.dist)})
        if a.startswith("collision_rim_") or b.startswith("collision_rim_"):
            radial = c.pos - center
            radial -= frame[:, 0] * np.dot(radial, frame[:, 0])
            radial /= np.linalg.norm(radial)
            normal = c.frame[:3] * (1 if a.startswith("collision_") else -1)
            normal_angles[side].append(np.arctan2(np.dot(normal, frame[:, 0]), np.dot(normal, radial)) % (2*np.pi))
    for side, p in result.items():
        for key in ("loaded_links", "proximal_fingers", "distal_fingers", "rim_fingers", "non_rim_fingers"):
            p[key] = sorted(set(p[key]))
        angles = np.sort(normal_angles[side])
        p["rim_normal_span_deg"] = float(np.rad2deg(2*np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2*np.pi])))) if len(angles) else 0.
        p["scope"] = "loaded contacts >0.05 N; normal span is a geometric distribution measure, not force closure proof"
    return result
