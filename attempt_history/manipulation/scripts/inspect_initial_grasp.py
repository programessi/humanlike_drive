"""Inspect the v2 physical grasp after initialization, before any PPO action."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wheel_manipulation.config import DEFAULT_CONFIG, load_config
from wheel_manipulation.env import ContactResidualEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/initial_grasp_contacts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(DEFAULT_CONFIG, "grasp")
    env = ContactResidualEnv(config, render_mode="rgb_array", camera="hands")
    try:
        _, initial = env.reset(seed=args.seed)
        center, axes = env.controller.center, env.controller.axis_frame
        contacts = []
        counts = {"left": 0, "right": 0, "other": 0}
        for i in range(env.data.ncon):
            con = env.data.contact[i]
            if (con.geom1 in env.wheel_geoms) == (con.geom2 in env.wheel_geoms):
                continue
            wheel = con.geom1 if con.geom1 in env.wheel_geoms else con.geom2
            robot = con.geom2 if con.geom1 in env.wheel_geoms else con.geom1
            body_id = int(env.model.geom_bodyid[robot])
            side = next((s for s, bodies in zip(["left", "right"], env.hand_body_groups)
                         if body_id in bodies), "other")
            counts[side] += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, i, force)
            local = axes.T @ (con.pos - center)
            body_rot = env.data.xmat[body_id].reshape(3, 3)
            body_local = body_rot.T @ (con.pos - env.data.xpos[body_id])
            contacts.append({
                "id": f"{ {'left': 'L', 'right': 'R', 'other': 'X'}[side]}{counts[side]:02}",
                "mujoco_contact_index": i, "side": side,
                "robot_body": env.model.body(body_id).name,
                "robot_geom_id": int(robot), "wheel_geom": env.model.geom(wheel).name,
                "world_position_m": con.pos.tolist(), "wheel_axis_position_mm": (local * 1000).tolist(),
                "robot_body_position_mm": (body_local * 1000).tolist(),
                "normal_force_N": float(force[0]), "tangent_force_N": float(np.linalg.norm(force[1:3])),
                "force_magnitude_N": float(np.linalg.norm(force[:3])),
                "distance_mm": float(con.dist * 1000), "constraint_active": bool(con.efc_address >= 0),
                "load_bearing_above_0_01N": bool(con.efc_address >= 0 and force[0] > .01),
            })
        groups = defaultdict(lambda: {"contact_count": 0, "sum_contact_force_magnitudes_N": 0., "ids": []})
        for con in contacts:
            group = groups[con["robot_body"]]
            group["contact_count"] += 1
            group["sum_contact_force_magnitudes_N"] += con["force_magnitude_N"]
            group["ids"].append(con["id"])
        report = {
            "state": "After 1.2 s deterministic grasp initialization, before first PPO action",
            "seed": args.seed, "task": asdict(config), "initial_info": initial,
            "scene_sha256": hashlib.sha256(env.scene_path.read_bytes()).hexdigest(),
            "wheel_angle_deg": float(np.rad2deg(env.wheel_angle)),
            "robot_wheel_contacts": contacts, "groups": dict(groups),
            "contacts_per_side": counts,
            "force_definition": "Sum of individual contact force magnitudes, not a net resultant or measured hand force",
            "notes": ["Multiple adjacent rim capsules can contact the same robot link",
                      "Contact points are MuJoCo mesh/capsule contacts; pad reward sites do not define them",
                      "One nominal initial-state snapshot, not persistent contacts during steering or real tactile measurements"],
        }
        (args.output / "contacts.json").write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / "contacts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(contacts[0]))
            writer.writeheader()
            writer.writerows(contacts)
        np.savez(args.output / "physical_state.npz", qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                 ctrl=env.data.ctrl.copy(), eq_active=env.data.eq_active.copy())

        import imageio.v2 as imageio
        imageio.imwrite(args.output / "initial_grasp.png", env.render())
        colors = {"left": np.array([1., .25, .05, 1.]), "right": np.array([.0, .8, 1., 1.]),
                  "other": np.array([1., 1., 0., 1.])}
        # Isolate hands and wheel for the inspection views. This only changes
        # rendering after recording the physical state; no physics is advanced.
        visible_bodies = set.union(*env.hand_body_groups) | {env.wheel_body}
        for geom_id, body_id in enumerate(env.model.geom_bodyid):
            if int(body_id) not in visible_bodies:
                env.model.geom_rgba[geom_id, 3] = 0
        for name, azimuth in [("contacts_front", 0), ("contacts_driver", 180)]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = center
            camera.distance = .65
            camera.azimuth, camera.elevation = azimuth, -8
            env.renderer.update_scene(env.data, camera=camera, scene_option=env.render_options)
            for con in contacts:
                scene = env.renderer.scene
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .003),
                                   np.array(con["world_position_m"]), np.eye(3).ravel(), colors[con["side"]])
                geom.matid = -1
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                scene.ngeom += 1
            rgb = env.renderer.render()
            imageio.imwrite(args.output / f"{name}.png", rgb)
            # Also show projected markers without depth occlusion so points on
            # hidden contact surfaces remain inspectable in the static view.
            from PIL import Image, ImageDraw
            annotated = Image.fromarray(rgb)
            draw = ImageDraw.Draw(annotated)
            cameras = env.renderer.scene.camera
            pos = np.mean([c.pos for c in cameras], axis=0)
            forward = np.mean([c.forward for c in cameras], axis=0)
            forward /= np.linalg.norm(forward)
            up = np.mean([c.up for c in cameras], axis=0)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            glcam = cameras[0]
            focal = env.height * glcam.frustum_near / (glcam.frustum_top - glcam.frustum_bottom)
            for con in contacts:
                delta = np.array(con["world_position_m"]) - pos
                depth = float(delta @ forward)
                x = env.width / 2 + focal * float(delta @ right) / depth
                y = env.height / 2 - focal * float(delta @ up) / depth
                color = tuple((255 * colors[con["side"]][:3]).astype(int))
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white", width=1)
            draw.text((15, 15), "Projected contacts (including hidden surfaces): orange left, cyan right", fill="white")
            annotated.save(args.output / f"{name}_marked.png")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        angle = np.linspace(0, 2 * np.pi, 361)
        ax.plot(140 * np.cos(angle), 140 * np.sin(angle), color="black", linewidth=9, alpha=.15)
        ax.plot(140 * np.cos(angle), 140 * np.sin(angle), color="black", linewidth=1)
        # Show the opposite sides separately so closely spaced seam contacts can
        # be identified by their IDs and matched to the CSV.
        for side, sign in [("left", 1), ("right", -1)]:
            points = [c for c in contacts if c["side"] == side]
            for c in points:
                xyz = c["wheel_axis_position_mm"]
                ax.scatter(xyz[1], xyz[2], color=colors[side][:3],
                           s=30 + 20 * c["force_magnitude_N"], edgecolors="black", linewidths=.4)
            ordered = sorted(points, key=lambda c: c["wheel_axis_position_mm"][2], reverse=True)
            labels_y = np.linspace(100, -110, len(ordered))
            for c, label_y in zip(ordered, labels_y):
                xyz = c["wheel_axis_position_mm"]
                body = c["robot_body"][2:]
                ax.annotate(f"{c['id']} {body}", xy=(xyz[1], xyz[2]), xytext=(sign * 175, label_y),
                            ha="left" if sign > 0 else "right", va="center", fontsize=8,
                            arrowprops=dict(arrowstyle="-", color=colors[side][:3], linewidth=.7))
        ax.set_xlim(-350, 350)
        ax.set_ylim(-165, 170)
        ax.set_aspect("equal")
        ax.set_xlabel("Wheel-axis y (mm); robot left is positive")
        ax.set_ylabel("Wheel-axis z (mm)")
        ax.set_title("Initial physical grasp contacts before PPO: orange = left, cyan = right\n"
                     "Axis-plane projection; marker size indicates individual simulated contact load")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(args.output / "contact_map.png", dpi=170)
        plt.close(fig)
        print(json.dumps({"contacts_per_side": counts, "groups": dict(groups),
                          "wheel_angle_deg": report["wheel_angle_deg"]}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
