"""Select a grasp by non-rim clearance throughout opening, then physical hold.

Kinematic screening is not dynamic proof. Only saved finite-motor trials
can establish that a candidate actually grasps the rim.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from humanlike_drive.controller.basic_grasps import PRESETS, solve_pose, joint_addresses, coupled_hand
from humanlike_drive.controller.grasp_contacts import contact_distribution
from humanlike_drive.env.inspire_hand import apply_mimic, effective_joint_ranges
from humanlike_drive.env.logitech_scene import ARM_NAMES, HAND_NAMES
from humanlike_drive.env.p0_scene import build_scene
from humanlike_drive.env_runner.audit_scene import wrist_posture

OPEN_HAND = np.array([.05, .02, .10, .10, .10, .10])


class Clearance:
    def __init__(self, model):
        self.model = model
        names = [model.geom(i).name or "" for i in range(model.ngeom)]
        self.hands = [i for i,n in enumerate(names) if model.geom_contype[i] == 2 and
                      (n.startswith(("L_", "R_")) or "wrist_yaw_link" in n)]
        self.paddles = [i for i,n in enumerate(names) if n.startswith("collision_paddle_")]
        self.spokes = [i for i,n in enumerate(names) if n.startswith("collision_") and
                       not n.startswith(("collision_rim_", "collision_paddle_"))]

    def distance(self, data, wheel_geoms):
        return min((mujoco.mj_geomDistance(self.model, data, a, b, .10, None)
                    for a in self.hands for b in wheel_geoms), default=.1)

    def opening_sweep(self, data, samples=11):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = data.qpos
        handq = joint_addresses(self.model, HAND_NAMES)
        closed = data.qpos[handq].copy()
        paddle, spoke = [], []
        for alpha in np.linspace(0., 1., samples):
            scratch.qpos[handq] = (1-alpha)*closed+alpha*np.tile(OPEN_HAND, 2)
            for short in ("L", "R"):
                apply_mimic(self.model, scratch, short)
            mujoco.mj_kinematics(self.model, scratch)
            paddle.append(self.distance(scratch, self.paddles))
            spoke.append(self.distance(scratch, self.spokes))
        return {"samples": samples, "closed_paddle_clearance_mm": 1000*paddle[0],
                "opening_min_paddle_clearance_mm": 1000*min(paddle),
                "opening_min_spoke_clearance_mm": 1000*min(spoke),
                "scope": "frozen wrist opening sweep, no wrist retreat, kinematic screening only"}


def candidate(angle_deg, family="enveloping", axial_mm=0., radial_mm=0., blend=1.):
    p = copy.deepcopy(PRESETS["ten_two"])
    p["angles"] = [angle_deg, -angle_deg]
    if family == "aligned":
        relaxed = PRESETS["nine_three"]
        start, finish = p["grip_pose"], relaxed["grip_pose"]
        p["grip_pose"] = {
            "position_m": ((1-blend)*np.array(start["position_m"])+blend*np.array(finish["position_m"])).tolist(),
            "rotation_matrix": Slerp([0.,1.], Rotation.from_matrix([
                start["rotation_matrix"], finish["rotation_matrix"]]))([blend]).as_matrix()[0].tolist()}
        p["hand_drives"] = ((1-blend)*np.array(p["hand_drives"])+blend*np.array(relaxed["hand_drives"])).tolist()
        p["arm_posture"]["wrist_yaw_target_deg"] = 25.-15.*blend
    p["shape_blend"] = blend if family == "aligned" else 0.
    p["grip_pose"]["position_m"][0] += axial_mm/1000
    p["grip_pose"]["position_m"][1] += radial_mm/1000
    p["label"] = f"P0 clear grasp: {family}, rim elevation {angle_deg} deg"
    return p


def screen(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(build_scene(output/"scene.xml")))
    clearance = Clearance(model)
    records = []
    for family in ("enveloping", "aligned"):
        for angle in (30., 40., 45., 50.):
            p = candidate(angle, family)
            data, ctrl, ik = solve_pose(model, "ten_two", p)
            geometry = clearance.opening_sweep(data)
            record = {"id": f"{family}_{angle:g}", "preset": p, "ik": ik,
                      "wrist": wrist_posture(model, data), **geometry,
                      "qpos": data.qpos.tolist(), "ctrl": ctrl.tolist()}
            records.append(record)
            print(record["id"], geometry, "wrist", record["wrist"], flush=True)
            (output/"screening.json").write_text(json.dumps(records, indent=2)+"\n")


def prepare(output, preset):
    """Finite-motor preshape/approach/closure, fixture removed before export."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"grasp.json").exists():
        raise FileExistsError(output/"grasp.json")
    model = mujoco.MjModel.from_xml_path(str(build_scene(output/"scene.xml")))
    _, target, ik = solve_pose(model, "ten_two", preset)
    for i, side in enumerate(("L", "R")):
        if side in preset.get("hand_drives_by_side", {}):
            target[14+6*i:20+6*i] = preset["hand_drives_by_side"][side]
    ranges = effective_joint_ranges(model)[[model.joint(n).id for n in ARM_NAMES+HAND_NAMES]]
    if not np.all((target >= ranges[:,0]) & (target <= ranges[:,1])):
        raise ValueError("Candidate targets exceed actual mimic-coupled joint ranges")
    pre = copy.deepcopy(preset)
    pre["grip_pose"]["position_m"][1] += .05
    pre["hand_drives"] = OPEN_HAND.tolist()
    data, approach, _ = solve_pose(model, "ten_two", pre)
    for side in ("L", "R"):
        coupled_hand(model, data, side, False)
    mujoco.mj_forward(model, data)
    armv = [model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES]
    wheelv = model.jnt_dofadr[model.joint("wheel_joint").id]
    wheelq = model.jnt_qposadr[model.joint("wheel_joint").id]
    rows, nonrim_peak, penetration = [], 0., 0.
    robot = model.geom_contype == 2
    illegal_peak = 0.
    force = np.zeros(6)
    for step in range(round(8./model.opt.timestep)):
        t = step*model.opt.timestep
        alpha = np.clip(t/.6, 0., 1.)
        data.ctrl[:] = (1-alpha)*approach+alpha*target
        u = np.clip(t/1., 0., 1.); u = u*u*(3-2*u)
        data.ctrl[:14] = (1-u)*approach[:14]+u*target[:14]
        thumb = np.clip((t-1.7)/1.2, 0., 1.)
        for j in (14,15,20,21):
            data.ctrl[j] = (1-thumb)*approach[j]+thumb*target[j]
        data.qfrc_applied[:] = 0.
        data.qfrc_applied[armv] = data.qfrc_bias[armv]
        if t < 2.5:
            data.qfrc_applied[wheelv] = -8.*data.qpos[wheelq]-.4*data.qvel[wheelv]
        mujoco.mj_forward(model, data)
        for k,c in enumerate(data.contact):
            mujoco.mj_contactForce(model, data, k, force)
            a,b = model.geom(c.geom1).name, model.geom(c.geom2).name
            wheel = next((n for n in (a,b) if n.startswith("collision_")), "")
            if wheel.startswith("collision_rim_"):
                penetration = max(penetration, -c.dist)
            elif wheel:
                nonrim_peak = max(nonrim_peak, force[0])
            elif robot[c.geom1] and robot[c.geom2]:
                illegal_peak = max(illegal_peak, force[0])
        rows.append((t, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.qfrc_applied.copy()))
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    geometry = Clearance(model).opening_sweep(data, samples=31)
    distribution = contact_distribution(model, data)
    report = {"preset": preset, "ik": ik, "wrist": wrist_posture(model, data),
              "preparation_seconds": 8., "fixture_removed_s": 2.5,
              "preparation_max_nonrim_contact_N": float(nonrim_peak),
              "preparation_max_illegal_self_contact_N": float(illegal_peak),
              "preparation_max_rim_penetration_mm": float(penetration*1000),
              "wheel_angle_deg": float(np.rad2deg(data.qpos[wheelq])),
              "distribution": distribution, "geometry": geometry,
              "qpos": data.qpos.tolist(), "ctrl": target.tolist()}
    (output/"grasp.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(output/"preparation.npz", time=[r[0] for r in rows],
        qpos=[r[1] for r in rows], qvel=[r[2] for r in rows], ctrl=[r[3] for r in rows],
        qfrc_applied=[r[4] for r in rows])
    print(json.dumps({k:v for k,v in report.items() if k not in ("qpos", "ctrl", "distribution", "preset")}, indent=2), flush=True)
    print({s: {"fingers": d["rim_fingers"], "palm": d["palm_rim_contact"], "thumb": d["thumb_rim_contact"]}
           for s,d in distribution.items()}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--angle", type=float, default=30.)
    parser.add_argument("--axial-mm", type=float, default=0.)
    parser.add_argument("--radial-mm", type=float, default=0.)
    parser.add_argument("--blend", type=float, default=1.)
    parser.add_argument("--preset-json", type=Path)
    args = parser.parse_args()
    if args.prepare:
        p = (json.loads(args.preset_json.read_text()) if args.preset_json else
             candidate(args.angle, "aligned", args.axial_mm, args.radial_mm, args.blend))
        prepare(args.output, p)
    else:
        screen(args.output)
