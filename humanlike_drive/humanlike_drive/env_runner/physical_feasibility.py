"""Layer 1: reproducible palm-driven engineering control, audit and replay.

Examples (from project root):
  bash scripts/run.sh -m humanlike_drive.env_runner.physical_feasibility prepare --output outputs/physical_feasibility/grasp_v1
  bash scripts/run.sh -m humanlike_drive.env_runner.physical_feasibility run --grasp outputs/physical_feasibility/grasp_v1 --angle 15 --output outputs/physical_feasibility/right15
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from humanlike_drive.controller.basic_grasps import PRESETS, solve_pose, joint_addresses
from humanlike_drive.controller.palm_contact import (
    ContactAudit,
    PARTS,
    required_wheel_contact,
)
from humanlike_drive.controller.palm_steering import PalmPointIK
from humanlike_drive.controller.p0_control import (
    RateExecutor,
    RoundTrip,
    smooth_fraction,
)
from humanlike_drive.env.logitech_scene import PROJECT, ARM_NAMES, HAND_NAMES, SCENE
from humanlike_drive.env.p0_scene import ASSEMBLY_EXCLUSIONS
from humanlike_drive.env.palm_collision import ASSETS, build_palm_scene as build_scene
from humanlike_drive.env.inspire_hand import mimic_relations, effective_joint_ranges

CONFIG = PROJECT / "humanlike_drive/config/controller/palm_primary_l1.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def freeze(output, config, scene):
    source = output / "source_snapshot"
    source.mkdir(exist_ok=True)
    hashes = {}
    for p in sorted((PROJECT / "humanlike_drive").rglob("*.py")) + [
        CONFIG,
        SCENE,
        scene,
    ]:
        relative = p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path(p.name)
        dst = source / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    assets = {}
    for asset in ET.parse(scene).getroot().findall("./asset/*"):
        if asset.get("file"):
            p = Path(asset.get("file"))
            assets[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT, capture_output=True, text=True
    )
    write_json(output / "config.json", config)
    write_json(
        output / "manifest.json",
        dict(
            git_commit=git.stdout.strip(),
            git_status=status.stdout,
            mujoco=mujoco.__version__,
            sha256=hashes,
            asset_sha256=assets,
            assembly_exclusions=ASSEMBLY_EXCLUSIONS,
            asset_scope="external mesh paths plus hashes; source snapshot is not a portable CAD bundle",
        ),
    )


class Recorder:
    def __init__(self, m):
        self.m = m
        self.audit = ContactAudit(m)
        self.rows = []
        self.raw = []
        self.events = []
        self.jids = np.array([m.joint(n).id for n in ARM_NAMES])
        self.qids = joint_addresses(m, ARM_NAMES)
        self.mimic = [
            (
                m.jnt_qposadr[m.joint(r["joint"]).id],
                m.jnt_qposadr[m.joint(r["parent"]).id],
                r["multiplier"],
                r["offset"],
            )
            for r in mimic_relations()
        ]
        scratch = mujoco.MjData(m)
        mujoco.mj_forward(m, scratch)
        self.center = scratch.site("wheel_frame").xpos.copy()
        self.frame = scratch.site("wheel_frame").xmat.reshape(3, 3).copy()

    def sample(self, d, ex, desired, ref, phase, active, inset, load, hand_phases):
        a = self.audit.sample(d, raw=True)
        k = len(self.rows)
        for point in a.pop("points"):
            if point[18] > 0.001:
                self.raw.append(np.r_[k, point])
        for pair in a.pop("bad"):
            if len(self.events) < 2000:
                self.events.append(dict(t=float(d.time), pair=pair))
        bend = []
        yaw = []
        surface = []
        for s in ("left", "right"):
            fa = d.body(s + "_wrist_roll_link").xmat.reshape(3, 3)[:, 0]
            ha = d.body(s + "_wrist_yaw_link").xmat.reshape(3, 3)[:, 0]
            bend.append(np.arccos(np.clip(fa @ ha, -1, 1)))
            yaw.append(float(d.joint(s + "_wrist_yaw_joint").qpos[0]))
            local = (d.site(s + "_grip").xpos - self.center) @ self.frame
            surface.append(np.arctan2(local[2], local[1]))
        q = d.qpos[self.qids]
        ranges = self.m.jnt_range[self.jids]
        required = required_wheel_contact(
            self.m, d, self.audit.wv, a["wheel_other_constraint"]
        )
        self.rows.append(
            dict(
                time=float(d.time),
                qpos=d.qpos.copy(),
                qvel=d.qvel.copy(),
                qacc=d.qacc.copy(),
                ctrl=d.ctrl.copy(),
                action=ex.action.copy(),
                previous_action=ex.previous_action.copy(),
                requested_ctrl=desired.copy(),
                executor_target=ex.target.copy(),
                qfrc_applied=d.qfrc_applied.copy(),
                actuator_force=d.actuator_force.copy(),
                qfrc_constraint=d.qfrc_constraint.copy(),
                qfrc_passive=d.qfrc_passive.copy(),
                qfrc_bias=d.qfrc_bias.copy(),
                reference=float(ref),
                wheel_angle=float(d.joint("wheel_joint").qpos[0]),
                wheel_velocity=float(d.joint("wheel_joint").qvel[0]),
                wheel_required_contact=float(required),
                wheel_dynamics_residual=float(required - a["part_torque"].sum()),
                load_torque=float(load),
                phase=phase,
                active_support=np.array(active, dtype=bool),
                inset=np.asarray(inset).copy(),
                hand_phases=np.asarray(hand_phases).copy(),
                grip_spatial_phase=np.array(surface),
                wrist_bend=np.array(bend),
                wrist_yaw=np.array(yaw),
                arm_margins=np.minimum(q - ranges[:, 0], ranges[:, 1] - q),
                mimic_error=max(
                    abs(d.qpos[x] - a * d.qpos[y] - b) for x, y, a, b in self.mimic
                ),
                **a,
            )
        )
        return self.rows[-1]

    def save(self, output):
        a = {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        np.savez_compressed(output / "trajectory.npz", **a)
        width = len(self.raw[0]) if self.raw else 31
        np.savez_compressed(
            output / "contacts.npz", contacts=np.asarray(self.raw).reshape(-1, width)
        )
        write_json(
            output / "contact_schema.json",
            dict(
                columns=[
                    "step",
                    "contact_index",
                    "geom1",
                    "geom2",
                    "side",
                    "part",
                    "distance",
                    "world_x",
                    "world_y",
                    "world_z",
                ]
                + [f"frame_{i}" for i in range(9)]
                + [
                    "Fn",
                    "Ft1",
                    "Ft2",
                    "Mn",
                    "Mt1",
                    "Mt2",
                    "palm_local_x",
                    "palm_local_y",
                    "palm_local_z",
                    "relative_slip_x",
                    "relative_slip_y",
                    "relative_slip_z",
                ],
                geom_names=self.audit.names,
                parts=PARTS,
                torque_components=["normal", "tangential_friction", "contact_couple"],
                force_frame="MuJoCo contact frame; wrench acts on geom2; world rows transformed with frame transpose",
                raw_threshold_N=0.001,
            ),
        )
        write_json(output / "collision_events.json", self.events)
        return a


def apply_forces(m, d, armv, wv, load):
    d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    # The sum of motor torque and model-based compensation obeys the original
    # per-drive effort range, including during blocked-arm contact.
    d.qfrc_applied[armv] = np.clip(
        d.qfrc_bias[armv],
        m.actuator_forcerange[:14, 0] - d.actuator_force[:14],
        m.actuator_forcerange[:14, 1] - d.actuator_force[:14],
    )
    d.qfrc_applied[wv] = load
    mujoco.mj_forward(m, d)


class Library:
    def __init__(self, m, d, cfg, phases=None):
        self.target_bias = np.asarray(
            cfg.get("target_calibration_bias_m", [[0.0, 0.0], [0.0, 0.0]])
        )
        self.ik = PalmPointIK(
            m,
            d,
            point_x=cfg["palm_point_x_m"],
            axial=cfg["palm_target_axial_m"],
            radius=cfg["palm_target_radius_m"],
        )
        self.records = []
        self.curves = []
        self.free = []
        for side in (0, 1):
            degrees = np.arange(-100, 101, 5.0) if phases is None else phases
            points = {}
            freed = {}
            seed = self.ik.seed[7 * side : 7 * side + 7]
            for direction in (1, -1):
                warm = seed.copy()
                wf = seed.copy()
                for deg in sorted(
                    [abs(v) for v in degrees if np.sign(v) in (0, direction)]
                ):
                    phase = np.deg2rad(direction * deg)
                    q, r = self.ik.solve(side, phase, warm=warm)
                    warm = q
                    points[phase] = q
                    self.records.append(r)
                    qf, rf = self.ik.solve(side, phase, warm=wf, radial_offset=0.07)
                    wf = qf
                    freed[phase] = qf
                    self.records.append(dict(kind="free_hand", **rf))
            keys = sorted(points)
            self.curves.append(
                PchipInterpolator(
                    keys, [points[p] for p in keys], axis=0, extrapolate=False
                )
            )
            self.free.append(
                PchipInterpolator(
                    keys, [freed[p] for p in keys], axis=0, extrapolate=False
                )
            )

    def arm(self, side, phase, retreat=0.0, inset=0.0, lead=0.0):
        q = self.curves[side](phase)
        if not np.isfinite(q).all():
            raise ValueError(
                "Requested hand phase outside independently scanned library"
            )
        if retreat < 1:
            q = self.ik.correction(
                side,
                q,
                phase,
                inset=inset - self.target_bias[side, 0],
                lead=lead + self.target_bias[side, 1] / self.ik.radius,
            )
        return (1 - retreat) * q + retreat * self.free[side](phase)


def setup(output, cfg, negative=False, friction=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trajectory.npz").exists():
        raise FileExistsError(output)
    scene = build_scene(output / "scene.xml", wheel_contacts=not negative)
    tree = ET.parse(scene)
    option = tree.getroot().find("option")
    for key, value in cfg.get("simulation_solver", {}).items():
        option.set(key, str(value))
    tree.write(scene, encoding="utf-8", xml_declaration=True)
    if friction is not None:
        tree = ET.parse(scene)
        for g in tree.getroot().iter("geom"):
            if g.get("contype") in ("1", "2"):
                g.set("friction", f"{friction} .01 .001")
        ET.indent(tree, space="  ")
        tree.write(scene, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(scene))
    if m.nu != 26 or m.joint("wheel_joint").id in m.actuator_trnid[:, 0]:
        raise ValueError("Layer 1 requires exactly 26 robot drives and a passive wheel")
    d = mujoco.MjData(m)
    armv = [m.jnt_dofadr[m.joint(n).id] for n in ARM_NAMES]
    wv = m.jnt_dofadr[m.joint("wheel_joint").id]
    return output, scene, m, d, armv, wv


def prepare(args, cfg):
    output, scene, m, d, armv, wv = setup(args.output, cfg)
    p = copy.deepcopy(PRESETS["nine_three"])
    p["hand_drives"] = cfg["hand_drives"]
    d, _, _ = solve_pose(m, "nine_three", p)
    ik = PalmPointIK(
        m,
        d,
        point_x=cfg["palm_point_x_m"],
        axial=cfg["palm_target_axial_m"],
        radius=cfg["palm_target_radius_m"],
    )
    target = []
    start = []
    reports = []
    for s in (0, 1):
        q, r = ik.solve(s, multi_start=True)
        target.extend(q)
        reports.append(r)
        qs, _ = ik.solve(s, warm=q, radial_offset=0.06)
        start.extend(qs)
    d.qpos[ik.qids] = start
    d.ctrl[:] = np.r_[start, cfg["hand_drives"] * 2]
    mujoco.mj_forward(m, d)
    ex = RateExecutor(m, d)
    rec = Recorder(m)
    inset = np.zeros(2)
    integral = 0.0
    last = None
    config = dict(
        cfg,
        command="prepare",
        duration_s=12.0,
        initialization="fresh collision-aware arm IK in clear space; prescribed compliant hand preshape; no wheel fixture",
        physics_dt=m.opt.timestep,
        control_dt=ex.dt,
        privileged_input="simulation palm normal force, joint/angle feedback",
        compensation="model bias; motor plus compensation capped at original effort limits",
    )
    freeze(output, config, scene)
    for k in range(round(12 / m.opt.timestep)):
        t = d.time
        if k % 25 == 0:
            u = smooth_fraction(t / 4)
            desired = np.r_[
                (1 - u) * np.array(start) + u * np.array(target), cfg["hand_drives"] * 2
            ]
            error = -float(d.joint("wheel_joint").qpos[0])
            integral = np.clip(
                integral + cfg["angle_ki_per_s"] * ex.dt * error, -0.08, 0.08
            )
            lead = np.clip(
                cfg["angle_kp"] * error + integral, -np.deg2rad(6), np.deg2rad(6)
            )
            if t > 4:
                for s in (0, 1):
                    force = last["rim_force"][s, 0] if last else 0
                    inset[s] = np.clip(
                        inset[s]
                        + cfg["pressure_admittance_m_per_N_s"]
                        * ex.dt
                        * (cfg["palm_force_target_N"] - force),
                        *cfg["pressure_inset_bounds_m"],
                    )
                    desired[7 * s : 7 * s + 7] = ik.correction(
                        s,
                        np.array(target[7 * s : 7 * s + 7]),
                        0,
                        inset=inset[s],
                        lead=lead,
                    )
            ramp_start = d.ctrl.copy()
            ramp_target = ex.follow(desired, d.qpos[ex.qids])
        fraction = (k % 25 + 1) / 25.0
        d.ctrl[:] = (1 - fraction) * ramp_start + fraction * ramp_target
        apply_forces(m, d, armv, wv, 0.0)
        last = rec.sample(
            d, ex, desired, 0.0, 0 if t < 6 else 4, [True, True], inset, 0.0, [0.0, 0.0]
        )
        mujoco.mj_step(m, d)
    a = rec.save(output)
    grasp = dict(
        config=cfg,
        qpos=d.qpos.tolist(),
        qvel=d.qvel.tolist(),
        ctrl=d.ctrl.tolist(),
        inset=inset.tolist(),
        nominal_q=target,
        ik=reports,
        preparation_source=str(output),
        no_wheel_fixture=True,
        palm_collision_sha256=hashlib.sha256(
            (ASSETS / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    write_json(output / "grasp.json", grasp)
    write_json(output / "grasp_used.json", grasp)
    result = evaluate(a, m, cfg, 12.0, prepare=True)
    write_json(output / "evaluation.json", result)
    plot(output, a)
    print(json.dumps(result, indent=2), flush=True)


def evaluate(a, m, cfg, duration, prepare=False, negative=False):
    scored = a["time"] >= (6.0 if prepare else 0.0)
    moving = np.isin(a["phase"], [1, 3])
    hold = scored & ~moving
    err = np.rad2deg(a["wheel_angle"] - a["reference"])
    armv = m.jnt_dofadr[[m.joint(n).id for n in ARM_NAMES]]
    totals = a["actuator_force"][:, :14] + a["qfrc_applied"][:, armv]
    jointq = a["qpos"][:, m.jnt_qposadr]
    limits = m.jnt_range
    limited = m.jnt_limited.astype(bool)
    violation = np.maximum(0, np.maximum(limits[:, 0] - jointq, jointq - limits[:, 1]))[
        :, limited
    ].max()
    checks = dict(
        complete=bool(a["time"][-1] >= duration - 0.003),
        finite=bool(all(np.isfinite(v).all() for v in a.values())),
        no_illegal_contact=bool(a["illegal"].max() <= 0.05),
        no_nonrim_contact=bool(a["nonrim"].max() <= 0.05),
        rim_penetration_lt_0p5mm=bool(a["penetration"].max() < 0.0005),
        arm_margin_gt_2deg=bool(a["arm_margins"].min() > np.deg2rad(2)),
        joint_limits=bool(violation < 1e-8),
        combined_effort_within_original=bool(
            np.all(totals >= m.actuator_forcerange[:14, 0] - 1e-8)
            and np.all(totals <= m.actuator_forcerange[:14, 1] + 1e-8)
        ),
        hand_effort_within_original=bool(
            np.max(np.abs(a["actuator_force"][:, 14:])) <= 0.35000001
        ),
        mimic_error_lt_0p1deg=bool(a["mimic_error"].max() < np.deg2rad(0.1)),
        torque_audit=bool(abs(a["torque_residual"]).max() < 1e-5),
    )
    checks["wheel_dynamics_balance"] = bool(
        abs(a["wheel_dynamics_residual"]).max()
        < cfg.get("max_wheel_dynamics_residual_Nm", 1e-5)
    )
    checks["wrist_bend"] = bool(
        a["wrist_bend"][scored].max()
        <= np.deg2rad(
            cfg["initial_wrist_bend_limit_deg"]
            if prepare
            else cfg["operating_wrist_bend_limit_deg"]
        )
    )
    rmse = float(np.sqrt(np.mean(err[moving] ** 2))) if moving.any() else None
    checks["motion_RMSE_le_2deg"] = rmse is None or rmse <= 2
    tail = a["time"] >= a["time"][-1] - 1
    checks["final_error_le_2deg"] = bool(abs(err[tail]).max() <= 2)
    stop_peaks = []
    for start in np.r_[0, np.flatnonzero(np.diff(a["phase"]) != 0) + 1]:
        if moving[start]:
            continue
        end = int(
            next(
                (
                    j
                    for j in range(start + 1, len(moving))
                    if a["phase"][j] != a["phase"][start]
                ),
                len(moving),
            )
        )
        if a["time"][end - 1] - a["time"][start] < 0.5:
            continue
        idx = np.arange(start, end)
        idx = idx[a["time"][idx] >= a["time"][end - 1] - 1.0]
        stop_peaks.append(float(abs(err[idx]).max()))
    checks["all_stop_tails_le_2deg"] = bool(stop_peaks and max(stop_peaks) <= 2)
    palm_occupancy = []
    max_gap = []
    slip_integrals = []
    for s in (0, 1):
        mask = scored & a["active_support"][:, s]
        valid = a["rim_force"][:, s, 0] > 0.05
        palm_occupancy.append(float(np.mean(valid[mask])) if mask.any() else None)
        longest = current = 0
        for on, v in zip(mask, valid):
            current = current + 1 if on and not v else 0
            longest = max(longest, current)
        max_gap.append(longest * m.opt.timestep)
        slip_integrals.append(float(np.sum(a["palm_slip"][mask, s]) * m.opt.timestep))
    checks["palm_contact_95pct"] = all(v is None or v >= 0.95 for v in palm_occupancy)
    checks["palm_contact_gap"] = max(max_gap) <= cfg["max_palm_contact_gap_s"]
    checks["palm_force_bounded"] = bool(
        a["rim_force"][scored, :, 0].max() <= cfg["palm_force_limit_N"]
    )
    checks["whole_hand_force_bounded"] = bool(
        a["rim_force"][scored].sum(axis=2).max()
        <= cfg.get("whole_hand_force_limit_N", 10.0)
    )
    if "palm_below_upper_deg" in a:
        checks["palm_upper_sector"] = bool(
            a["palm_below_upper_deg"][scored].max()
            <= cfg.get("upper_sector_endpoint_allowance_deg", 10.0)
        )
    checks["palm_slip_speed"] = bool(
        a["palm_slip"][scored].max() <= cfg["max_palm_slip_m_s"]
    )
    checks["palm_slip_distance"] = max(slip_integrals) <= cfg["max_cumulative_slip_m"]
    # Compare signed contributions to required wheel interaction on stable-sign,
    # nonzero-load segments. Normal torque and finger torque stay separate.
    total = a["part_torque"].sum(axis=(1, 2, 3))
    palm = a["part_torque"][:, :, 0, 1].sum(axis=1)
    mechanism = []
    mechanism_by_activity = []
    for sign in (-1, 1):
        mask = scored & (abs(a["load_torque"]) > 0.02) & (sign * total > 0.02)
        if mask.sum() >= 50:
            needed = float(np.sum(sign * total[mask]))
            share = float(np.sum(sign * palm[mask]) / needed)
            mechanism.append(
                dict(
                    required_torque_sign=sign,
                    seconds=float(mask.sum() * m.opt.timestep),
                    palm_friction_share=share,
                )
            )
            velocity = a["wheel_velocity"]
            for activity, selection in (
                ("holding", abs(velocity) <= np.deg2rad(0.2)),
                ("driving", (abs(velocity) > np.deg2rad(0.2)) & (total * velocity > 0)),
                ("braking", (abs(velocity) > np.deg2rad(0.2)) & (total * velocity < 0)),
            ):
                window = mask & selection
                if window.sum() >= 50:
                    mechanism_by_activity.append(
                        dict(
                            required_torque_sign=sign,
                            activity=activity,
                            seconds=float(window.sum() * m.opt.timestep),
                            palm_friction_share=float(
                                np.sum(sign * palm[window])
                                / np.sum(sign * total[window])
                            ),
                        )
                    )
    checks["palm_friction_primary"] = bool(
        prepare
        or (mechanism and all(x["palm_friction_share"] > 0.5 for x in mechanism))
    )
    checks["palm_friction_primary_by_activity"] = bool(
        prepare
        or (
            mechanism_by_activity
            and all(x["palm_friction_share"] > 0.5 for x in mechanism_by_activity)
        )
    )
    result = dict(
        passed=bool(all(checks.values()) and not negative),
        checks=checks,
        motion_RMSE_deg=rmse,
        final_error_peak_deg=float(abs(err[tail]).max()),
        max_error_deg=float(abs(err).max()),
        max_illegal_N=float(a["illegal"].max()),
        max_nonrim_N=float(a["nonrim"].max()),
        max_penetration_mm=float(a["penetration"].max() * 1000),
        min_arm_margin_deg=float(np.rad2deg(a["arm_margins"].min())),
        max_wrist_bend_deg=np.rad2deg(a["wrist_bend"][scored].max(axis=0)).tolist(),
        palm_contact_fraction=palm_occupancy,
        max_contact_gap_s=max_gap,
        palm_cumulative_slip_m=slip_integrals,
        max_palm_slip_m_s=float(a["palm_slip"][scored].max()),
        max_palm_force_N=float(a["rim_force"][scored, :, 0].max()),
        max_whole_hand_force_N=float(a["rim_force"][scored].sum(axis=2).max()),
        max_palm_below_upper_deg=(
            float(a["palm_below_upper_deg"][scored].max())
            if "palm_below_upper_deg" in a
            else None
        ),
        mechanism=mechanism,
        mechanism_by_activity=mechanism_by_activity,
        max_torque_audit_residual_Nm=float(abs(a["torque_residual"]).max()),
        max_wheel_dynamics_residual_Nm=float(abs(a["wheel_dynamics_residual"]).max()),
        stop_tail_peaks_deg=stop_peaks,
        max_joint_violation_deg=float(np.rad2deg(violation)),
        negative_control=negative,
        duration_s=float(a["time"][-1]),
    )
    return result


def plot(output, a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    t = a["time"]
    axs[0].plot(t, np.rad2deg(a["reference"]), "--", label="reference")
    axs[0].plot(t, np.rad2deg(a["wheel_angle"]), label="wheel")
    axs[0].set_ylabel("angle deg")
    axs[1].plot(t, np.rad2deg(a["wheel_angle"] - a["reference"]))
    axs[1].set_ylabel("error deg")
    for s, label in enumerate(("left", "right")):
        axs[2].plot(t, a["rim_force"][:, s, 0], label=label + " palm")
        axs[2].plot(
            t, a["rim_force"][:, s, 1:].sum(axis=1), ":", label=label + " fingers/thumb"
        )
        axs[3].plot(t, a["part_torque"][:, s, 0, 1], label=label + " palm friction")
        axs[4].plot(t, np.rad2deg(a["wrist_bend"][:, s]), label=label + " wrist")
    axs[2].set_ylabel("contact N")
    axs[3].plot(t, a["load_torque"], "k--", label="external load")
    axs[3].set_ylabel("axis torque Nm")
    axs[4].set_ylabel("wrist bend deg")
    for ax in axs:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=3, fontsize=8)
    axs[-1].set_xlabel("time s")
    fig.suptitle(output.name + " | engineering controller, uncalibrated simulation")
    fig.tight_layout()
    fig.savefig(output / "tracking.png", dpi=120)
    plt.close(fig)


def run(args, cfg):
    if not args.grasp:
        raise ValueError("--grasp is required")
    grasp_dir = Path(args.grasp).resolve()
    grasp = json.loads((grasp_dir / "grasp.json").read_text())
    collision_hash = hashlib.sha256((ASSETS / "manifest.json").read_bytes()).hexdigest()
    if grasp.get("palm_collision_sha256") != collision_hash:
        raise ValueError(
            "Grasp collision version differs; prepare a fresh Layer 1 grasp"
        )
    preparation = json.loads((grasp_dir / "evaluation.json").read_text())
    if not preparation.get("passed"):
        raise ValueError(
            "Preparation failed engineering checks; cannot start qualification"
        )
    for key in (
        "hand_drives",
        "palm_point_x_m",
        "palm_target_axial_m",
        "palm_target_radius_m",
        "simulation_solver",
    ):
        if cfg.get(key) != grasp["config"].get(key):
            raise ValueError(f"Grasp configuration differs at {key}; prepare again")
    if args.finger_closure is not None:
        cfg["hand_drives"][2:] = [args.finger_closure] * 4
    if args.target_offset_mm:
        cfg["target_calibration_bias_m"] = (
            np.random.default_rng(args.seed).uniform(
                -args.target_offset_mm, args.target_offset_mm, (2, 2)
            )
            / 1000
        ).tolist()
    output, scene, m, d, armv, wv = setup(
        args.output, cfg, args.negative_control, args.friction
    )
    if args.target_offset_mm:
        write_json(
            output / "target_perturbation.json",
            dict(
                seed=args.seed,
                radial_tangential_m=cfg["target_calibration_bias_m"],
                scope="Constant errors in commanded palm target registration; legal prepared physical state retained, no instantaneous qpos displacement",
            ),
        )
    d.qpos[:] = grasp["qpos"]
    d.qvel[:] = grasp["qvel"]
    d.ctrl[:] = grasp["ctrl"]
    mujoco.mj_forward(m, d)
    ex = RateExecutor(m, d)
    rec = Recorder(m)
    # Reference frame is a spatial zero frame, not the rotating wheel material.
    scratch = mujoco.MjData(m)
    mujoco.mj_forward(m, scratch)
    frame = scratch.site("wheel_frame").xmat.reshape(3, 3).copy()
    side = (
        (0 if args.angle >= 0 else 1)
        if args.side is None
        else ("left", "right").index(args.side)
    )
    other = 1 - side
    if args.offset_mm:
        rng = np.random.default_rng(args.seed)
        perturb = rng.uniform(-args.offset_mm, args.offset_mm, (2, 2)) / 1000
        ik_perturb = PalmPointIK(
            m,
            d,
            point_x=cfg["palm_point_x_m"],
            axial=cfg["palm_target_axial_m"],
            radius=cfg["palm_target_radius_m"],
        )
        for side_index in (0, 1):
            idx = slice(7 * side_index, 7 * side_index + 7)
            d.qpos[ik_perturb.qids[idx]] = ik_perturb.correction(
                side_index,
                d.qpos[ik_perturb.qids[idx]],
                0.0,
                inset=-perturb[side_index, 0],
                lead=perturb[side_index, 1] / ik_perturb.radius,
            )
        mujoco.mj_forward(m, d)
        write_json(
            output / "initial_perturbation.json",
            dict(
                seed=args.seed,
                radial_tangential_m=perturb.tolist(),
                scope="initial simulated hand positions only; no task-time resets",
            ),
        )
    if args.case in ("relay", "crosszero"):
        return run_relay(args, cfg, grasp, output, scene, m, d, armv, wv, ex, rec)
    plan = (
        RoundTrip(
            args.angle, speed_deg_s=cfg["speed_deg_s"], initial_hold_s=5.0, hold_s=3.0
        )
        if args.case != "hold"
        else None
    )
    duration = plan.duration if plan else 14.0
    if args.duration:
        duration = min(duration, args.duration)
    load = args.load if args.load is not None else -0.1 * (1 if args.angle >= 0 else -1)
    config = dict(
        cfg,
        command="run",
        arguments=vars(args),
        expected_duration_s=plan.duration if plan else 14.0,
        run_duration_s=duration,
        load_model=dict(
            constant_Nm=load,
            damping_Nm_s=cfg["load_damping_Nm_s"],
            original_joint_damping_and_dry_friction="retained",
        ),
        grasp_source=str(grasp_dir),
        grasp_sha256=hashlib.sha256(
            (grasp_dir / "grasp.json").read_bytes()
        ).hexdigest(),
        privileged_input="palm contact normal forces; no learned policy",
        action_interface="26 normalized position target rates at 20 Hz; original finite motors at 500 Hz",
    )
    freeze(output, config, scene)
    shutil.copy2(grasp_dir / "grasp.json", output / "grasp_used.json")
    span = max(5.0, np.ceil((abs(args.angle) + 7) / 5) * 5) if plan else 5.0
    library = Library(m, d, cfg, phases=np.arange(-span, span + 0.1, 5.0))
    # Library constructor uses current wheel material frame; replace it with the
    # canonical frozen frame (initial wheel offset is tiny, still not ignored).
    library.ik.frame = frame
    library.ik.axis = frame[:, 0]
    write_json(output / "reachability.json", library.records)
    inset = np.array(grasp["inset"])
    integral = 0.0
    last = None
    for k in range(round(duration / m.opt.timestep)):
        t = float(d.time)
        ref, phase, _ = plan.sample(t) if plan else (0.0, 2, 0.0)
        retreat = 0.0 if args.both_hands else smooth_fraction(t / 3.0)
        support = [args.both_hands, args.both_hands]
        support[side] = True
        if k % 25 == 0:
            err = ref - float(d.joint("wheel_joint").qpos[0])
            integral = np.clip(
                integral + cfg["angle_ki_per_s"] * ex.dt * err,
                -np.deg2rad(4),
                np.deg2rad(4),
            )
            lead = np.clip(
                cfg["angle_kp"] * err + integral,
                -np.deg2rad(cfg["angle_lead_limit_deg"]),
                np.deg2rad(cfg["angle_lead_limit_deg"]),
            )
            force = last["rim_force"][side, 0] if last else cfg["palm_force_target_N"]
            inset[side] = np.clip(
                inset[side]
                + cfg["pressure_admittance_m_per_N_s"]
                * ex.dt
                * (cfg["palm_force_target_N"] - force),
                *cfg["pressure_inset_bounds_m"],
            )
            desired = np.r_[np.zeros(14), cfg["hand_drives"] * 2]
            desired[7 * side : 7 * side + 7] = library.arm(
                side, ref, inset=inset[side], lead=lead
            )
            if args.both_hands:
                other_force = (
                    last["rim_force"][other, 0] if last else cfg["palm_force_target_N"]
                )
                inset[other] = np.clip(
                    inset[other]
                    + cfg["pressure_admittance_m_per_N_s"]
                    * ex.dt
                    * (cfg["palm_force_target_N"] - other_force),
                    *cfg["pressure_inset_bounds_m"],
                )
            desired[7 * other : 7 * other + 7] = library.arm(
                other,
                0.0,
                retreat=retreat,
                inset=inset[other] * (1 - retreat),
                lead=lead * (1 - retreat),
            )
            ramp_start = d.ctrl.copy()
            ramp_target = ex.follow(desired, d.qpos[ex.qids])
        fraction = (k % 25 + 1) / 25.0
        d.ctrl[:] = (1 - fraction) * ramp_start + fraction * ramp_target
        applied_load = load - cfg["load_damping_Nm_s"] * d.qvel[wv]
        apply_forces(m, d, armv, wv, applied_load)
        phases = [0.0, 0.0]
        phases[side] = ref
        last = rec.sample(
            d, ex, desired, ref, phase, support, inset, applied_load, phases
        )
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or abs(d.qvel).max() > 100:
            break
        if k % 5000 == 0:
            print(
                f't={t:.1f} ref={np.rad2deg(ref):.1f} wheel={np.rad2deg(d.joint("wheel_joint").qpos[0]):.2f} palm={last["rim_force"][side,0]:.2f}',
                flush=True,
            )
    arrays = rec.save(output)
    result = evaluate(
        arrays, m, cfg, config["expected_duration_s"], negative=args.negative_control
    )
    clear = arrays["time"] >= 3.5
    result["checks"]["inactive_hand_clear"] = args.both_hands or (
        bool(np.max(arrays["rim_force"][clear, other].sum(axis=1)) <= 0.05)
        if clear.any()
        else False
    )
    result["passed"] = bool(
        all(result["checks"].values()) and not args.negative_control
    )
    write_json(output / "evaluation.json", result)
    plot(output, arrays)
    print(json.dumps(result, indent=2), flush=True)


def run_relay(args, cfg, grasp, output, scene, m, d, armv, wv, ex, rec):
    from humanlike_drive.controller.palm_relay import RelayPlan

    goals = (
        [args.angle, args.angle / 2.0, -args.angle / 2.0, 0.0]
        if args.case == "crosszero"
        else [args.angle, 0.0]
    )
    if args.goals_deg is not None:
        goals = args.goals_deg
    limits = np.rad2deg(m.joint("wheel_joint").range)
    if min(goals) < limits[0] + 1 or max(goals) > limits[1] - 1:
        raise ValueError(
            f"Goals exceed the original wheel travel {limits.tolist()} deg"
        )
    plan = RelayPlan(goals, cfg["stroke_deg"], cfg["speed_deg_s"])
    duration = min(plan.duration, args.duration) if args.duration else plan.duration
    max_wait = cfg.get("handover_max_guard_wait_s", 0.0)
    maximum_runtime = (
        min(args.duration, plan.duration + 2 * len(plan.handovers) * max_wait)
        if args.duration
        else plan.duration + 2 * len(plan.handovers) * max_wait
    )
    first = next((g for g in goals if abs(g) > 1e-8), 1.0)
    load = (
        args.load
        if args.load is not None
        else -cfg["nominal_constant_load_Nm"] * np.sign(first)
    )
    config = dict(
        cfg,
        command="run",
        arguments=vars(args),
        goals_deg=goals,
        expected_duration_s=plan.duration,
        run_duration_s=duration,
        maximum_runtime_s=maximum_runtime,
        constant_load_Nm=load,
        grasp_source=grasp["preparation_source"],
        control_interpolation="linear 20 Hz targets interpolated at 500 Hz",
        transfer_torque_feedback="receiver lead integral, bounded 3 deg",
        privileged_input="palm normal force and individual wheel contact torques",
        wheel_assistance="none",
        compensation="bias bounded so motor plus applied compensation stays within original effort range",
    )
    freeze(output, config, scene)
    write_json(output / "grasp_used.json", grasp)
    serial = []
    for v in plan.segments:
        serial.append(
            {
                k: (x.tolist() if isinstance(x, np.ndarray) else x)
                for k, x in v.items()
                if k != "motion"
            }
        )
    write_json(
        output / "schedule.json", dict(segments=serial, handovers=plan.handovers)
    )
    shutil.copy2(output / "schedule.json", output / "schedule_nominal.json")
    library = Library(m, d, cfg, phases=np.arange(-65, 66, 5.0))
    write_json(output / "reachability.json", library.records)
    inset = np.array(grasp["inset"])
    bias = np.zeros(2)
    integral = 0.0
    last = None
    lastseg = -1
    stop_reason = None
    guard_events = []
    wait_started = {}
    for k in range(round(maximum_runtime / m.opt.timestep)):
        t = float(d.time)
        if t >= plan.duration - m.opt.timestep * 0.1:
            break
        state = plan.sample(t)
        ref = state["reference"]
        retreat = state["retreat"]
        support = state["support"]
        phases = state["hand_phases"]
        if state["segment"] != lastseg:
            if state["kind"] in ("transfer", "release", "solo") and rec.rows:
                new, old = state["new"], state["old"]
                window = rec.rows[-100:]
                if state["kind"] == "transfer":
                    valid = [r["rim_force"][new, 0] > 0.5 for r in window]
                    ready = len(valid) >= 50 and np.mean(valid) >= 0.95
                elif state["kind"] == "release":
                    valid = [
                        r["rim_force"][new, 0] > 0.05
                        and r["rim_force"][old, 0] > 0.05
                        and -np.sign(r["load_torque"]) * r["part_torque"][new].sum()
                        >= 0.5 * abs(r["load_torque"])
                        for r in window
                    ]
                    longest = current = 0
                    for value in valid:
                        current = current + 1 if value else 0
                        longest = max(longest, current)
                    ready = longest * m.opt.timestep >= 0.1
                else:
                    ready = (
                        last["rim_force"][old].sum() <= 0.05
                        and last["rim_force"][new, 0] > 0.05
                    )
                transition_key = (state["segment"], state["kind"])
                waiting = False
                if not ready:
                    began = wait_started.setdefault(transition_key, t)
                    waiting = (
                        state["kind"] in ("transfer", "release")
                        and t - began < max_wait - 1e-8
                    )
                guard_events.append(
                    dict(
                        time_s=t,
                        entering=state["kind"],
                        passed=bool(ready),
                        waiting=waiting,
                        waited_s=float(t - wait_started.get(transition_key, t)),
                    )
                )
                if not ready:
                    if waiting:
                        plan.extend_stationary_wait(state["segment"], ex.dt)
                        state = plan.sample(t)
                        ref, retreat, support, phases = (
                            state["reference"],
                            state["retreat"],
                            state["support"],
                            state["hand_phases"],
                        )
                    else:
                        stop_reason = f"Measured contact/load-transfer guard timed out before {state['kind']}"
                        print(stop_reason, flush=True)
                        break
            if state["segment"] != lastseg:
                print(
                    f't={t:.2f} {state["kind"]} ref={np.rad2deg(ref):.1f} phases={np.rad2deg(phases)}',
                    flush=True,
                )
            lastseg = state["segment"]
        if k % 25 == 0:
            error = ref - float(d.joint("wheel_joint").qpos[0])
            integral = np.clip(
                integral + cfg["angle_ki_per_s"] * ex.dt * error,
                -np.deg2rad(4),
                np.deg2rad(4),
            )
            lead = np.clip(
                cfg["angle_kp"] * error + integral,
                -np.deg2rad(cfg["angle_lead_limit_deg"]),
                np.deg2rad(cfg["angle_lead_limit_deg"]),
            )
            if state["kind"] == "transfer" and last is not None:
                new = state["new"]
                required = -last["load_torque"]
                measured = last["part_torque"][new].sum()
                bias[new] = np.clip(
                    bias[new] + 0.20 * ex.dt * (0.65 * required - measured),
                    -np.deg2rad(3),
                    np.deg2rad(3),
                )
            desired = np.r_[np.zeros(14), cfg["hand_drives"] * 2]
            for side in (0, 1):
                if retreat[side] < 0.15:
                    force = (
                        last["rim_force"][side, 0]
                        if last
                        else cfg["palm_force_target_N"]
                    )
                    inset[side] = np.clip(
                        inset[side]
                        + cfg["pressure_admittance_m_per_N_s"]
                        * ex.dt
                        * (cfg["palm_force_target_N"] - force),
                        *cfg["pressure_inset_bounds_m"],
                    )
                elif retreat[side] > 0.8:
                    inset[side] = 0.0
                    bias[side] = 0.0
                desired[7 * side : 7 * side + 7] = library.arm(
                    side,
                    phases[side],
                    retreat=retreat[side],
                    inset=inset[side],
                    lead=lead + bias[side],
                )
            ramp_start = d.ctrl.copy()
            ramp_target = ex.follow(desired, d.qpos[ex.qids])
        fraction = (k % 25 + 1) / 25.0
        d.ctrl[:] = (1 - fraction) * ramp_start + fraction * ramp_target
        applied_load = load - cfg["load_damping_Nm_s"] * d.qvel[wv]
        apply_forces(m, d, armv, wv, applied_load)
        last = rec.sample(
            d, ex, desired, ref, state["phase"], support, inset, applied_load, phases
        )
        rec.rows[-1]["schedule_segment"] = state["segment"]
        rec.rows[-1]["goal"] = state["goal"]
        rec.rows[-1]["transfer_bias"] = bias.copy()
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or abs(d.qvel).max() > 100:
            break
        if last["illegal"] > 5 or last["nonrim"].max() > 5:
            print("Stopped for significant illegal or non-rim contact", flush=True)
            break
    a = rec.save(output)
    config["actual_scheduled_duration_s"] = plan.duration
    write_json(output / "config.json", config)
    serial = [
        {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in segment.items()
            if k != "motion"
        }
        for segment in plan.segments
    ]
    write_json(
        output / "schedule.json", dict(segments=serial, handovers=plan.handovers)
    )
    result = evaluate(a, m, cfg, plan.duration, negative=args.negative_control)
    result["stop_reason"] = stop_reason
    write_json(output / "transition_guards.json", guard_events)
    transfers = []
    for event in plan.handovers:
        new, old = event["new"], event["old"]
        solo = (a["time"] >= event["solo_begin"]) & (a["time"] < event["end"])
        overlap = (a["time"] >= event["transfer_end"] - 3) & (
            a["time"] < event["transfer_end"]
        )
        counter = -np.sign(a["load_torque"]) * a["part_torque"][:, new].sum(axis=(1, 2))
        witness = (
            overlap
            & (a["rim_force"][:, old, 0] > 0.05)
            & (a["rim_force"][:, new, 0] > 0.05)
            & (counter >= 0.5 * abs(a["load_torque"]))
        )
        longest = current = 0
        for v in witness:
            current = current + 1 if v else 0
            longest = max(longest, current)
        window = (a["time"] >= event["begin"]) & (a["time"] < event["end"])
        change = (
            float(abs(np.rad2deg(a["wheel_angle"][window] - event["reference"])).max())
            if window.any()
            else None
        )
        metric = dict(
            **event,
            solo_seconds=float(solo.sum() * m.opt.timestep),
            transfer_witness_s=longest * m.opt.timestep,
            solo_old_force_max=(
                float(a["rim_force"][solo, old].sum(axis=1).max())
                if solo.any()
                else None
            ),
            solo_new_mean_counterload=(
                float(counter[solo].mean()) if solo.any() else None
            ),
            error_peak_deg=change,
        )
        metric["passed"] = bool(
            metric["solo_seconds"] >= 9.998
            and metric["transfer_witness_s"] >= 0.1
            and metric["solo_old_force_max"] <= 0.05
            and metric["solo_new_mean_counterload"] >= 0.5 * abs(load)
            and change <= 2
        )
        transfers.append(metric)
    result["handovers"] = transfers
    result["checks"]["all_handovers"] = (
        bool(transfers and all(x["passed"] for x in transfers))
        if plan.handovers
        else True
    )
    result["passed"] = bool(
        all(result["checks"].values()) and not args.negative_control
    )
    write_json(output / "evaluation.json", result)
    plot(output, a)
    print(json.dumps(result, indent=2), flush=True)


def scan(args, cfg):
    from humanlike_drive.controller.palm_workspace import WorkspaceClearance

    if not args.grasp:
        raise ValueError("--grasp is required")
    grasp = json.loads((Path(args.grasp) / "grasp.json").read_text())
    output, scene, m, d, _, _ = setup(args.output, cfg)
    d.qpos[:] = grasp["qpos"]
    d.joint("wheel_joint").qpos[0] = 0.0
    d.ctrl[:] = grasp["ctrl"]
    mujoco.mj_forward(m, d)
    ik = PalmPointIK(
        m,
        d,
        point_x=cfg["palm_point_x_m"],
        axial=cfg["palm_target_axial_m"],
        radius=cfg["palm_target_radius_m"],
    )
    geometry = WorkspaceClearance(m)
    config = dict(
        cfg,
        command="scan",
        scope="scratch kinematic screen, not dynamic proof",
        hand_upper_sweep_deg=list(range(0, 181, 15)),
        wheel_material_angles_deg=[0, 90, 180, 270, 360],
        other_hand_conditions=["initial_contact", "radial_retreat_70mm"],
    )
    freeze(output, config, scene)
    records = []
    poses = []
    initial = d.qpos.copy()
    free = []
    for side in (0, 1):
        q, _ = ik.solve(side, 0.0, radial_offset=0.07)
        free.append(q)
    for side in (0, 1):
        warm = ik.seed[side * 7 : side * 7 + 7].copy()
        for degree in range(0, 181, 15):
            phase = np.deg2rad(degree * (1 if side == 0 else -1))
            q, r = ik.solve(side, phase, warm=warm)
            warm = q
            for mode in ("initial_contact", "radial_retreat_70mm"):
                for material in (0, 90, 180, 270, 360):
                    d.qpos[:] = initial
                    d.qpos[ik.qids[7 * side : 7 * side + 7]] = q
                    if mode != "initial_contact":
                        d.qpos[ik.qids[7 * (1 - side) : 7 * (1 - side) + 7]] = free[
                            1 - side
                        ]
                    d.joint("wheel_joint").qpos[0] = np.deg2rad(material)
                    mujoco.mj_kinematics(m, d)
                    g = geometry.measure(d, side)
                    record = dict(
                        **r,
                        upper_sweep_deg=degree,
                        other_hand=mode,
                        wheel_material_deg=material,
                        **g,
                    )
                    record["kinematic_screen_passed"] = bool(
                        r["position_error_mm"] <= 5
                        and r["normal_error_deg"] <= 10
                        and r["wrist_bend_deg"] <= cfg["operating_wrist_bend_limit_deg"]
                        and r["min_margin_deg"] > 2
                        and g["min_obstacle_clearance_mm"] >= 0
                    )
                    records.append(record)
                    poses.append(d.qpos.copy())
            print(
                ("left", "right")[side],
                degree,
                "position mm",
                round(r["position_error_mm"], 1),
                "wrist deg",
                round(r["wrist_bend_deg"], 1),
                flush=True,
            )
    write_json(output / "workspace.json", records)
    np.savez_compressed(output / "workspace_poses.npz", qpos=poses)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for side, name in enumerate(("left", "right")):
        for mode, style in [("initial_contact", "--"), ("radial_retreat_70mm", "-")]:
            subset = [
                x
                for x in records
                if x["side"] == name
                and x["other_hand"] == mode
                and x["wheel_material_deg"] == 0
            ]
            deg = [x["upper_sweep_deg"] for x in subset]
            for row, key in enumerate(
                ("position_error_mm", "wrist_bend_deg", "min_obstacle_clearance_mm")
            ):
                axs[row, side].plot(deg, [x[key] for x in subset], style, label=mode)
                axs[row, side].grid(alpha=0.2)
        axs[0, side].set_title(name + " hand along upper semicircle")
        axs[2, side].set_xlabel("upper sweep deg")
        axs[0, side].legend(fontsize=8)
    for row, label in enumerate(
        ("palm target error mm", "wrist bend deg", "obstacle clearance mm")
    ):
        axs[row, 0].set_ylabel(label)
    fig.tight_layout()
    fig.savefig(output / "workspace.png", dpi=130)
    plt.close(fig)


def render(args):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    output = Path(args.output).resolve()
    archive = np.load(output / "trajectory.npz")
    a = {
        key: archive[key]
        for key in ("time", "qpos", "qvel", "reference", "wheel_angle", "rim_force")
    }
    archive.close()
    m = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=720, width=960)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    times = a["time"]
    indices = np.unique(
        np.clip(
            np.searchsorted(times, np.arange(0, times[-1], args.speed / args.fps)),
            0,
            len(times) - 1,
        )
    )
    dest = output / f"replay_{args.camera}.mp4"
    if dest.exists():
        raise FileExistsError(dest)
    with imageio.get_writer(
        dest, fps=args.fps, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for n, k in enumerate(indices):
            d.qpos[:] = a["qpos"][k]
            d.qvel[:] = a["qvel"][k]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=args.camera, scene_option=opt)
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 960, 60), fill=(15, 22, 30))
            draw.text(
                (12, 8),
                f"Layer 1 | contact-driven engineering controller | synthetic load | {args.speed:g}x",
                fill="white",
            )
            draw.text(
                (12, 30),
                f't={times[k]:.2f}s  ref={np.rad2deg(a["reference"][k]):+.1f} deg  wheel={np.rad2deg(a["wheel_angle"][k]):+.1f} deg  palms={a["rim_force"][k,0,0]:.2f}/{a["rim_force"][k,1,0]:.2f} N',
                fill="white",
            )
            writer.append_data(np.asarray(picture))
            if n in (
                0,
                len(indices) // 4,
                len(indices) // 2,
                3 * len(indices) // 4,
                len(indices) - 1,
            ):
                picture.save(output / f"preview_{args.camera}_{n:05d}.png")
    renderer.close()
    print(dest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run", "render", "scan"])
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--grasp")
    p.add_argument("--angle", type=float, default=15)
    p.add_argument(
        "--goals-deg",
        type=float,
        nargs="+",
        help="Unwrapped waypoints for --case relay or crosszero; state is preserved between goals",
    )
    p.add_argument("--side", choices=["left", "right"])
    p.add_argument("--load", type=float)
    p.add_argument("--friction", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--negative-control", action="store_true")
    p.add_argument(
        "--both-hands",
        action="store_true",
        help="Dual-palm stationary hold; only valid with --case hold",
    )
    p.add_argument("--camera", default="driver")
    p.add_argument(
        "--case",
        choices=["roundtrip", "hold", "relay", "crosszero"],
        default="roundtrip",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset-mm", type=float, default=0)
    p.add_argument(
        "--target-offset-mm",
        type=float,
        default=0,
        help="Per-hand radial/tangential target registration error; finite motors preserve the legal initial physical state",
    )
    p.add_argument("--finger-closure", type=float)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()
    if args.speed <= 0 or args.fps <= 0 or args.offset_mm < 0:
        p.error("Speed/fps must be positive and initial offset nonnegative")
    if args.duration is not None and args.duration <= 0:
        p.error("Duration must be positive")
    if args.friction is not None and args.friction <= 0:
        p.error("Friction must be positive")
    if args.goals_deg is not None and args.case not in ("relay", "crosszero"):
        p.error("--goals-deg requires --case relay or crosszero")
    if args.both_hands and args.case != "hold":
        p.error("--both-hands is only supported for --case hold")
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        prepare(args, cfg)
    elif args.command == "run":
        run(args, cfg)
    elif args.command == "scan":
        scan(args, cfg)
    else:
        render(args)


if __name__ == "__main__":
    main()
