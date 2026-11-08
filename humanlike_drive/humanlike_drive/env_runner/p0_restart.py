"""Restart P0 with an explicitly selected, clearance-tested new grasp.

Example: bash scripts/run.sh -m humanlike_drive.env_runner.p0_restart \
    --grasp assets/scenes/p0_clear_grasp_v1.json --angle 30 \
    --output outputs/p0_restart/right30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from humanlike_drive.env.logitech_scene import ARM_NAMES, HAND_NAMES, PROJECT, SCENE
from humanlike_drive.env.inspire_hand import HAND_URDF, mimic_relations
from humanlike_drive.env.p0_scene import ASSEMBLY_EXCLUSIONS, build_scene
from humanlike_drive.controller.basic_grasps import apply, TARGET_CONFIG, joint_addresses
from humanlike_drive.controller.p0_control import RateExecutor, HandPath, RoundTrip, release_motion
from humanlike_drive.controller.clearance_grasp import Clearance

PARTS = ("index", "middle", "ring", "pinky", "palm", "thumb")


class Recorder:
    def __init__(self, model):
        self.model = model
        self.names = [model.geom(i).name or f"geom_{i}" for i in range(model.ngeom)]
        self.armj = [model.joint(n).id for n in ARM_NAMES]
        self.armq = joint_addresses(model, ARM_NAMES)
        self.armv = [model.jnt_dofadr[j] for j in self.armj]
        self.drivev = [model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0]]
        self.robot = model.geom_contype == 2
        self.wheel = np.array([n.startswith("collision_") for n in self.names])
        self.rim = np.array([n.startswith("collision_rim_") for n in self.names])
        self.hand_side = np.array([0 if n.startswith(("L_", "left_")) else
                                   1 if n.startswith(("R_", "right_")) else -1 for n in self.names])
        self.hand_part = np.array([next((i for i, p in enumerate(PARTS)
            if (p+"_") in n or (p == "palm" and "wrist_yaw_link" in n)), -1) for n in self.names])
        self.rows, self.contacts, self.anomalies = [], [], []
        self.clearance = Clearance(model)
        self.last_clearance = np.array([.1, .1])
        self.mimic = [(model.jnt_qposadr[model.joint(r["joint"]).id],
                       model.jnt_qposadr[model.joint(r["parent"]).id],
                       r["multiplier"], r["offset"]) for r in mimic_relations()]

    def sample(self, data, reference, phase, segment_target, executor, desired, load, support):
        model, step = self.model, len(self.rows)
        forces = np.zeros((2, 6))
        torques = np.zeros(2)
        nonrim = np.zeros(2)
        rim_penetration, illegal_force, illegal_penetration = 0., 0., 0.
        center = data.site("wheel_frame").xpos
        axis = data.site("wheel_frame").xmat.reshape(3, 3)[:, 0]
        for index, c in enumerate(data.contact):
            a, b = c.geom1, c.geom2
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, f)
            self.contacts.append(np.r_[step, a, b, c.dist, c.pos, f])
            wheel_id = a if self.wheel[a] else b if self.wheel[b] else -1
            if wheel_id >= 0:
                other = b if wheel_id == a else a
                side, part = self.hand_side[other], self.hand_part[other]
                if self.rim[wheel_id]:
                    rim_penetration = max(rim_penetration, -c.dist)
                if side >= 0:
                    if self.rim[wheel_id] and part >= 0:
                        forces[side, part] += f[0]
                    elif not self.rim[wheel_id]:
                        nonrim[side] += f[0]
                    # mj_contactForce is force on geom2 in the contact frame.
                    sign = 1. if wheel_id == b else -1.
                    frame = c.frame.reshape(3, 3)
                    force_world, moment_world = sign*frame.T@f[:3], sign*frame.T@f[3:]
                    torques[side] += axis @ (np.cross(c.pos-center, force_world)+moment_world)
            elif self.robot[a] or self.robot[b]:
                # Ground/seat interfaces of the fixed lower body are separate
                # from motion collisions. All self contacts remain recorded.
                moving = any(self.names[g].startswith(("left_", "right_", "L_", "R_")) for g in (a, b))
                if (self.robot[a] and self.robot[b]) or moving:
                    illegal_force = max(illegal_force, f[0])
                    illegal_penetration = max(illegal_penetration, -c.dist)
                    if f[0] > .05:
                        self.anomalies.append({"time_s": float(data.time), "pair": [self.names[a], self.names[b]],
                                               "normal_N": float(f[0]), "distance_mm": float(c.dist*1000)})
        q = data.qpos[self.armq]
        margins = np.minimum(q-model.jnt_range[self.armj, 0], model.jnt_range[self.armj, 1]-q)
        mimic = max(abs(data.qpos[a]-m*data.qpos[b]-o) for a,b,m,o in self.mimic)
        wheel = data.joint("wheel_joint")
        if step % 10 == 0:
            self.last_clearance = np.array([
                self.clearance.distance(data, self.clearance.paddles),
                self.clearance.distance(data, self.clearance.spokes)])
        wrist_bend = []
        for side in ("left", "right"):
            forearm = data.body(f"{side}_wrist_roll_link").xmat.reshape(3, 3)[:, 0]
            hand = data.body(f"{side}_wrist_yaw_link").xmat.reshape(3, 3)[:, 0]
            wrist_bend.append(np.arccos(np.clip(forearm @ hand, -1., 1.)))
        self.rows.append(dict(time=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
            reference=reference, phase=phase, segment_target=segment_target,
            wheel_angle=float(wheel.qpos[0]), wheel_velocity=float(wheel.qvel[0]),
            command=desired.copy(), action=executor.action.copy(), previous_action=executor.previous_action.copy(),
            ctrl=data.ctrl.copy(), actuator_force=data.actuator_force.copy(),
            drive_total_force=data.actuator_force.copy()+data.qfrc_applied[self.drivev],
            qfrc_applied=data.qfrc_applied.copy(), qfrc_constraint=data.qfrc_constraint.copy(),
            qfrc_passive=data.qfrc_passive.copy(), qacc=data.qacc.copy(),
            arm_margins=margins, rim_forces=forces, hand_wheel_torque=torques,
            nonrim_forces=nonrim, rim_penetration=rim_penetration,
            illegal_force=illegal_force, illegal_penetration=illegal_penetration,
            mimic_error=mimic, load_torque=load, active_support=np.array(support), wrist_bend=wrist_bend,
            clearance_sampled=(step % 10 == 0), nonrim_clearance_sample_hold=self.last_clearance.copy()))

    def save(self, output):
        arrays = {key: np.array([r[key] for r in self.rows]) for key in self.rows[0]}
        np.savez_compressed(output / "trajectory.npz", **arrays)
        np.savez_compressed(output / "contacts.npz", contacts=np.asarray(self.contacts).reshape(-1, 13))
        (output / "contact_schema.json").write_text(json.dumps({
            "columns": ["step", "geom1", "geom2", "distance_m", "x", "y", "z", "Fn", "Ft1", "Ft2", "Mn", "Mt1", "Mt2"],
            "geom_names": self.names, "force_frame": "MuJoCo contact frame; force on geom2",
            "support_parts": PARTS}, indent=2)+"\n")
        (output / "collision_events.json").write_text(json.dumps(self.anomalies, indent=2)+"\n")
        return arrays


def evaluate(arrays, model, expected_duration, *, hold_only=False, negative_control=False, single=False, require_nonzero_load=False, handover=False, hold_phase_ids=None, motion_phase_ids=None):
    """Read trajectory evidence, never the controller's completion flag."""
    a = arrays
    error = np.rad2deg(a["wheel_angle"]-a["reference"])
    motion = np.isin(a["phase"], [1, 3] if motion_phase_ids is None else motion_phase_ids)
    hold_peak, hold_fractions = {}, {}
    support_ok = True
    hold_phases = [0] if hold_only else [0, 2, 6, 8, 4] if single else [0, 2, 4]
    if handover:
        hold_phases += [10, 12]
    if hold_phase_ids is not None:
        hold_phases = list(hold_phase_ids)
    for phase in hold_phases:
        idx = np.flatnonzero(a["phase"] == phase)
        if not len(idx):
            continue
        tail = idx[a["time"][idx] >= a["time"][idx[-1]]-1.]
        hold_peak[str(phase)] = float(np.max(np.abs(error[tail])))
        hold_fractions[str(phase)] = np.mean(a["rim_forces"][idx] > .05, axis=0).tolist()
        for side in (0, 1):
            active = idx[a["active_support"][idx, side]]
            if len(active):
                support_ok &= bool(np.min(np.mean(a["rim_forces"][active, side] > .05, axis=0)) >= .95)
    # Include mimic descendants and the wheel, all limited hinges.
    allq = a["qpos"][:, model.jnt_qposadr]
    limit_violation = np.maximum(model.jnt_range[:,0]-allq, allq-model.jnt_range[:,1])
    limited = model.jnt_limited.astype(bool)
    all_limit_violation = float(np.max(np.maximum(0., limit_violation[:, limited])))
    force_violation = float(np.max(np.maximum(0., np.maximum(
        model.actuator_forcerange[:,0]-a["actuator_force"],
        a["actuator_force"]-model.actuator_forcerange[:,1]))))
    rmse = float(np.sqrt(np.mean(error[motion]**2))) if motion.any() else None
    checks = {
        "complete_finite_execution": bool(a["time"][-1] >= expected_duration-.003 and
                                           all(np.isfinite(v).all() for v in a.values())),
        "motion_RMSE_le_2deg": bool(rmse is None or rmse <= 2.),
        "hold_tail_peak_le_2deg": bool(len(hold_peak)==len(hold_phases) and max(hold_peak.values()) <= 2.),
        "arm_margin_gt_2deg": bool(np.rad2deg(np.min(a["arm_margins"])) > 2.),
        "joint_limits_respected": bool(all_limit_violation <= 1e-10),
        "finite_actuator_limits": bool(force_violation < 1e-8),
        "rim_penetration_lt_0p5mm": bool(np.max(a["rim_penetration"])*1000 < .5),
        "no_illegal_collision_gt_0p05N": bool(np.max(a["illegal_force"]) <= .05),
        "no_nonrim_contact_gt_0p05N": bool(np.max(a["nonrim_forces"]) <= .05),
        "support_hold_fractions_ge_0p95": bool(support_ok),
    }
    # The wheel's qfrc_applied must contain declared load only.
    wv = model.jnt_dofadr[model.joint("wheel_joint").id]
    checks["wheel_applied_equals_declared_load"] = bool(np.max(np.abs(a["qfrc_applied"][:, wv]-a["load_torque"])) < 1e-12)
    drivev = [model.jnt_dofadr[j] for j in model.actuator_trnid[:,0]]
    drive_total = a["actuator_force"]+a["qfrc_applied"][:,drivev]
    checks["motor_plus_compensation_within_force_limits"] = bool(np.all(
        (drive_total >= model.actuator_forcerange[:,0]-1e-8) &
        (drive_total <= model.actuator_forcerange[:,1]+1e-8)))
    # Wheel constraint torque also includes its dry joint friction. Agreement
    # within that bound verifies the contact-force sign and moment convention.
    torque_residual = np.sum(a["hand_wheel_torque"], axis=1)-a["qfrc_constraint"][:, wv]
    checks["contact_torque_accounting"] = bool(np.max(np.abs(torque_residual)) <= model.dof_frictionloss[wv]+1e-7)
    sampled = a["nonrim_clearance_sample_hold"][a["clearance_sampled"]]
    checks["sampled_paddle_clearance_ge_1mm"] = bool(np.min(sampled[:,0]) >= .001)
    if single:
        idx = np.flatnonzero(a["phase"] == 6)
        released = ~a["active_support"][idx]
        checks["released_hand_clear_of_rim_during_single_hold"] = bool(len(idx) and
            np.max(np.sum(a["rim_forces"][idx], axis=2)[released]) <= .05)
        checks["single_hold_lasts_10s"] = bool(len(idx)*model.opt.timestep >= 9.998)
        if require_nonzero_load:
            checks["single_hold_declared_load_ge_0p02Nm"] = bool(len(idx) and np.min(np.abs(a["load_torque"][idx])) >= .02)
    handover_metrics = None
    if handover:
        idx = np.flatnonzero(a["phase"] == 10)
        transition = np.flatnonzero(np.isin(a["phase"], [9, 10]))
        if len(idx) and len(transition):
            new_side = int(np.flatnonzero(a["active_support"][idx[0]])[0])
            old_side = 1-new_side
            released_force = np.sum(a["rim_forces"][idx, old_side], axis=1)
            against_load = -np.sign(a["load_torque"][idx])*a["hand_wheel_torque"][idx, new_side]
            angle_retreat = float(np.max(np.abs(np.rad2deg(
                a["wheel_angle"][transition]-a["wheel_angle"][transition[0]]))))
            # Witness load transfer while both hands still touch. Merely
            # scheduling a new contact or returning the same hand is insufficient.
            transfer = np.flatnonzero(a["phase"] == 9)
            witness = (np.min(a["rim_forces"][transfer, new_side], axis=1) > .05) & (
                np.sum(a["rim_forces"][transfer, old_side], axis=1) > .05) & (
                -np.sign(a["load_torque"][transfer])*a["hand_wheel_torque"][transfer, new_side]
                >= np.maximum(.02, .5*np.abs(a["load_torque"][transfer])))
            longest, current = 0, 0
            for valid in witness:
                current = current+1 if valid else 0
                longest = max(longest, current)
            witness_s = longest*model.opt.timestep
            checks.update({
                "handover_old_hand_clear_during_new_support": bool(np.max(released_force) <= .05),
                "handover_new_hand_holds_10s": bool(len(idx)*model.opt.timestep >= 9.998),
                "handover_new_hand_mean_counterload_torque_ge_0p02Nm": bool(np.mean(against_load) >= .02),
                "handover_load_transfer_overlap_ge_0p1s": bool(witness_s >= .1),
                "handover_wheel_retreat_le_2deg": bool(angle_retreat <= 2.),
            })
            handover_metrics = {"new_support_side": ("left", "right")[new_side],
                "mean_new_support_counterload_torque_Nm": float(np.mean(against_load)),
                "max_absolute_wheel_change_during_transfer_and_hold_deg": angle_retreat,
                "continuous_load_transfer_overlap_s": witness_s,
                "old_hand_max_rim_force_during_new_support_N": float(np.max(released_force))}
        else:
            checks["handover_completed"] = False
    first_fail = {}
    for name, mask in {
        "nonrim": np.max(a["nonrim_forces"], axis=1) > .05,
        "illegal_collision": a["illegal_force"] > .05,
        "penetration": a["rim_penetration"] >= .0005,
        "arm_margin": np.min(a["arm_margins"], axis=1) <= np.deg2rad(2.),
    }.items():
        indices = np.flatnonzero(mask)
        if len(indices):
            first_fail[name] = float(a["time"][indices[0]])
    return {"passed": bool(all(checks.values()) and not negative_control), "checks": checks,
            "negative_control": negative_control, "motion_RMSE_deg": rmse,
            "hold_tail_peak_deg": hold_peak, "hold_tail_support_fraction_left_right": hold_fractions,
            "minimum_arm_margin_deg": float(np.rad2deg(np.min(a["arm_margins"]))),
            "max_all_joint_limit_violation_deg": float(np.rad2deg(all_limit_violation)),
            "max_rim_penetration_mm": float(np.max(a["rim_penetration"])*1000),
            "max_nonrim_force_N": float(np.max(a["nonrim_forces"])),
            "max_illegal_force_N": float(np.max(a["illegal_force"])),
            "max_mimic_error_deg": float(np.rad2deg(np.max(a["mimic_error"]))),
            "max_load_torque_Nm": float(np.max(np.abs(a["load_torque"]))),
            "max_wheel_angle_deg": float(np.rad2deg(np.max(np.abs(a["wheel_angle"])))),
            "max_contact_minus_constraint_torque_Nm": float(np.max(np.abs(torque_residual))),
            "minimum_paddle_clearance_mm_at_50Hz": float(np.min(sampled[:,0])*1000),
            "minimum_spoke_clearance_mm_at_50Hz": float(np.min(sampled[:,1])*1000),
            "wrist_bend_max_deg_left_right": np.rad2deg(np.max(a["wrist_bend"], axis=0)).tolist(),
            "max_motor_plus_compensation_abs_Nm_by_drive": np.max(np.abs(drive_total), axis=0).tolist(),
            "first_failure_s": first_fail, "duration_s": float(a["time"][-1]),
            "handover": handover_metrics,
            "hold_support_scope": "entire hold for support fractions, final 1 s for peak angle error; not force closure proof"}


def plots(output, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(12, 17), sharex=True)
    t = a["time"]
    axes[0].plot(t, np.rad2deg(a["reference"]), "--", label="reference")
    axes[0].plot(t, np.rad2deg(a["wheel_angle"]), label="wheel")
    axes[0].set_ylabel("angle (deg)")
    axes[1].plot(t, np.rad2deg(a["wheel_angle"]-a["reference"]))
    axes[1].axhline(2, color="r", ls=":"); axes[1].axhline(-2, color="r", ls=":")
    axes[1].set_ylabel("error (deg)")
    for side, label in enumerate(("left", "right")):
        axes[2].plot(t, a["hand_wheel_torque"][:, side], label=label)
        for part, name in enumerate(PARTS):
            axes[3+side].plot(t, a["rim_forces"][:, side, part], label=name)
    axes[2].plot(t, a["load_torque"], "k--", label="synthetic load")
    axes[2].set_ylabel("wheel torque (Nm)")
    axes[3].set_ylabel("left rim force (N)"); axes[4].set_ylabel("right rim force (N)")
    axes[5].plot(t, np.rad2deg(np.min(a["arm_margins"], axis=1)), label="min arm margin (deg)")
    axes[5].plot(t, 1000*a["rim_penetration"], label="rim penetration (mm)")
    axes[5].plot(t, np.max(a["nonrim_forces"], axis=1), label="nonrim force (N)")
    for ax in axes:
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(output.name+" | uncalibrated simulation engineering trial")
    fig.tight_layout()
    fig.savefig(output / "tracking.png", dpi=130)
    plt.close(fig)


def render_video(output, camera="driver", fps=25):
    """Replay saved states for visualization only; never counted as physics."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    a = np.load(output / "trajectory.npz")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    path = output / f"replay_{camera}.mp4"
    indices = np.unique(np.searchsorted(a["time"], np.arange(0, a["time"][-1], 1/fps)))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
        for k in indices:
            data.qpos[:] = a["qpos"][k]
            data.qvel[:] = a["qvel"][k]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 960, 45), fill=(15, 22, 30))
            draw.text((12, 8), f"P0 simulation | t={a['time'][k]:.2f}s | ref={np.rad2deg(a['reference'][k]):+.1f} deg | wheel={np.rad2deg(a['wheel_angle'][k]):+.1f} deg", fill="white")
            writer.append_data(np.asarray(picture))
            if k == indices[len(indices)//2]:
                picture.save(output / f"preview_{camera}.png")
    renderer.close()
    return path


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trajectory.npz").exists():
        raise FileExistsError(f"Refusing to overwrite an existing trial: {output}")
    scene = build_scene(output / "scene.xml", wheel_contacts=not args.negative_control)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    grasp_file = Path(args.grasp).resolve()
    grasp = json.loads(grasp_file.read_text())
    apply(model, data, "p0_clear", {"p0_clear": grasp})
    (output / "grasp_used.json").write_text(grasp_file.read_text())
    executor = RateExecutor(model, data)
    initial_ctrl = data.ctrl.copy()
    handover = args.case == "handover"
    single = args.case in ("single", "handover")
    reference = RoundTrip(args.angle, target_hold_s=45. if handover else 24. if single else None) if args.case != "hold" else None
    expected_duration = reference.duration if reference else 10.
    duration = expected_duration
    if args.duration:
        duration = min(duration, args.duration)
    path = None
    retreat_paths = {}
    release_side = 0 if args.release_side == "left" else 1
    if reference:
        print("Computing offline IK (scratch data only)", flush=True)
        path = HandPath(model, data, maximum_deg=args.scan or abs(args.angle)+6, step_deg=2.)
        (output / "reachability.json").write_text(json.dumps(path.records, indent=2)+"\n")
        if single:
            for side in ((0, 1) if handover else (release_side,)):
                retreat_paths[side], retreat_reports = path.retreat_commands(np.deg2rad(args.angle), side)
                (output / f"retreat_reachability_{side}.json").write_text(json.dumps(retreat_reports, indent=2)+"\n")
    recorder = Recorder(model)
    armv = [model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES]
    wv = model.jnt_dofadr[model.joint("wheel_joint").id]
    wq = model.jnt_qposadr[model.joint("wheel_joint").id]
    config = vars(args).copy()
    config.update({"physics_dt_s": model.opt.timestep, "control_dt_s": executor.dt,
                   "action_joint_order": ARM_NAMES+HAND_NAMES, "action": "normalized target rate [-1,1]",
                   "rate_limit_rad_s": executor.vmax.tolist(), "preload_limit_rad": executor.preload.tolist(),
                   "arm_command_limit_margin_deg": 2.1, "positive_wheel_direction": "driver clockwise / right",
                   "load_model": "uncalibrated synthetic tau=-k*theta-b*omega; original joint damping/friction also active",
                   "setup_fixture": "none during rollout; new finite-motor grasp preparation recorded separately",
                   "arm_compensation": "qfrc_bias on 14 arm DOFs only; logged each physics step",
                   "collision_exclusions": ASSEMBLY_EXCLUSIONS,
                   "wheel_joint_range_deg": np.rad2deg(model.jnt_range[model.joint("wheel_joint").id]).tolist(),
                   "expected_duration_s": expected_duration, "run_duration_s": duration,
                   "motion_scene_version": "p0_clearance_restart_v1", "seed": 0,
                   "nonrim_geometry_measurement_Hz": 50.,
                   "retreat_direction": "0.6 toward driver + 0.8 radial outward, total 0.06 m" if single else None,
                   "regrasp_version": "v2: half-curled fingers near housing; extend only after retreat; feedback-continuous approach" if single else None,
                   "single_release_schedule_s_after_arrival": {"double_hold": [0,3], "release": [3,6],
                       "single_hold": [6,16], "regrasp": [16,21], "double_hold_after": [21,24]} if single else None,
                   "handover_schedule_s_after_arrival": {"old_support_release": [24,27], "new_support_hold": [27,37],
                       "old_support_regrasp": [37,42], "double_hold": [42,45]} if handover else None,
                   "handover_criteria": "new hand counterload torque >= max(0.02 Nm, half declared load) while both touch for 0.1 s; then new hand supports alone for 10 s; wheel change <=2 deg" if handover else None})
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    sources = [SCENE, grasp_file, TARGET_CONFIG, HAND_URDF, Path(__file__),
               PROJECT / "humanlike_drive/controller/p0_control.py",
               PROJECT / "humanlike_drive/controller/basic_grasps.py",
               PROJECT / "humanlike_drive/env/inspire_hand.py",
               PROJECT / "humanlike_drive/env/p0_scene.py",
               PROJECT / "humanlike_drive/controller/clearance_grasp.py", scene]
    hashes = {}
    source_dir = output / "source_snapshot"; source_dir.mkdir(exist_ok=True)
    for source in sources:
        content = source.read_bytes()
        hashes[str(source)] = hashlib.sha256(content).hexdigest()
        (source_dir / source.name).write_bytes(content)
    # Asset hashes complement the scene and source snapshot without copying CAD.
    import xml.etree.ElementTree as ET
    for asset in ET.parse(scene).getroot().findall("./asset/*"):
        if asset.get("file"):
            p = Path(asset.get("file"))
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=PROJECT, capture_output=True, text=True)
    (output / "manifest.json").write_text(json.dumps({"git_commit": git.stdout.strip(),
        "git_status": status.stdout, "sha256": hashes, "mujoco_version": mujoco.__version__}, indent=2)+"\n")
    desired = initial_ctrl.copy()
    period = round(executor.dt/model.opt.timestep)
    print(f"Physics: {args.case}, angle={args.angle}, load k={args.load_k} b={args.load_b}, {duration:.2f}s", flush=True)
    for step in range(round(duration/model.opt.timestep)):
        t = float(data.time)
        ref, phase, segment = reference.sample(t) if reference else (0., 0, 0.)
        support = [True, True]
        moving_side = release_side
        retreat_alpha, released_hand_target = 0., None
        hand_slice = slice(14+release_side*6, 20+release_side*6)
        if single and phase == 2:
            stage_t = t-reference.initial_hold-reference.move
            if stage_t < 3.:
                pass  # both hands hold the nonzero wheel angle
            elif stage_t < 6.:
                phase = 5  # open and retreat
                u = stage_t-3.
                retreat_alpha, released_hand_target = release_motion(u, initial_ctrl[hand_slice])
            elif stage_t < 16.:
                phase = 6
                retreat_alpha, released_hand_target = release_motion(3., initial_ctrl[hand_slice])
                support[release_side] = False
            elif stage_t < 21.:
                phase = 7  # approach half-curled, then close at the rim
                u = stage_t-16.
                retreat_alpha, released_hand_target = release_motion(u, initial_ctrl[hand_slice], regrasp=True)
            else:
                phase = 8
            if handover and stage_t >= 24.:
                moving_side = 1-release_side  # original support now leaves
                hand_slice = slice(14+moving_side*6, 20+moving_side*6)
                if stage_t < 27.:
                    phase = 9
                    retreat_alpha, released_hand_target = release_motion(stage_t-24., initial_ctrl[hand_slice])
                elif stage_t < 37.:
                    phase = 10
                    retreat_alpha, released_hand_target = release_motion(3., initial_ctrl[hand_slice])
                    support[moving_side] = False
                elif stage_t < 42.:
                    phase = 11
                    retreat_alpha, released_hand_target = release_motion(stage_t-37., initial_ctrl[hand_slice], regrasp=True)
                else:
                    phase = 12
        if step % period == 0:
            if path:
                # Feedback acts through the finite arm motors, never the wheel.
                lead = np.clip(args.feedback_gain*(ref-data.qpos[wq]), -np.deg2rad(4), np.deg2rad(4))
                desired = path.command(ref+lead)
                if phase in (5, 6, 7, 9, 10, 11):
                    retreat_path = retreat_paths[moving_side]
                    arm_slice = slice(moving_side*7, moving_side*7+7)
                    # Match the active feedback anchor continuously at the rim.
                    correction = desired[arm_slice]-retreat_path(0.)
                    desired[arm_slice] = retreat_path(retreat_alpha)+(1.-retreat_alpha)*correction
                    desired[hand_slice] = released_hand_target
            data.ctrl[:] = executor.follow(desired, data.qpos[executor.qids])
        load = -args.load_k*data.qpos[wq] - args.load_b*data.qvel[wv]
        data.qfrc_applied[:] = 0.
        data.qfrc_applied[armv] = data.qfrc_bias[armv]
        data.qfrc_applied[wv] = load
        # mj_forward aligns forces/contact state with the state being logged.
        # The same controls and applied torques drive the following mj_step.
        mujoco.mj_forward(model, data)
        recorder.sample(data, ref, phase, segment, executor, desired, load, support)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 100.:
            print(f"Stopped for numerical instability at {t:.3f}s", flush=True)
            break
        if step % 2500 == 0:
            print(f"t={t:.1f} ref={np.rad2deg(ref):+.2f} wheel={np.rad2deg(data.qpos[wq]):+.2f}", flush=True)
    a = recorder.save(output)
    result = evaluate(a, model, expected_duration, hold_only=reference is None, negative_control=args.negative_control,
                      single=single, require_nonzero_load=single and abs(args.angle)>1., handover=handover)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    plots(output, a)
    print(json.dumps(result, indent=2), flush=True)
    if args.video:
        print(render_video(output, args.camera), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grasp", help="new, physically prepared grasp JSON; required for physics")
    parser.add_argument("--case", choices=["hold", "roundtrip", "single", "handover"], default="roundtrip")
    parser.add_argument("--release-side", choices=["left", "right"], default="right")
    parser.add_argument("--angle", type=float, default=30.)
    parser.add_argument("--load-k", type=float, default=.1)
    parser.add_argument("--load-b", type=float, default=.02)
    parser.add_argument("--feedback-gain", type=float, default=.5)
    parser.add_argument("--scan", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--camera", default="driver")
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    if args.render_existing:
        print(render_video(Path(args.output).resolve(), args.camera))
    else:
        if not args.grasp:
            parser.error("--grasp is required: select the new clearance-tested grasp")
        run(args)


if __name__ == "__main__":
    main()
