"""S00: freeze a scene and probe its fixture, without a policy or wheel servo."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from .contract import ACTION_DIM, MOTOR_NAMES, OBSERVATION_DIM, INTERFACE_VERSION
from .wheel_load import WheelLoadConfig

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(source):
    """Snapshot XML only; external robot meshes remain a declared dependency."""
    source = Path(source).expanduser().resolve()
    target = ROOT / "assets/scene.xml"
    provenance = target.with_suffix(".provenance.json")
    if target.exists() or provenance.exists():
        raise FileExistsError("Scene snapshot exists; use a separate version rather than overwriting it")
    tree = ET.parse(source)
    compiler = tree.getroot().find("compiler")
    mesh_dir = Path(os.environ.get("H1_MESH_DIR", compiler.get("meshdir", "."))).expanduser()
    if not mesh_dir.is_absolute():
        mesh_dir = source.parent / mesh_dir
    mesh_dir = mesh_dir.resolve()
    meshes = {}
    for mesh in tree.getroot().findall("asset/mesh"):
        name = mesh.get("file")
        if name:
            meshes[name] = digest(mesh_dir / name)
    compiler.set("meshdir", str(mesh_dir))
    tree.write(target, encoding="utf-8", xml_declaration=True)
    write_json(provenance, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": digest(source),
        "snapshot_sha256": digest(target), "mesh_directory": str(mesh_dir),
        "mesh_sha256": meshes,
        "reuse": "XML and external meshes only; no previous controller, policy, or environment imports",
    })
    print(target)


class Fixture:
    def __init__(self, scene, config):
        self.config = config
        self.model = m = mujoco.MjModel.from_xml_path(str(scene))
        self.data = d = mujoco.MjData(m)
        self.jids = np.array([m.joint(name).id for name in MOTOR_NAMES])
        self.qidx = m.jnt_qposadr[self.jids]
        self.vidx = m.jnt_dofadr[self.jids]
        self.wheel_id = m.joint("wheel_joint").id
        self.wheel_q = int(m.jnt_qposadr[self.wheel_id])
        self.wheel_v = int(m.jnt_dofadr[self.wheel_id])
        self.wheel_body = m.body("steering_wheel").id
        self.wheel_load = WheelLoadConfig.from_config(config)
        self.wheel_load.configure_joint(m, self.wheel_id)
        self.last_wheel_load = {}
        self.grips = [m.site(f"{side}_grip").id for side in ("left", "right")]
        self.hand_bodies = []
        for side in ("left", "right"):
            wrist = m.body(f"{side}_wrist_yaw_link").id
            bodies = {wrist}
            for body in range(wrist + 1, m.nbody):
                if m.body_parentid[body] in bodies:
                    bodies.add(body)
            self.hand_bodies.append(bodies)
        mujoco.mj_forward(m, d)
        site = m.site("wheel_center").id
        self.center = d.site_xpos[site].copy()
        self.frame = d.site_xmat[site].reshape(3, 3).copy()
        self.substeps = round(1 / config["policy_hz"] / m.opt.timestep)
        if abs(self.substeps * m.opt.timestep - 1 / config["policy_hz"]) > 1e-9:
            raise ValueError("Policy period must be an integer number of physics steps")

    def structural_report(self):
        m = self.model
        joint_transmission = m.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)
        passive = not np.any(joint_transmission & (m.actuator_trnid[:, 0] == self.wheel_id))
        ordering = m.nu == ACTION_DIM and np.array_equal(m.actuator_trnid[:, 0], self.jids)
        position_pd = bool(np.all(m.actuator_gainprm[:, 0] > 0)
                           and np.all(m.actuator_biastype == int(mujoco.mjtBias.mjBIAS_AFFINE))
                           and np.allclose(m.actuator_biasprm[:, 1], -m.actuator_gainprm[:, 0])
                           and np.all(m.actuator_biasprm[:, 2] <= 0))
        finite = bool(np.all(m.actuator_forcelimited)
                      and np.isfinite(m.actuator_forcerange).all()
                      and np.all(m.actuator_forcerange[:, 0] < 0)
                      and np.all(m.actuator_forcerange[:, 1] > 0))
        sample = [m.body(name).id for name in ("L_thumb_distal", "L_index_intermediate")]
        signature = min(sample) * (1 << 16) + max(sample)
        cross_finger = signature not in m.exclude_signature.tolist()
        checks = {"wheel_passive": bool(passive), "26_joint_actuator_order": bool(ordering),
                  "finite_actuator_force_limits": finite, "position_pd_actuators": position_pd,
                  "sample_cross_finger_pair_not_excluded": bool(cross_finger)}
        return {"checks": checks, "passed": all(checks.values()),
                "nu": m.nu, "nq": m.nq, "nv": m.nv, "equality_constraints": m.neq,
                "physics_timestep_s": float(m.opt.timestep),
                "wheel_limited": bool(m.jnt_limited[self.wheel_id]),
                "wheel_range_deg": np.rad2deg(m.jnt_range[self.wheel_id]).tolist(),
                "wheel_load": asdict(self.wheel_load),
                "actuator_force_ranges": m.actuator_forcerange.tolist(),
                "collision_limit": "Selected existing collision geometry only; this is not a full collision coverage proof"}

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        errors = []
        for side in range(2):
            indices = self.qidx[7 * side:7 * (side + 1)]
            limits = m.jnt_range[self.jids[7 * side:7 * (side + 1)]]
            seed = np.array([-.65, .3, .1, 1.15, 0, 0, 0])
            if side:
                seed[[1, 2, 4, 6]] *= -1
            position = self.center + self.frame @ np.array([-.002, .163 if not side else -.163, 0])
            def residual(q):
                d.qpos[indices] = q
                mujoco.mj_kinematics(m, d)
                rotation = d.site_xmat[self.grips[side]].reshape(3, 3)
                return np.r_[d.site_xpos[self.grips[side]] - position,
                             .12 * Rotation.from_matrix(self.frame @ rotation.T).as_rotvec(),
                             .0002 * (q - seed)]
            lower, upper = limits[:, 0] + .035, limits[:, 1] - .035
            solution = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                                     max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            d.qpos[indices] = solution.x
            errors.append(float(np.linalg.norm(residual(solution.x)[:3])))
        open_hand = np.tile(self.config["initial_hand_open"], 2)
        closed_hand = np.tile(self.config["initial_hand_closed"], 2)
        self.target = np.r_[d.qpos[self.qidx[:14]], closed_hand]
        d.qpos[self.qidx[14:]] = open_hand
        # Mimic positions are initialized only at reset; runtime uses equalities.
        for eq in range(m.neq):
            if m.eq_type[eq] == int(mujoco.mjtEq.mjEQ_JOINT):
                a, b = int(m.eq_obj1id[eq]), int(m.eq_obj2id[eq])
                if b >= 0:
                    x = d.qpos[m.jnt_qposadr[b]] - m.qpos0[m.jnt_qposadr[b]]
                    d.qpos[m.jnt_qposadr[a]] = m.qpos0[m.jnt_qposadr[a]] + np.polynomial.polynomial.polyval(x, m.eq_data[eq, :5])
        mujoco.mj_forward(m, d)
        steps = round(self.config["settle_seconds"] / m.opt.timestep)
        for step in range(steps):
            fraction = min(1., (step + 1) / max(1, .65 * steps))
            target = self.target.copy()
            target[14:] = open_hand + fraction * (closed_hand - open_hand)
            self.physics_step(target, 0.)
        return errors

    def physics_step(self, target, load):
        m, d = self.model, self.data
        centering = self.wheel_load.centering_torque(float(d.qpos[self.wheel_q]))
        d.qfrc_applied[:] = 0
        d.qfrc_applied[self.wheel_v] = load + centering
        self.last_wheel_load = {"disturbance_torque_applied_nm": float(load),
                               "centering_torque_applied_nm": centering,
                               "total_external_wheel_torque_applied_nm": float(load + centering)}
        # Gravity/bias compensation stays INSIDE each finite actuator force cap.
        d.ctrl[:] = target + d.qfrc_bias[self.vidx] / m.actuator_gainprm[:, 0]
        mujoco.mj_step(m, d)

    def metrics(self, initial_angle):
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        force = np.zeros(2)
        collision_peak = 0.
        vector = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            a, b = m.geom_bodyid[[contact.geom1, contact.geom2]]
            mujoco.mj_contactForce(m, d, i, vector)
            value = float(np.linalg.norm(vector[:3]))
            allowed = False
            if self.wheel_body in (a, b):
                other = b if a == self.wheel_body else a
                for side, bodies in enumerate(self.hand_bodies):
                    if other in bodies:
                        force[side] += value
                        allowed = True
            same_hand = any(a in bodies and b in bodies for bodies in self.hand_bodies)
            if not allowed and not same_hand:
                collision_peak = max(collision_peak, value)
        limits = m.jnt_range[self.jids[:14]]
        q = d.qpos[self.qidx[:14]]
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        forces = d.actuator_force
        force_ok = bool(np.all(forces >= m.actuator_forcerange[:, 0] - 1e-6)
                        and np.all(forces <= m.actuator_forcerange[:, 1] + 1e-6))
        return {"wheel_angle_deg": float(np.rad2deg(d.qpos[self.wheel_q])),
                "wheel_drift_deg": float(np.rad2deg(d.qpos[self.wheel_q] - initial_angle)),
                "wheel_speed_deg_s": float(np.rad2deg(d.qvel[self.wheel_v])),
                "centering_torque_state_nm": self.wheel_load.centering_torque(float(d.qpos[self.wheel_q])),
                "viscous_torque_state_nm": float(-m.dof_damping[self.wheel_v] * d.qvel[self.wheel_v]),
                **self.last_wheel_load,
                "min_arm_margin_deg": float(np.rad2deg(margin)),
                "left_contact_force_norm_sum_n": float(force[0]),
                "right_contact_force_norm_sum_n": float(force[1]),
                "non_hand_collision_peak_n": collision_peak,
                "finite": bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
                "actuator_force_limits_respected": force_ok}


def verified_scene(config):
    scene = ROOT / config["scene"]
    provenance = scene.with_suffix(".provenance.json")
    frozen = json.loads(provenance.read_text())
    if digest(scene) != frozen["snapshot_sha256"]:
        raise ValueError("Scene changed after snapshot; create a new version")
    for name, expected in frozen["mesh_sha256"].items():
        if digest(Path(frozen["mesh_directory"]) / name) != expected:
            raise ValueError(f"External mesh changed after snapshot: {name}")
    return scene, frozen


def run(config_path, output):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    scene, frozen = verified_scene(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    fixture = Fixture(scene, config)
    structure = fixture.structural_report()
    report = {"kind": "scripted_fixture_audit_not_RL", "interface_version": INTERFACE_VERSION,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python_version": platform.python_version(), "mujoco_version": mujoco.__version__,
              "config_sha256": digest(config_path), "scene_provenance": frozen,
              "observation_contract_dim": OBSERVATION_DIM, "action_contract_dim": ACTION_DIM,
              "structure": structure, "cases": [], "training_started": False,
              "single_hand_verified": False, "handover_verified": False}
    write_json(output / "summary.json", report)

    if not structure["passed"]:
        raise RuntimeError("Structural audit failed; inspect summary.json before physical probes")
    for index, torque in enumerate(config["probe_torques_nm"]):
        errors = fixture.reset()
        initial = float(fixture.data.qpos[fixture.wheel_q])
        initial_metrics = fixture.metrics(initial)
        rows = []
        substep_collision_peak = initial_metrics["non_hand_collision_peak_n"]
        substep_min_margin = initial_metrics["min_arm_margin_deg"]
        finite = initial_metrics["finite"]
        force_ok = initial_metrics["actuator_force_limits_respected"]
        for step in range(round(config["probe_seconds"] * config["policy_hz"])):
            load = 0.
            for sub in range(fixture.substeps):
                t = (step * fixture.substeps + sub + 1) * fixture.model.opt.timestep
                load = torque * min(1., t / config["load_ramp_seconds"])
                fixture.physics_step(fixture.target, load)
                metrics = fixture.metrics(initial)
                substep_collision_peak = max(substep_collision_peak, metrics["non_hand_collision_peak_n"])
                substep_min_margin = min(substep_min_margin, metrics["min_arm_margin_deg"])
                finite = finite and metrics["finite"]
                force_ok = force_ok and metrics["actuator_force_limits_respected"]
            rows.append({"time_s": (step + 1) / config["policy_hz"], "load_nm": load, **metrics})
        filename = f"hold_{index:02d}_{torque:+.2f}Nm.csv"
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        gates = config["gates"]
        peak_drift = max(abs(row["wheel_drift_deg"]) for row in rows)
        checks = {"initial_ik": max(errors) <= gates["max_initial_ik_position_error_m"],
                  "finite": finite, "actuator_force_limits": force_ok,
                  "wheel_drift": peak_drift <= gates["max_wheel_drift_deg"],
                  "arm_margin": substep_min_margin >= gates["min_arm_margin_deg"],
                  "collision": substep_collision_peak <= gates["max_non_hand_collision_force_n"],
                  "both_hands_contact_at_start": min(initial_metrics["left_contact_force_norm_sum_n"], initial_metrics["right_contact_force_norm_sum_n"]) > .2}
        case = {"support": "both", "load_nm": torque, "initial_ik_errors_m": errors,
                "initial_metrics": initial_metrics, "max_abs_wheel_drift_deg": peak_drift,
                "min_arm_margin_deg": substep_min_margin, "max_collision_force_n": substep_collision_peak,
                "checks": checks, "passed": all(checks.values()), "trace": filename}
        report["cases"].append(case)
        write_json(output / "summary.json", report)
        print(json.dumps(case, ensure_ascii=False), flush=True)
    report["fixture_gate_passed"] = all(case["passed"] for case in report["cases"])
    report["ready_for_PPO"] = False
    report["pending"] = ["single-hand support", "26D execution interface and 152D observation builder",
                         "reference trajectories and no-contact controls", "network/rollout/PPO implementation"]
    write_json(output / "summary.json", report)


def wheel_check(config_path, output):
    """No hand-wheel contact: release a displaced wheel, plus a spring-off control."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    scene, frozen = verified_scene(config)
    load = WheelLoadConfig.from_config(config)
    if load.centering_stiffness_nm_per_rad <= 0:
        raise ValueError("wheel-check requires enabled centering")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config.yaml")
    fixture = Fixture(scene, config)
    m, d = fixture.model, fixture.data
    wheel_geoms = np.flatnonzero(m.geom_bodyid == fixture.wheel_body)
    m.geom_contype[wheel_geoms] = 0
    m.geom_conaffinity[wheel_geoms] = 0
    cases = []
    report = {"kind": "scripted_no_contact_centering_diagnostic_not_RL",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python_version": platform.python_version(), "mujoco_version": mujoco.__version__,
              "config_sha256": digest(config_path), "scene_provenance": frozen,
              "wheel_load": asdict(load), "duration_s": 5.0, "cases": cases,
              "hand_wheel_contacts_disabled": True}
    for index, (angle_deg, spring_on) in enumerate(((90., True), (-90., True),
                                                  (360., True), (-360., True), (90., False))):
        if not m.jnt_range[fixture.wheel_id, 0] < np.deg2rad(angle_deg) < m.jnt_range[fixture.wheel_id, 1]:
            raise ValueError("Probe angle outside physical wheel range")
        fixture.wheel_load = load if spring_on else WheelLoadConfig(**{
            **asdict(load), "centering_stiffness_nm_per_rad": 0.0})
        fixture.reset()
        # Test reset only: place a stationary wheel at the requested displacement.
        d.qpos[fixture.wheel_q] = np.deg2rad(angle_deg)
        d.qvel[fixture.wheel_v] = 0.
        mujoco.mj_forward(m, d)
        initial = float(d.qpos[fixture.wheel_q])
        initial_torque = fixture.wheel_load.centering_torque(initial)
        rows = []
        finite = True
        for step in range(round(5 * config["policy_hz"])):
            for _ in range(fixture.substeps):
                fixture.physics_step(fixture.target, 0.)
                finite = finite and bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            metrics = fixture.metrics(initial)
            rows.append({"time_s": (step + 1) / config["policy_hz"], **metrics})
        filename = f"free_wheel_{index:02d}_{angle_deg:+.0f}deg_spring_{int(spring_on)}.csv"
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        final = rows[-1]["wheel_angle_deg"]
        center_deg = load.center_angle_deg
        peak_contact = max(max(row["left_contact_force_norm_sum_n"], row["right_contact_force_norm_sum_n"]) for row in rows)
        peak_drift = max(abs(row["wheel_drift_deg"]) for row in rows)
        checks = {"finite": finite, "no_hand_wheel_contact": peak_contact == 0.,
                  "no_disturbance": all(row["disturbance_torque_applied_nm"] == 0 for row in rows)}
        if spring_on:
            checks["initial_torque_toward_fixed_center"] = (angle_deg - center_deg) * initial_torque < 0
            checks["returned_toward_center"] = abs(final - center_deg) < .2 * abs(angle_deg - center_deg)
            checks["bounded_centering_torque"] = max(abs(row["centering_torque_applied_nm"]) for row in rows) <= load.centering_torque_limit_nm
        else:
            checks["spring_off_wheel_stays_displaced"] = peak_drift < 1e-6
        case = {"initial_angle_deg": angle_deg, "spring_enabled": spring_on,
                "initial_centering_torque_nm": initial_torque, "final_angle_deg": final,
                "max_abs_drift_deg": peak_drift, "peak_hand_wheel_contact_n": peak_contact,
                "checks": checks, "passed": all(checks.values()), "trace": filename}
        cases.append(case)
        write_json(output / "summary.json", report)
        print(json.dumps(case), flush=True)
    report["passed"] = all(case["passed"] for case in cases)
    report["limits"] = "Equivalent passive load, not a vehicle/tire model. Dry friction permits residual offset. No learned policy or hand-release skill."
    write_json(output / "summary.json", report)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("--source", type=Path, required=True)
    audit = commands.add_parser("run")
    audit.add_argument("--config", type=Path, default=ROOT / "configs/s00.yaml")
    audit.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("wheel-check")
    check.add_argument("--config", type=Path, default=ROOT / "configs/s00.yaml")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source)
    elif args.command == "run":
        run(args.config, args.output)
    else:
        wheel_check(args.config, args.output)


if __name__ == "__main__":
    main()
