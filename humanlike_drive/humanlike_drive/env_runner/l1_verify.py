"""Independent saved-contact CAD audit and recorded-command physics replay."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def source_evidence(run, name):
    source = Path(__file__).read_bytes()
    filename = f"{name}_verifier.py"
    (Path(run) / filename).write_bytes(source)
    return dict(
        verifier_sha256=hashlib.sha256(source).hexdigest(), verifier_source=filename
    )


def model_audit(run):
    from humanlike_drive.env.logitech_scene import SCENE
    from humanlike_drive.env.palm_collision import ASSETS

    run = Path(run)
    reference = mujoco.MjModel.from_xml_path(str(SCENE))
    actual = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    checks = {
        "same_state_and_drive_dimensions": (
            actual.nq,
            actual.nv,
            actual.nu,
            actual.nbody,
        )
        == (reference.nq, reference.nv, reference.nu, reference.nbody)
    }
    errors = {}
    for key in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_range",
        "jnt_axis",
        "dof_frictionloss",
        "dof_armature",
        "dof_damping",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_biasprm",
        "eq_data",
    ):
        difference = float(np.max(abs(getattr(actual, key) - getattr(reference, key))))
        errors[key] = difference
        checks[key + "_unchanged"] = difference < 1e-12
    wheel = actual.joint("wheel_joint")
    checks["26_finite_robot_motors_passive_wheel"] = bool(
        actual.nu == 26
        and wheel.id not in actual.actuator_trnid[:, 0]
        and np.isfinite(actual.actuator_forcerange).all()
        and (actual.actuator_forcerange[:, 1] > 0).all()
    )
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    mismatches = []
    for relative, expected in manifest["sha256"].items():
        if hashlib.sha256((ASSETS / relative).read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
    for filename, expected in manifest["sources"].items():
        source = ASSETS.parent / "meshes" / Path(filename).name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            mismatches.append(filename)
    checks["palm_assets_match_original_CAD_and_piece_hashes"] = not mismatches
    result = dict(
        passed=all(checks.values()),
        checks=checks,
        max_abs_changes=errors,
        asset_mismatches=mismatches,
        wheel_limits_deg=np.rad2deg(wheel.range).tolist(),
        **source_evidence(run, "model_invariants"),
        scope="Compared with the preserved original scene. Intended changes are palm collision decomposition, collision masks/documented assembly exclusions and solver settings; they do not change inertia, kinematics, mimic or finite drive limits.",
    )
    (run / "model_invariants.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def body_mesh(m, d, geom_name):
    g = m.geom(geom_name).id
    mid = m.geom_dataid[g]
    va = m.mesh_vertadr[mid]
    vn = m.mesh_vertnum[mid]
    fa = m.mesh_faceadr[mid]
    fn = m.mesh_facenum[mid]
    bid = m.geom_bodyid[g]
    verts = (
        m.mesh_vert[va : va + vn] @ d.geom_xmat[g].reshape(3, 3).T
        + d.geom_xpos[g]
        - d.xpos[bid]
    ) @ d.xmat[bid].reshape(3, 3)
    return trimesh.Trimesh(
        vertices=verts, faces=m.mesh_face[fa : fa + fn], process=False
    )


def nearest_exact(mesh, points):
    if not len(points):
        return np.empty(0)
    # Exhaustive triangles in small batches: no rtree dependency or approximate
    # nearest-neighbor shortcut, bounded peak memory for detailed vendor CAD.
    chunks = np.array_split(points, max(1, int(np.ceil(len(points) / 4))))
    return np.concatenate(
        [trimesh.proximity.closest_point_naive(mesh, p)[1] for p in chunks]
    )


def cad_audit(run, samples=250):
    run = Path(run)
    m = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    a = np.load(run / "trajectory.npz")
    c = np.load(run / "contacts.npz")["contacts"]
    result = dict(
        scope="stratified deterministic actual loaded contact samples versus original visual CAD; not a global Hausdorff bound",
        samples_per_palm=samples,
        **source_evidence(run, "contact_CAD_audit"),
    )
    selected = []
    for side, name in enumerate(("left", "right")):
        rows = c[(c[:, 4] == side) & (c[:, 5] == 0) & (c[:, 19] > 0.05)]
        if not len(rows):
            result[name] = dict(samples=0)
            continue
        rows = rows[np.linspace(0, len(rows) - 1, min(samples, len(rows))).astype(int)]
        selected.append(rows)
        dist = nearest_exact(
            body_mesh(m, d, name + "_wrist_yaw_link_geom_1"), rows[:, 25:28]
        )
        result[name] = dict(
            samples=len(rows),
            max_mm=float(dist.max() * 1000),
            p95_mm=float(np.quantile(dist, 0.95) * 1000),
            mean_mm=float(dist.mean() * 1000),
            local_palm_bounds_m=[
                rows[:, 25:28].min(axis=0).tolist(),
                rows[:, 25:28].max(axis=0).tolist(),
            ],
        )
    if selected:
        rows = np.concatenate(selected)
        steps = rows[:, 0].astype(int)
        body = d.body("g29_wheel")
        axis = d.site("wheel_frame").xmat.reshape(3, 3)[:, 0]
        points = Rotation.from_rotvec(-a["wheel_angle"][steps, None] * axis).apply(
            rows[:, 7:10] - body.xpos
        ) @ body.xmat.reshape(3, 3)
        dist = nearest_exact(body_mesh(m, d, "wheel_visual"), points)
        result["wheel"] = dict(
            samples=len(rows),
            max_mm=float(dist.max() * 1000),
            p95_mm=float(np.quantile(dist, 0.95) * 1000),
            mean_mm=float(dist.mean() * 1000),
        )
        result["passed"] = bool(
            all(
                v.get("max_mm", 0) <= 0.5
                for v in (
                    result.get("left", {}),
                    result.get("right", {}),
                    result["wheel"],
                )
            )
        )
    else:
        result["passed"] = False
    (run / "contact_CAD_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def replay_check(run, seconds=12.0, interface="controls"):
    run = Path(run)
    m = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    d = mujoco.MjData(m)
    archive = np.load(run / "trajectory.npz")
    keys = ["time", "qpos", "qvel", "ctrl", "qfrc_applied"]
    if interface == "actions":
        keys.append("action")
    a = {key: archive[key] for key in keys}
    archive.close()
    d.qpos[:] = a["qpos"][0]
    d.qvel[:] = a["qvel"][0]
    peak_q = 0.0
    peak_v = 0.0
    peak_ctrl = 0.0
    if interface == "actions":
        from humanlike_drive.controller.p0_control import RateExecutor
        from humanlike_drive.env.logitech_scene import ARM_NAMES
        from humanlike_drive.env_runner.physical_feasibility import apply_forces

        config = json.loads((run / "config.json").read_text())
        if config["command"] != "run":
            raise ValueError(
                "Action replay is for task runs; use controls for preparation"
            )
        grasp = json.loads((run / "grasp_used.json").read_text())
        d.ctrl[:] = grasp["ctrl"]
        executor = RateExecutor(m, d)
        period = round(executor.dt / m.opt.timestep)
        arm_dofs = [m.joint(name).dofadr[0] for name in ARM_NAMES]
        wheel_dof = m.joint("wheel_joint").dofadr[0]
        load = config.get(
            "constant_load_Nm", config.get("load_model", {}).get("constant_Nm")
        )
    n = min(len(a["time"]) - 1, round(seconds / m.opt.timestep))
    for k in range(n):
        if interface == "actions":
            if k % period == 0:
                ramp_start = d.ctrl.copy()
                ramp_target = executor.step(a["action"][k], d.qpos[executor.qids])
            fraction = (k % period + 1) / period
            d.ctrl[:] = (1 - fraction) * ramp_start + fraction * ramp_target
            apply_forces(
                m,
                d,
                arm_dofs,
                wheel_dof,
                load - config["load_damping_Nm_s"] * d.qvel[wheel_dof],
            )
        else:
            d.ctrl[:] = a["ctrl"][k]
            d.qfrc_applied[:] = a["qfrc_applied"][k]
            mujoco.mj_forward(m, d)
        peak_ctrl = max(peak_ctrl, float(abs(d.ctrl - a["ctrl"][k]).max()))
        mujoco.mj_step(m, d)
        peak_q = max(peak_q, float(abs(d.qpos - a["qpos"][k + 1]).max()))
        peak_v = max(peak_v, float(abs(d.qvel - a["qvel"][k + 1]).max()))
    result = dict(
        seconds=n * m.opt.timestep,
        steps=n,
        max_qpos_error_rad=peak_q,
        max_qvel_error_rad_s=peak_v,
        max_ctrl_error_rad=peak_ctrl,
        interface=interface,
        **source_evidence(
            run,
            (
                "action_replay_check"
                if interface == "actions"
                else "dynamics_replay_check"
            ),
        ),
        passed=bool(peak_q < 1e-5 and peak_v < 1e-3 and peak_ctrl < 1e-6),
        scope=(
            "20 Hz normalized actions through live executor, 500 Hz interpolation, live bounded bias compensation and synthetic load"
            if interface == "actions"
            else "reexecuted recorded finite motor controls and applied bias/load; no intermediate state injection"
        ),
    )
    filename = (
        "action_replay_check.json"
        if interface == "actions"
        else "dynamics_replay_check.json"
    )
    (run / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["cad", "replay", "model"])
    p.add_argument("--run", required=True)
    p.add_argument("--samples", type=int, default=250)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--interface", choices=["controls", "actions"], default="controls")
    args = p.parse_args()
    if args.command == "cad":
        cad_audit(args.run, args.samples)
    elif args.command == "model":
        model_audit(args.run)
    else:
        replay_check(args.run, args.seconds, args.interface)


if __name__ == "__main__":
    main()
