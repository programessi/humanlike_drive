"""S00-B scripted single-hand support, motion, release and bounded initial jitter.

Default probes use time-only joint targets after a finite reset preparation jig.
The explicit --feedback option adds a separately reported bounded script baseline.
All robot commands execute through capped PD actuators; no policy is trained here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from .audit import Fixture, ROOT, digest, verified_scene, write_json
from .clearance import Clearance


def blend(x):
    x = np.clip(x, 0., 1.)
    return x ** 3 * (10. + x * (-15. + 6. * x))


def motion_phase(time_s, amplitude_deg, segment_s, pause_s):
    """Time-only smooth trajectory: 0,+A,0,-A,0, with endpoint pauses."""
    points = (0., amplitude_deg, 0., -amplitude_deg, 0.)
    for start, end in zip(points[:-1], points[1:]):
        if time_s <= segment_s:
            return float(start + (end - start) * blend(time_s / segment_s))
        time_s -= segment_s
        if time_s <= pause_s:
            return float(end)
        time_s -= pause_s
    return 0.


class SupportFixture(Fixture):
    def __init__(self, scene, config):
        super().__init__(scene, config)
        # Use the existing static pregrasp solve, without its runtime controller.
        super().reset()
        self.nominal_arm = self.target[:14].copy()
        self.kin = mujoco.MjData(self.model)
        self.clearance = Clearance(self)
        self.parks = []
        self.park_diagnostics = []
        for side, name in enumerate(("left", "right")):
            position = self.center + self.frame @ np.asarray(config[f"park_local_{name}_m"])
            q, info = self.solve_pose(side, position, self.frame, self.nominal_arm[7*side:7*side+7], 5.)
            self.parks.append(q)
            self.park_diagnostics.append(info)
        self.last_jig_torque = 0.

    def solve_pose(self, side, position, rotation, seed, margin_deg=2.):
        m, d = self.model, self.kin
        indices = self.qidx[7*side:7*side+7]
        limits = m.jnt_range[self.jids[7*side:7*side+7]]
        d.qpos[:] = self.data.qpos
        def residual(q):
            d.qpos[indices] = q
            mujoco.mj_kinematics(m, d)
            current = d.site_xmat[self.grips[side]].reshape(3, 3)
            return np.r_[d.site_xpos[self.grips[side]] - position,
                         .12 * Rotation.from_matrix(rotation @ current.T).as_rotvec(), .0002 * (q - seed)]
        lower, upper = limits[:, 0] + np.deg2rad(margin_deg), limits[:, 1] - np.deg2rad(margin_deg)
        solution = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                                 max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = residual(solution.x)
        info = {"position_error_m": float(np.linalg.norm(error[:3])),
                "rotation_error_deg": float(np.rad2deg(np.linalg.norm(error[3:6]) / .12))}
        if not solution.success or info["position_error_m"] > .002 or info["rotation_error_deg"] > .5:
            raise ValueError(f"IK failed: {info}")
        return solution.x.copy(), info

    def set_mimic(self, data):
        m = self.model
        for eq in range(m.neq):
            if m.eq_type[eq] == int(mujoco.mjtEq.mjEQ_JOINT):
                a, b = int(m.eq_obj1id[eq]), int(m.eq_obj2id[eq])
                x = data.qpos[m.jnt_qposadr[b]] - m.qpos0[m.jnt_qposadr[b]]
                data.qpos[m.jnt_qposadr[a]] = m.qpos0[m.jnt_qposadr[a]] + np.polynomial.polynomial.polyval(x, m.eq_data[eq, :5])

    def park_check(self, free_side, q):
        m, d = self.model, self.kin
        d.qpos[:] = m.qpos0
        d.qpos[self.qidx[:14]] = self.nominal_arm
        d.qpos[self.qidx[7*free_side:7*free_side+7]] = q
        hands = np.tile(self.config["initial_hand_closed"], 2)
        hands[6*free_side:6*free_side+6] = self.config["initial_hand_open"]
        d.qpos[self.qidx[14:]] = hands
        self.set_mimic(d)
        mujoco.mj_forward(m, d)
        limits = m.jnt_range[self.jids[:14]]
        arms = d.qpos[self.qidx[:14]]
        values = self.clearance.read(d, free_side)
        values["min_arm_margin_deg"] = float(np.rad2deg(np.min(np.minimum(arms - limits[:, 0], limits[:, 1] - arms))))
        gates = self.config["gates"]
        values["passed"] = bool(values["free_hand_wheel_gap_m"] >= gates["min_free_hand_wheel_gap_m"]
                                and values["free_assembly_other_gap_m"] >= gates["min_free_assembly_other_gap_m"]
                                and values["min_arm_margin_deg"] >= gates["min_initial_arm_margin_deg"])
        return values

    def prepare(self, support, angle_deg, both=False, park=None):
        m, d, cfg = self.model, self.data, self.config
        free = 1 - support
        park = self.parks[free] if park is None else park
        mujoco.mj_resetData(m, d)
        d.qpos[self.wheel_q] = np.deg2rad(angle_deg)
        self.target = np.r_[self.nominal_arm, np.tile(cfg["initial_hand_closed"], 2)]
        if not both:
            self.target[7*free:7*free+7] = park
            self.target[14+6*free:20+6*free] = cfg["initial_hand_open"]
        d.qpos[self.qidx[:14]] = self.target[:14]
        d.qpos[self.qidx[14:]] = np.tile(cfg["initial_hand_open"], 2)
        self.set_mimic(d)
        mujoco.mj_forward(m, d)
        peak_jig = 0.
        steps = round(cfg["settle_seconds"] / m.opt.timestep)
        start = np.tile(cfg["initial_hand_open"], 2)
        jig = cfg["preparation_jig"]
        for step in range(steps):
            target = self.target.copy()
            target[14:] = start + blend((step + 1) / max(1., .65 * steps)) * (self.target[14:] - start)
            error = np.deg2rad(angle_deg) - d.qpos[self.wheel_q]
            setup = (-self.wheel_load.centering_torque(float(d.qpos[self.wheel_q]))
                     + jig["kp_nm_per_rad"] * error - jig["kd_nm_s_per_rad"] * d.qvel[self.wheel_v])
            setup = float(np.clip(setup, -jig["torque_limit_nm"], jig["torque_limit_nm"]))
            peak_jig = max(peak_jig, abs(setup))
            self.physics_step(target, setup)
        self.last_jig_torque = setup
        # No state edits follow this point. The next step removes the jig entirely.
        initial = float(d.qpos[self.wheel_q])
        metrics = self.read(support, initial)
        metrics["preparation_jig_active"] = True
        return {"requested_angle_deg": angle_deg, "actual_angle_deg": float(np.rad2deg(initial)),
                "angle_error_deg": float(np.rad2deg(initial) - angle_deg),
                "peak_preparation_jig_torque_nm": peak_jig,
                "jig_torque_before_removal_nm": setup, "jig_removed_for_entire_probe": True,
                "initial_metrics": metrics}

    def release_preflight(self, free_side, park):
        """Sample the intended open-hand joint path at <=0.005 rad joint spacing."""
        m, d = self.model, self.kin
        start = self.nominal_arm[7*free_side:7*free_side+7]
        samples = max(2, int(np.ceil(np.max(np.abs(park-start))/.005))+1)
        min_gap, min_margin = .25, float("inf")
        for fraction in np.linspace(0,1,samples):
            d.qpos[:] = self.data.qpos
            d.qpos[self.qidx[:14]] = self.nominal_arm
            d.qpos[self.qidx[7*free_side:7*free_side+7]] = start+fraction*(park-start)
            d.qpos[self.qidx[14+6*free_side:20+6*free_side]] = self.config["initial_hand_open"]
            self.set_mimic(d)
            mujoco.mj_forward(m,d)
            min_gap = min(min_gap,self.clearance.minimum(d,self.clearance.other[free_side]))
            limits = m.jnt_range[self.jids[:14]]
            q = d.qpos[self.qidx[:14]]
            min_margin = min(min_margin,float(np.rad2deg(np.min(np.minimum(q-limits[:,0],limits[:,1]-q)))))
        g=self.config["gates"]
        return {"samples":samples,"max_joint_spacing_rad":.005,
                "min_other_body_gap_m":min_gap,"min_arm_margin_deg":min_margin,
                "passed":min_gap>=g["min_free_assembly_other_gap_m"] and min_margin>=g["min_initial_arm_margin_deg"],
                "scope":"Existing guards, open-hand target path; wheel contact allowed during release. Actual motion checked separately."}

    def read(self, support, initial):
        values = self.metrics(initial)
        values.update(self.clearance.read(self.data, 1-support))
        torques = np.zeros(2)
        vector = np.zeros(6)
        m, d = self.model, self.data
        for i in range(d.ncon):
            contact = d.contact[i]
            bodies = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.wheel_body not in bodies:
                continue
            other = bodies[1] if bodies[0] == self.wheel_body else bodies[0]
            for side, group in enumerate(self.hand_bodies):
                if other not in group:
                    continue
                mujoco.mj_contactForce(m, d, i, vector)
                rotation = contact.frame.reshape(3, 3).T
                sign = -1 if bodies[0] == self.wheel_body else 1
                force = sign * (rotation @ vector[:3])
                moment = sign * (rotation @ vector[3:])
                torque = np.cross(contact.pos - d.xpos[self.wheel_body], force) + moment
                torques[side] += np.dot(torque, d.xaxis[self.wheel_id])
        values.update(left_wheel_contact_torque_nm=float(torques[0]),
                      right_wheel_contact_torque_nm=float(torques[1]),
                      preparation_jig_active=False)
        return values


def summarize(rows, preparation, cfg, support, kind):
    g = cfg["gates"]
    support_name, free_name = ("left", "right") if support == 0 else ("right", "left")
    initial = preparation["initial_metrics"]
    after = [r for r in rows if r["phase"] == "post_release"] if kind == "release" else rows
    after = after or rows
    errors = np.array([r["tracking_error_deg"] for r in rows])
    after_errors = np.array([r["tracking_error_deg"] for r in after])
    contact_fraction = float(np.mean([r["interval_support_contact_fraction"] for r in rows]))
    min_gap = min(r["free_hand_wheel_gap_m"] for r in after)
    other_gap = min(r["free_assembly_other_gap_m"] for r in rows)
    max_free_contact = max(r[f"{free_name}_contact_force_norm_sum_n"] for r in after)
    checks = {
        "initial_angle": abs(preparation["angle_error_deg"]) <= g["max_initial_angle_error_deg"],
        "initial_speed": abs(initial["wheel_speed_deg_s"]) <= g["max_initial_speed_deg_s"],
        "initial_arm_margin": initial["min_arm_margin_deg"] >= g["min_initial_arm_margin_deg"],
        "initial_support_contact": initial[f"{support_name}_contact_force_norm_sum_n"] > g["min_support_contact_n"],
        "runtime_arm_margin": min(r["min_arm_margin_deg"] for r in rows) >= g["min_runtime_arm_margin_deg"],
        "free_hand_off_wheel": min_gap >= g["min_free_hand_wheel_gap_m"] and max_free_contact <= g["max_free_hand_contact_n"],
        "other_body_clearance": other_gap >= g["min_free_assembly_other_gap_m"],
        "support_contact": contact_fraction >= g["min_support_contact_fraction"],
        "no_non_hand_collision": max(r["non_hand_collision_peak_n"] for r in rows) <= g["max_non_hand_collision_force_n"],
        "actuator_limits": all(r["actuator_force_limits_respected"] for r in rows),
        "finite": all(r["finite"] for r in rows),
        "no_preparation_assistance": all(r["disturbance_torque_applied_nm"] == 0 for r in rows),
        "target_speed": max(r["arm_target_speed_rad_s"] for r in rows) <= g["max_arm_target_speed_rad_s"] + 1e-6,
    }
    if kind == "motion":
        checks["tracking_rmse"] = float(np.sqrt(np.mean(errors**2))) <= g["max_motion_rmse_deg"]
    else:
        checks["hold_drift"] = max(r["tracking_error_peak_deg"] for r in after) <= g["max_hold_drift_deg"]
    if kind == "release":
        checks["starts_with_both_hands_contact"] = initial[f"{free_name}_contact_force_norm_sum_n"] > g["min_support_contact_n"]
        checks["post_release_window"] = len(after) >= round(cfg["hold_seconds"]*cfg["policy_hz"])
        checks["release_error"] = max(r["tracking_error_peak_deg"] for r in rows) <= g["max_release_error_deg"]
    return {"checks": checks, "passed": all(checks.values()),
            "tracking_rmse_deg": float(np.sqrt(np.mean(errors**2))),
            "max_abs_error_deg": max(r["tracking_error_peak_deg"] for r in rows),
            "max_post_release_error_deg": max(r["tracking_error_peak_deg"] for r in after),
            "min_free_hand_wheel_gap_m": min_gap, "min_free_assembly_other_gap_m": other_gap,
            "max_free_hand_contact_n_after_release": max_free_contact,
            "support_contact_fraction": contact_fraction,
            "min_arm_margin_deg": min(r["min_arm_margin_deg"] for r in rows),
            "max_non_hand_collision_n": max(r["non_hand_collision_peak_n"] for r in rows),
            "max_arm_target_speed_rad_s": max(r["arm_target_speed_rad_s"] for r in rows)}


def execute(fixture, support, angle, kind, cfg, path, park=None, feedback=False):
    import csv
    free = 1-support
    park = fixture.parks[free] if park is None else park
    preparation = fixture.prepare(support, angle, both=kind == "release", park=park)
    np.savez_compressed(path.with_suffix(".initial.npz"), qpos=fixture.data.qpos.copy(),
                        qvel=fixture.data.qvel.copy(), act=fixture.data.act.copy(),
                        qacc_warmstart=fixture.data.qacc_warmstart.copy(), target=fixture.target.copy(),
                        support=np.array(support), requested_angle_deg=np.array(angle))
    initial = float(fixture.data.qpos[fixture.wheel_q])
    base = fixture.target.copy()
    target = base.copy()
    previous = target.copy()
    local_data = mujoco.MjData(fixture.model)
    local_data.qpos[:] = fixture.data.qpos
    local_data.qpos[fixture.qidx] = base
    fixture.set_mimic(local_data)
    mujoco.mj_forward(fixture.model, local_data)
    base_position = local_data.site_xpos[fixture.grips[support]].copy()
    base_rotation = local_data.site_xmat[fixture.grips[support]].reshape(3,3).copy()
    segment = cfg["motion_segment_seconds"]
    pause = cfg["motion_pause_seconds"]
    duration = cfg["hold_seconds"]
    if kind == "motion":
        duration = 4*(segment+pause)
    if kind == "release":
        duration += cfg["release_open_seconds"] + cfg["release_move_seconds"]
    rows = []
    last_ik = base[7*support:7*support+7].copy()
    correction = 0.
    for step in range(round(duration * cfg["policy_hz"])):
        t = (step+1)/cfg["policy_hz"]
        phase_deg = 0.
        label = kind
        if kind == "motion":
            phase_deg = motion_phase(t, cfg["motion_amplitude_deg"], segment, pause)
        elif kind == "release":
            open_s, move_s = cfg["release_open_seconds"], cfg["release_move_seconds"]
            fraction = blend(t/open_s)
            hs = slice(14+6*free,20+6*free)
            target[hs] = base[hs] + fraction*(np.asarray(cfg["initial_hand_open"])-base[hs])
            arm_fraction = blend((t-open_s)/move_s)
            arms = slice(7*free,7*free+7)
            target[arms] = base[arms]+arm_fraction*(park-base[arms])
            label = "opening" if t <= open_s else "retreat" if t < open_s+move_s else "post_release"
        if feedback:
            # Explicit diagnostic baseline ONLY; never part of the future policy executor.
            observed_phase = np.rad2deg(fixture.data.qpos[fixture.wheel_q]-initial)
            wanted = float(np.clip(3.*(phase_deg-observed_phase),-8.,8.))
            correction += float(np.clip(wanted-correction,-4./cfg["policy_hz"],4./cfg["policy_hz"]))
        if kind == "motion" or feedback:
            command_phase = phase_deg+correction
            r = fixture.frame @ Rotation.from_rotvec([np.deg2rad(command_phase),0,0]).as_matrix() @ fixture.frame.T
            position = fixture.center + r @ (base_position-fixture.center)
            last_ik, _ = fixture.solve_pose(support, position, r @ base_rotation, last_ik)
            target[7*support:7*support+7] = last_ik
        speed = float(np.max(np.abs(target[:14]-previous[:14]))*cfg["policy_hz"])
        interval = []
        for sub in range(fixture.substeps):
            command = previous + (sub+1)/fixture.substeps*(target-previous)
            fixture.physics_step(command, 0.)
            metrics = fixture.read(support, initial)
            sample_t = (step*fixture.substeps+sub+1)*fixture.model.opt.timestep
            desired = motion_phase(sample_t, cfg["motion_amplitude_deg"], segment, pause) if kind == "motion" else 0.
            metrics.update(time_s=sample_t, phase=label,
                           reference_angle_deg=float(np.rad2deg(initial)+desired),
                           tracking_error_deg=float(metrics["wheel_angle_deg"]-np.rad2deg(initial)-desired),
                           wrist_reference_phase_deg=phase_deg+correction,
                           feedback_correction_deg=correction,
                           arm_target_speed_rad_s=speed)
            interval.append(metrics)
        previous[:] = target
        # Preserve substep extrema in each 20Hz record, not only endpoint values.
        row = interval[-1].copy()
        for key in ("min_arm_margin_deg", "free_hand_wheel_gap_m", "free_assembly_other_gap_m"):
            row[key] = min(x[key] for x in interval)
        for key in ("non_hand_collision_peak_n", "left_contact_force_norm_sum_n", "right_contact_force_norm_sum_n"):
            row[key] = max(x[key] for x in interval)
        row["tracking_error_peak_deg"] = max(abs(x["tracking_error_deg"]) for x in interval)
        support_name=("left","right")[support]
        row["interval_support_contact_fraction"] = float(np.mean([
            x[f"{support_name}_contact_force_norm_sum_n"]>cfg["gates"]["min_support_contact_n"] for x in interval]))
        row["finite"] = all(x["finite"] for x in interval)
        row["actuator_force_limits_respected"] = all(x["actuator_force_limits_respected"] for x in interval)
        rows.append(row)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    result = {"support": ("left","right")[support], "requested_angle_deg": angle, "kind": kind,
              "controller_mode":"bounded_wheel_feedback_diagnostic" if feedback else "open_loop",
              "initial_state":path.with_suffix(".initial.npz").name,
              "preparation": preparation, "trace": path.name, **summarize(rows, preparation, cfg, support, kind)}
    return result


def run(config_path, output, stages, feedback=False, nominal_report=None):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    scene, provenance = verified_scene(cfg)
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(config_path,output/"config.yaml")
    fixture = SupportFixture(scene,cfg)
    report = {"kind":"scripted_single_hand_physical_diagnostic_not_RL", "created_at_utc":datetime.now(timezone.utc).isoformat(),
              "python_version":platform.python_version(),"mujoco_version":mujoco.__version__,
              "config_sha256":digest(config_path), "scene_provenance":provenance,
              "wheel_load":asdict(fixture.wheel_load),"stages":stages,"parks":[],"cases":[],
              "controller_mode":"bounded_wheel_feedback_diagnostic" if feedback else "open_loop",
              "feedback_parameters":{"gain":3.,"correction_limit_deg":8.,"correction_rate_deg_s":4.} if feedback else None,
              "geometry_sampling_hz":1/fixture.model.opt.timestep,
              "limits":"Existing enabled collision shapes only; no complete CAD coverage or learned policy claims."}
    for side in range(2):
        check=fixture.park_check(side,fixture.parks[side])
        path_check=fixture.release_preflight(side,fixture.parks[side])
        report["parks"].append({"side":("left","right")[side],"arm_q_rad":fixture.parks[side].tolist(),
                                "ik":fixture.park_diagnostics[side],"geometry":check,"release_path":path_check})
    write_json(output/"summary.json",report)
    if not all(p["geometry"]["passed"] and p["release_path"]["passed"] for p in report["parks"]):
        raise ValueError("Nominal park rejected; inspect summary.json")
    for kind in ("hold","motion","release"):
        if kind not in stages:
            continue
        for support in range(2):
            for angle in cfg["angles_deg"]:
                name=f"{kind}_{('left','right')[support]}_{angle:+.0f}deg.csv"
                case=execute(fixture,support,angle,kind,cfg,output/name,feedback=feedback)
                report["cases"].append(case);write_json(output/"summary.json",report)
                print(json.dumps({k:v for k,v in case.items() if k!="preparation"}),flush=True)
    if "jitter" in stages:
        nominal = [c for c in report["cases"] if c["kind"] == "hold"]
        if nominal_report is not None:
            evidence=json.loads(Path(nominal_report).read_text())
            if evidence["config_sha256"]!=digest(config_path) or evidence["scene_provenance"]["snapshot_sha256"]!=provenance["snapshot_sha256"]:
                raise ValueError("Nominal evidence must use identical config and scene")
            nominal=[c for c in evidence["cases"] if c["kind"]=="hold"]
            report["nominal_evidence"]={"path":str(Path(nominal_report).resolve()),"sha256":digest(nominal_report)}
        expected=2*len(cfg["angles_deg"])
        if len(nominal)!=expected or not all(c["passed"] for c in nominal):
            report["randomization_skipped"]="All nominal hold cases must pass before expanding parked initial distributions; motion/release scored separately"
        else:
            rng=np.random.default_rng(cfg["randomization"]["seed"])
            records=[]
            for support in range(2):
                free=1-support;accepted=0
                for attempt in range(cfg["randomization"]["max_attempts_per_side"]):
                    delta=rng.uniform(-1,1,7)*np.deg2rad(cfg["randomization"]["joint_jitter_deg"])
                    park=fixture.parks[free]+delta
                    check=fixture.park_check(free,park)
                    if not check["passed"]:
                        records.append({"support":support,"attempt":attempt,"accepted":False,"geometry":check})
                        continue
                    name=f"jitter_{('left','right')[support]}_{accepted:02d}.csv"
                    case=execute(fixture,support,cfg["randomization"]["probe_angle_deg"],"hold",cfg,output/name,park)
                    case.update(kind="jitter_hold",joint_jitter_rad=delta.tolist(),attempt=attempt)
                    report["cases"].append(case);records.append({"support":support,"attempt":attempt,"accepted":True,"geometry":check})
                    accepted+=1
                    write_json(output/"summary.json",report)
                    print(json.dumps({k:v for k,v in case.items() if k!="preparation"}),flush=True)
                    if accepted>=cfg["randomization"]["samples_per_side"]:
                        break
                if accepted<cfg["randomization"]["samples_per_side"]:
                    report.setdefault("randomization_errors",[]).append(f"Insufficient valid parks for support={support}")
            report["randomization_sampling"]=records
    report["counts"]={k:{"passed":sum(c["passed"] for c in report["cases"] if c["kind"]==k),
                           "total":sum(c["kind"]==k for c in report["cases"])}
                      for k in ("hold","motion","release","jitter_hold")}
    report["all_executed_cases_passed"]=bool(report["cases"]) and all(c["passed"] for c in report["cases"])
    report["training_started"]=False
    write_json(output/"summary.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/s00_support.yaml")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stages",nargs="+",choices=("hold","motion","release","jitter"),default=["hold","motion","release","jitter"])
    parser.add_argument("--feedback",action="store_true",help="Explicit bounded wheel-angle feedback SCRIPT baseline, not a policy executor")
    parser.add_argument("--nominal-report",type=Path,help="Reuse matching nominal hold evidence before jitter probes")
    args=parser.parse_args();run(args.config,args.output,args.stages,args.feedback,args.nominal_report)


if __name__=="__main__":
    main()
