"""Frozen motion/contact audit, with no modifications to policy execution."""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .evaluate import load_teacher, run_case
from .interventions import DiagnosticConfig, DiagnosticEnv, TRACE_FIELDS


class MotionAuditEnv(DiagnosticEnv):
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        c = self.controller
        local = (self.data.site_xpos[c.grip_ids] - c.center) @ c.axis_frame
        command = (c.wrist_positions - c.center) @ c.axis_frame
        rotation = self.data.site_xmat[c.grip_ids].reshape(2, 3, 3)
        phase = np.arctan2(local[:, 2], local[:, 1])
        command_phase = np.arctan2(command[:, 2], command[:, 1])
        info.update(audit_action=np.asarray(action).tolist(),
                    audit_phase_rad=phase.tolist(),
                    audit_phase_lead_rad=np.arctan2(np.sin(command_phase-phase), np.cos(command_phase-phase)).tolist(),
                    audit_orientation_lead_rad=Rotation.from_matrix(c.wrist_rotations @ rotation.transpose(0, 2, 1)).magnitude().tolist(),
                    audit_arm_target_error_rad=(c.target[:14]-self.data.qpos[c.arm_q]).tolist(),
                    audit_actuator_force=self.data.actuator_force.tolist())
        return obs, reward, done, truncated, info


FIELDS = TRACE_FIELDS + ("target_velocity", "wheel_velocity", "centering_torque_nm",
    "audit_action", "audit_phase_rad", "audit_phase_lead_rad", "audit_orientation_lead_rad",
    "audit_arm_target_error_rad", "audit_actuator_force")


def job(args):
    name, path, support, case, seed, seconds = args
    torch.set_num_threads(1)
    metadata = json.loads((path.parent / "config.json").read_text())
    task = metadata["task"]
    config = DiagnosticConfig(hand_action_mode=task["hand_action_mode"],
        hand_leak_alpha=task["hand_leak_alpha"], angle_reward_sigma_deg=task["angle_reward_sigma_deg"],
        hand_joint_lead_rad=task.get("hand_joint_lead_rad"))
    record, trace = run_case(load_teacher(path), config, case, seed, seconds, 10., support,
                              env_class=MotionAuditEnv, trace_fields=FIELDS)
    record["model"] = name
    return record, trace


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output / "protocol.json")
    protocol = json.loads(args.protocol.read_text())
    jobs = [(name, Path(path), support, case, seed, protocol["seconds"])
            for name, path in protocol["models"].items() for support in protocol["supports"]
            for case in protocol["cases"] for seed in protocol["development_seeds"]]
    records = []
    with mp.get_context("fork").Pool(4) as pool:
        for record, trace in pool.imap(job, jobs):
            filename = f"{record['model']}_{record['support']}_{record['case']}_s{record['seed']}"
            arrays = {key: np.array([row[key] for row in trace]) for key in trace[0] if key != "termination_reason"}
            np.savez_compressed(args.output / (filename + ".npz"), **arrays)
            error = arrays["error_deg"]
            moving = abs(np.rad2deg(arrays["target_velocity"])) > protocol["motion_threshold_deg_s"]
            sides = (0, 1) if record["support"] == "both" else (0,) if record["support"] == "left" else (1,)
            slip = arrays["audit_phase_rad"] - arrays["wheel_angle"][:, None]
            slip = np.unwrap(slip, axis=0)
            phase_lead = np.rad2deg(arrays["audit_phase_lead_rad"][:, sides])
            hand_error = abs(arrays["audit_hand_command"][:, sides] - arrays["audit_hand_q"][:, sides])
            forces = arrays["left_force"] if sides == (0,) else arrays["right_force"] if sides == (1,) else arrays["left_force"] + arrays["right_force"]
            record.update(motion_rmse_deg=float(np.sqrt(np.mean(error[moving]**2))),
                stopped_rmse_deg=float(np.sqrt(np.mean(error[~moving]**2))),
                support_force_min_n=float(forces.min()),
                support_low_force_fraction=float(np.mean(forces < .5)),
                material_phase_range_deg=np.rad2deg(np.ptp(slip[:, sides], axis=0)).tolist(),
                max_wrist_phase_lead_deg=float(abs(phase_lead).max()),
                max_hand_target_error_rad=float(hand_error.max()),
                hand_lead_rad=float(arrays["audit_hand_lead_rad"][-1]),
                hand_at_lead_fraction=float(np.mean(hand_error >= arrays["audit_hand_lead_rad"][:, None, None]-1e-4)))
            records.append(record)
            (args.output / "summary.json").write_text(json.dumps(dict(cases=records,
                hashes={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in protocol["models"].items()}), indent=2))
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/r22_motion_audit_v3.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
