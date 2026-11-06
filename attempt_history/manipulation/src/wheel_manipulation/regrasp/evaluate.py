"""Frozen checkpoints: timed references, candidate regrasp events, real traces."""
import argparse
from dataclasses import asdict, fields
import csv
import hashlib
import json
import os
from pathlib import Path

# CPU rendering also works on the current machine without a working GPU driver.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO

from . import INTERFACE_VERSION
from .env import RegraspEnv, RegraspConfig
from .controller import RegraspControlConfig
from ..paths import ROOT


def evaluate(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.video is None:
        args.video = args.control != "zero"
    video_case = "sequence" if "sequence" in args.cases else args.cases[0]
    teacher = PPO.load(args.model, device="cpu") if args.control == "teacher" else None
    student = torch.jit.load(str(args.model)).eval() if args.control == "student" else None
    controller = RegraspControlConfig()
    metadata_path = args.controller_from or (args.model.parent / "config.json" if args.model else None)
    if metadata_path and metadata_path.exists():
        stored = json.loads(metadata_path.read_text())["task"]["controller"]
        stored.setdefault("guard_max_iterations", 30)  # Pre-fix checkpoint execution.
        allowed = {f.name for f in fields(RegraspControlConfig)}
        controller = RegraspControlConfig(**{k: v for k, v in stored.items() if k in allowed})
    results = []
    for case in args.cases:
        stage = "B" if case.startswith("single_") else "D" if case == "full" else "A"
        config = RegraspConfig(curriculum=stage, episode_seconds=args.seconds, amplitude_deg=args.amplitude,
                               randomization_strength=args.randomization, target_mode="hold" if case == "hold" else "sequence",
                               shaping_weight=0., controller=RegraspControlConfig(**asdict(controller)))
        for seed in args.seeds:
            env = RegraspEnv(config, width=960, height=540)
            options = {"direction": -1 if case == "negative" else 1, "exact_curriculum": True}
            if case.startswith("single_"): options["free_side"] = 1 if case == "single_left" else 0
            if case in ("positive", "negative"): options["mode"] = "step"
            rows, writer = [], None
            try:
                obs, preparation = env.reset(seed=seed, options=options)
                h, starts = torch.zeros(1, 128), torch.ones(1)
                if args.video and case == video_case and seed == args.seeds[0]:
                    import imageio.v2 as imageio
                    writer = imageio.get_writer(str(args.output / f"{case}_s{seed}.mp4"), fps=config.policy_hz,
                                               codec="libx264", macro_block_size=2)
                    writer.append_data(env.render())
                for _ in range(round(args.seconds * config.policy_hz)):
                    if teacher:
                        action, _ = teacher.predict(obs, deterministic=True)
                    elif student:
                        with torch.no_grad():
                            output, h = student(torch.from_numpy(obs["actor"])[None], h, starts)
                        action = output[0].numpy(); starts[:] = 0
                    else:
                        action = np.zeros(24)
                    obs, reward, done, truncated, info = env.step(action)
                    info.update(reward=reward, terminated=done, truncated=truncated)
                    for i, value in enumerate(action): info[f"action_{i}"] = float(value)
                    rows.append(info)
                    if writer:
                        from PIL import Image, ImageDraw
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 960, 48), fill=(0, 0, 0))
                        text = f't={env.time:5.1f}s target={np.rad2deg(env.target_angle):6.1f}deg actual={np.rad2deg(env.wheel_angle):6.1f}deg | candidate regrasp L/R={env.regrasp_count.tolist()}'
                        draw.text((8, 8), text, fill=(255, 255, 255))
                        draw.text((8, 28), f'{args.control.upper()} policy evaluation - stage 1 not yet accepted', fill=(255, 210, 100))
                        writer.append_data(np.asarray(frame))
                    if done or truncated: break
                name = f"{case}_s{seed}"
                with (args.output / f"{name}.csv").open("w") as f:
                    csv_writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    csv_writer.writeheader(); csv_writer.writerows(rows)
                error = np.array([r["error_deg"] for r in rows])
                full = len(rows) == round(args.seconds * config.policy_hz) and not rows[-1]["terminated"]
                both_contact = float(np.mean([min(r["left_force"], r["right_force"]) > .5 for r in rows]))
                drift = max(r["max_grasp_drift_deg"] for r in rows)
                joint_margin = min(min(r["left_joint_margin_deg"], r["right_joint_margin_deg"]) for r in rows)
                result = {"case": case, "seed": seed, "steps": len(rows), "full_duration": full,
                          "rmse_deg": float(np.sqrt(np.mean(error ** 2))),
                          "tail_rmse_deg": float(np.sqrt(np.mean(error[-100:] ** 2))),
                          "peak_error_deg": float(np.max(abs(error))),
                          "wheel_min_deg": float(np.rad2deg(min(r["wheel_angle"] for r in rows))),
                          "wheel_max_deg": float(np.rad2deg(max(r["wheel_angle"] for r in rows))),
                          "candidate_regrasp_events": env.regrasp_count.tolist(),
                          "both_contact_fraction": both_contact,
                          "max_grasp_drift_deg": drift,
                          "min_joint_margin_deg": joint_margin,
                          "max_palm_target_error_mm": max(r["actual_palm_error_mm"] for r in rows),
                          "small_angle_grasp_gate_passed": bool(full and both_contact >= .95 and drift <= 5 and joint_margin >= 1) if stage == "A" else None,
                          "peak_robot_collision_n": max(r["robot_collision_peak"] for r in rows),
                          "peak_contact_n": max(r["peak_force"] for r in rows),
                          "termination_reason": rows[-1]["termination_reason"],
                          "tracking_gate_passed": bool(full and np.sqrt(np.mean(error ** 2)) <= 2 and np.max(abs(error)) <= 5),
                          "preparation": preparation, "task": asdict(config)}
                results.append(result)
                print(json.dumps({k: result[k] for k in ["case", "seed", "rmse_deg", "tail_rmse_deg", "full_duration", "candidate_regrasp_events"]}), flush=True)
                (args.output / "progress.json").write_text(json.dumps(results, indent=2))
            finally:
                if writer: writer.close()
                env.close()
    summary = {"interface": INTERFACE_VERSION, "control": args.control,
               "model": str(args.model) if args.model else None,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None,
               "controller_from": str(metadata_path) if metadata_path else None,
               "video": f"{video_case}_s{args.seeds[0]}.mp4" if args.video else None,
               "evaluation_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / "src/wheel_manipulation").rglob("*.py")},
               "cases": results, "stage1_passed": False, "hardware_ready": False,
               "limits": "Contact-transfer candidates require separate dynamic support verification; tracking alone is not stage1 acceptance."}
    summary["criteria"] = {"tracking": "full duration, RMSE <=2deg and peak <=5deg",
                           "small_angle_grasp_diagnostic": "A only: full duration, both contact >=95%, wrist-material drift proxy <=5deg, joint margin >=1deg; not real slip or full stage-1 acceptance"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--control", choices=["zero", "teacher", "student"], default="teacher")
    p.add_argument("--model", type=Path)
    p.add_argument("--controller-from", type=Path, help="Training config JSON; useful for matching a zero-action control")
    p.add_argument("--cases", nargs="+", choices=["hold", "positive", "negative", "sequence", "single_left", "single_right", "full"], default=["hold", "positive", "negative", "sequence"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1007, 1017, 1027])
    p.add_argument("--seconds", type=float, default=20.)
    p.add_argument("--amplitude", type=float, default=10.)
    p.add_argument("--randomization", type=float, default=0.)
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=None,
                   help="Record first-seed sequence (default for teacher/student); --no-video disables rendering")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.control != "zero" and a.model is None: p.error("Policy evaluation requires --model")
    if a.seconds < .05: p.error("Evaluation must contain at least one policy step")
    evaluate(a)
