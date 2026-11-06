"""Matched offline pilot evaluations; one environment at a time."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import INTERFACE_VERSION, HIDDEN_DIM
from .env import CentralizedWheelEnv
from .train import DEFAULT_CONFIG, load_config, load_checkpoint, tensor

CASES = ["hold", "step_positive", "step_negative", "sine", "sequence", "randomized"]


def evaluate(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a fresh evaluation output directory")
    if args.control == "policy" and args.model is None:
        raise ValueError("Policy evaluation requires --model")
    network, checkpoint = load_checkpoint(args.model) if args.control == "policy" else (None, {})
    if network:
        network.eval()
    results = []
    for case in args.cases:
        for seed in args.seeds:
            config, _ = load_config(args.config, args.stage)
            config.episode_seconds = args.seconds
            config.amplitude_deg = args.amplitude
            config.randomization_strength = .5 if case == "randomized" else 0.
            config.closure_jitter_rad = .08 if case == "randomized" else 0.
            config.settle_jitter_seconds = .2 if case == "randomized" else 0.
            config.target_mode = "hold" if case == "hold" else "step" if case.startswith("step_") else "sequence" if case == "randomized" else case
            config.frequency = .05
            config.no_contact = args.no_contact
            task_settings = asdict(config)
            env = CentralizedWheelEnv(config, width=960, height=540)
            options = {"target_deg": args.amplitude * (1 if case == "step_positive" else -1)} if case.startswith("step_") else None
            frames, rows = [], []
            name = f"{case}_s{seed}"
            recorder = None
            try:
                observation, preparation = env.reset(seed=seed, options=options)
                hidden, episode_start = torch.zeros((1, HIDDEN_DIM)), torch.ones(1)
                rng = np.random.default_rng(seed + 900000)
                if case == args.video_case and seed == args.seeds[0]:
                    import imageio.v2 as imageio
                    recorder = imageio.get_writer(str(args.output / f"{name}.mp4"), fps=config.policy_hz, codec="libx264", macro_block_size=2)
                    recorder.append_data(env.render())
                for _ in range(round(args.seconds * config.policy_hz)):
                    if network is not None:
                        with torch.no_grad():
                            mean, hidden = network.actor(tensor(observation["actor"])[None], hidden, episode_start)
                            action = torch.tanh(mean)[0].numpy()
                        episode_start[:] = 0
                    else:
                        action = np.zeros(24) if args.control == "zero" else rng.uniform(-.1, .1, 24)
                    observation, reward, terminated, truncated, info = env.step(action)
                    info.update(reward=reward, terminated=terminated, truncated=truncated,
                                action_rms=float(np.sqrt(np.mean(action ** 2))))
                    for index, value in enumerate(action):
                        info[f"action_{index}"] = float(value)
                    rows.append(info)
                    if recorder is not None:
                        recorder.append_data(env.render())
                    if terminated or truncated:
                        break
                with (args.output / f"{name}.csv").open("w") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                errors = np.array([x["error_deg"] for x in rows])
                tail = errors[-round(5 * config.policy_hz):]
                full = bool(len(rows) == round(args.seconds * config.policy_hz) and not rows[-1]["terminated"])
                case_passed = full and (np.sqrt(np.mean(errors ** 2)) <= 1. and np.sqrt(np.mean(tail ** 2)) <= 1.
                                       and rows[-1]["max_hold_seconds"] >= min(10., args.seconds - 1) if case == "hold"
                                       else np.sqrt(np.mean(tail ** 2)) <= 1. if case.startswith("step_")
                                       else np.sqrt(np.mean(errors ** 2)) <= 1. and np.max(np.abs(errors)) <= 5.)
                result = {"case": case, "seed": seed, "steps": len(rows), "full_duration": full,
                          "rmse_deg": float(np.sqrt(np.mean(errors ** 2))), "tail_rmse_deg": float(np.sqrt(np.mean(tail ** 2))),
                          "peak_error_deg": float(np.max(np.abs(errors))),
                          "mean_reward": float(np.mean([x["reward"] for x in rows])),
                          "peak_contact_force_n": max(x["peak_force"] for x in rows),
                          "peak_robot_collision_n": max(x["robot_collision_peak"] for x in rows),
                          "max_grasp_drift_deg": max(x["max_grasp_drift_deg"] for x in rows),
                          "max_motor_target_gap_rad": max(x["max_motor_target_gap_rad"] for x in rows),
                          "wheel_min_deg": float(np.rad2deg(min(x["wheel_angle"] for x in rows))),
                          "wheel_max_deg": float(np.rad2deg(max(x["wheel_angle"] for x in rows))),
                          "max_hold_seconds": rows[-1]["max_hold_seconds"],
                          "qp_failures": rows[-1]["qp_failures"], "workspace_clips": rows[-1]["workspace_clips"],
                          "termination_reason": rows[-1]["termination_reason"], "pilot_case_passed": bool(case_passed),
                          "task": task_settings, "preparation": preparation}
                results.append(result)
                print(json.dumps({k: result[k] for k in ["case", "seed", "rmse_deg", "tail_rmse_deg", "full_duration", "termination_reason"]}), flush=True)
                (args.output / "progress.json").write_text(json.dumps(results, indent=2))
            finally:
                if recorder is not None:
                    recorder.close()
                env.close()
    summary = {"interface_version": INTERFACE_VERSION, "control": args.control,
               "model": str(args.model) if args.model else None,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None,
               "training_seed": checkpoint.get("seed"), "training_steps": checkpoint.get("total_steps"),
               "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "criteria": {"hold": "full duration, RMSE and last-5s RMSE <= 1deg, 10s stable hold diagnostic",
                            "step": "full duration, last-5s RMSE <= 1deg",
                            "sine_sequence_randomized": "full duration, RMSE <= 1deg, peak error <= 5deg",
                            "max_hold_seconds": "diagnostic only: 2deg error and speed thresholds, both hands in contact"},
               "cases": results, "all_pilot_cases_passed": all(x["pilot_case_passed"] for x in results),
               "accepted_in_simulation": False, "hardware_ready": False,
               "limits": "Exploratory short-horizon tests; not multi-training-seed or long-duration acceptance"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--control", choices=["policy", "zero", "random"], default="policy")
    parser.add_argument("--stage", choices=["hold", "reach3", "reach10"], default="reach3")
    parser.add_argument("--seconds", type=float, default=15.)
    parser.add_argument("--amplitude", type=float, default=3.)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1007, 1017, 1027])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--video-case", choices=CASES)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or not 0 <= args.amplitude <= 10:
        parser.error("Stage-one evaluation needs positive duration and amplitude in [0, 10]")
    evaluate(args)


if __name__ == "__main__":
    main()
