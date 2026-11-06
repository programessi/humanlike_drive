"""Summarize frozen-policy evaluations without running a simulator."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("zero", "hold", "reach3")
LABELS = {"zero": "Zero action", "hold": "Hold PPO (8,192)", "reach3": "Reach PPO (24,576 total)"}
CASES = ("hold", "step_positive", "step_negative", "sine", "sequence")


def command_diagnostic():
    # Pure command generation: no MuJoCo environment is constructed.
    from wheel_manipulation.exploration.env import ExplorationConfig, CentralizedWheelEnv
    result = {}
    for mode in ("hold", "sine", "sequence", "random_steps"):
        means, first, positive, negative = [], [], [], []
        for seed in range(300):
            rng = np.random.default_rng(seed)
            stub = SimpleNamespace(config=ExplorationConfig(target_mode=mode), dt=.05,
                                   episode_mode=mode, frequency=float(rng.uniform(.01, .1)),
                                   step_target=0., np_random=rng)
            angles, _ = CentralizedWheelEnv._make_commands(stub)
            angles = np.rad2deg(angles[:300])
            nonzero = angles[np.abs(angles) > 1e-6]
            first.append(float(nonzero[0] > 0) if len(nonzero) else 0.)
            means.append(float(angles.mean()))
            positive.append(float(np.mean(angles > 1e-6)))
            negative.append(float(np.mean(angles < -1e-6)))
        result[mode] = {"generated_traces": 300, "mean_command_deg": float(np.mean(means)),
                        "first_nonzero_positive_fraction": float(np.mean(first)),
                        "positive_time_fraction": float(np.mean(positive)),
                        "negative_time_fraction": float(np.mean(negative))}
    result["note"] = "Command-generator diagnostic only; not replay of actual training RNG or simulator evaluation."
    return result


def read_trace(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in
            ("time", "target_angle", "wheel_angle", "error_deg", "action_rms")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/exploration_stage1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/exploration_stage1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "command_generator_diagnostic.json").write_text(json.dumps(command_diagnostic(), indent=2))
    rows, response, randomized = [], {}, {}
    for name in NAMES:
        for split in ("nominal", "randomized"):
            summary = json.loads((args.input / f"{name}_{split}" / "summary.json").read_text())
            for case in summary["cases"]:
                rows.append({"controller": name, "split": split, **{key: case[key] for key in (
                    "case", "seed", "full_duration", "rmse_deg", "tail_rmse_deg", "peak_error_deg",
                    "max_grasp_drift_deg", "peak_contact_force_n", "peak_robot_collision_n",
                    "wheel_min_deg", "wheel_max_deg", "max_hold_seconds", "qp_failures",
                    "workspace_clips", "termination_reason", "pilot_case_passed")}})
            if split == "randomized":
                cases = summary["cases"]
                randomized[name] = {"n": len(cases), "rmse_mean_deg": float(np.mean([x["rmse_deg"] for x in cases])),
                    "rmse_std_deg": float(np.std([x["rmse_deg"] for x in cases])),
                    "rmse_min_deg": min(x["rmse_deg"] for x in cases),
                    "rmse_max_deg": max(x["rmse_deg"] for x in cases),
                    "full_duration_count": sum(x["full_duration"] for x in cases),
                    "passed_count": sum(x["pilot_case_passed"] for x in cases)}
        positive = read_trace(args.input / f"{name}_nominal/step_positive_s1007.csv")
        negative = read_trace(args.input / f"{name}_nominal/step_negative_s1007.csv")
        response[name] = {
            "last5s_positive_wheel_mean_deg": float(np.rad2deg(positive["wheel_angle"][-100:]).mean()),
            "last5s_negative_wheel_mean_deg": float(np.rad2deg(negative["wheel_angle"][-100:]).mean()),
            "last5s_directional_span_deg": float(np.rad2deg(positive["wheel_angle"][-100:]).mean()
                                                 - np.rad2deg(negative["wheel_angle"][-100:]).mean()),
            "ideal_directional_span_deg": 6.,
            "note": "Paired nominal +3/-3 commands, identical preparation. Not a success metric by itself."}
    with (args.output / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"cases": rows, "paired_direction_response": response, "randomized": randomized,
              "accepted_in_simulation": False, "hardware_ready": False,
              "limits": "One training seed; 15s episodes; nominal seed repeats are not independent trials."}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for axis, case in zip(axes.flat, (*CASES, "randomized")):
        split = "randomized" if case == "randomized" else "nominal"
        for i, name in enumerate(NAMES):
            trace = read_trace(args.input / f"{name}_{split}/{case}_s1007.csv")
            if i == 0:
                axis.plot(trace["time"], np.rad2deg(trace["target_angle"]), "k--", label="Command", linewidth=1.5)
            axis.plot(trace["time"], np.rad2deg(trace["wheel_angle"]), label=LABELS[name], linewidth=1.3)
        axis.set(title=case.replace("_", " "), ylabel="Wheel angle (deg)")
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time after physical preparation (s)")
    fig.suptitle("Stage 1 pilot: frozen policy, +/-3 deg, seed 1007\nRandomized panel uses one preparation seed; see table for all three")
    fig.savefig(args.output / "tracking.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"paired_direction_response": response, "randomized": randomized}, indent=2))


if __name__ == "__main__":
    main()
