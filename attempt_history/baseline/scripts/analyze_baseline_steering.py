"""Summarize completed offline tests without changing simulation or policies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/humanlike-drive-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_baseline_steering import case_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    summaries = json.loads((args.directory / "summary.json").read_text())
    protocol = json.loads((args.directory / "protocol.json").read_text())
    assert len(summaries) == len(protocol["cases"]) == 34
    by_id = {case_id(s["case"]): s for s in summaries}
    diagnostics = []
    paired_checks = []
    arrays = {}
    for summary in summaries:
        key = case_id(summary["case"])
        rows = np.genfromtxt(args.directory / key / "metrics.csv", delimiter=",", names=True)
        arrays[key] = rows
        for window in summary["windows"] + summary["minute_windows"]:
            window["actual_end_s"] = min(window["end_s"], summary["simulated_seconds"])
        error = rows["error_deg"]
        actual = np.rad2deg(rows["wheel_angle"])
        command = rows["command_deg"]
        assert np.isfinite(error).all() and np.isfinite(actual).all()
        assert len(rows) == round(summary["simulated_seconds"] * 50)
        assert np.max(np.abs(error-(command-actual))) < 1e-10
        assert np.max(np.abs(np.diff(rows["time"])-.02)) < 1e-9
        start = (rows["time"] >= 10) & (rows["time"] < 60)
        end = rows["time"] > rows["time"][-1]-60
        record = {"id": key, "row_count": len(rows),
                  "last60_mean_signed_error_deg": float(error[end].mean()),
                  "last60_wheel_std_deg": float(actual[end].std()),
                  "velocity_rmse_rad_s": float(np.sqrt(np.mean(
                      (rows["target_velocity_rad_s"]-rows["wheel_velocity"])**2))),
                  "saturated_substep_fraction": float(rows["saturated_substep_fraction"].mean()),
                  "peak_net_inplane_contact_force_N": float(rows["net_inplane_contact_force_N"].max()),
                  "final_left_palm_wheel_change_deg": float(rows["left_palm_wheel_change_deg"][-1]),
                  "final_right_palm_wheel_change_deg": float(rows["right_palm_wheel_change_deg"][-1])}
        if start.any():
            for side in ("left", "right"):
                column = side + "_palm_wheel_change_deg"
                record[side + "_palm_wheel_median_change_last_vs_10to60_deg"] = float(
                    np.median(rows[column][end])-np.median(rows[column][start]))
        diagnostics.append(record)
        case = summary["case"]
        if case["policy"] == "ppo":
            other_case = dict(case, policy="no_ppo")
            other_key = case_id(other_case)
            other = by_id[other_key]
            assert summary["physics_parameters"] == other["physics_parameters"]
            # Every paired run starts from the same deterministic pregrasp.
            assert summary["initial_state"] == other["initial_state"]
            paired_checks.append({"ppo": key, "no_ppo": other_key,
                                  "physics_and_initial_info_match": True})
    for pair in paired_checks:
        a, b = arrays[pair["ppo"]], arrays[pair["no_ppo"]]
        n = min(len(a), len(b))
        difference = float(np.abs(a["command_deg"][:n]-b["command_deg"][:n]).max())
        assert difference == 0
        pair["max_command_difference_deg"] = difference

    replay_validation = {}
    for policy, old_name in (("ppo", "position_ppo.csv"), ("no_ppo", "position_baseline.csv")):
        key = f"replay_clockwise_nominal_s7_{policy}"
        new = arrays[key][4::5]
        old = np.genfromtxt(args.directory.parent / "servo_replay" / old_name,
                            delimiter=",", names=True)
        assert len(new) == len(old) == 3000
        difference = float(np.max(np.abs(new["wheel_angle"]-old["actual_rad"])))
        assert difference < 1e-10
        replay_validation[policy] = {"samples": len(new), "max_wheel_difference_rad": difference,
                                    "rmse_10hz_deg": float(np.sqrt(np.mean(new["error_deg"]**2))),
                                    "rmse_50hz_deg": by_id[key]["rmse_deg"]}

    analysis = {"case_count": len(summaries), "paired_checks": paired_checks,
                "replay_validation": replay_validation, "diagnostics": diagnostics,
                "early_failures": [case_id(s["case"]) for s in summaries if s["early_failure"]],
                "completed_requested_seconds": float(sum(s["simulated_seconds"] for s in summaries)),
                "max_500hz_contact_force_N": float(max(s["peak_single_contact_force_N_500hz"] for s in summaries)),
                "max_500hz_non_hand_wheel_force_N": float(max(s["peak_non_hand_wheel_force_N_500hz"] for s in summaries)),
                "baseline_driver_sha256": hashlib.sha256(Path(__file__).with_name(
                    "benchmark_baseline_steering.py").read_bytes()).hexdigest()}
    (args.directory / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (args.directory / "summary.json").write_text(json.dumps(summaries, indent=2))

    names = ["hold_center", "hold_pos15", "hold_neg15", "hold_pos30", "hold_neg30",
             "sine15_slow", "sine30_slow", "sine30_fast", "stop_reverse", "replay_clockwise"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    y = np.arange(len(names))
    for offset, policy, label, color in ((-.17, "ppo", "PPO residual enabled", "#2166ac"),
                                        (.17, "no_ppo", "Zero residual", "#b35806")):
        selected = [by_id[f"{name}_nominal_s7_{policy}"] for name in names]
        axes[0].barh(y+offset, [s["rmse_deg"] for s in selected], .32, label=label, color=color)
        for position, s in zip(y+offset, selected):
            if s["early_failure"]:
                axes[0].text(s["rmse_deg"], position, " *early stop", color="red", fontsize=8)
        axes[1].barh(y+offset, [100*s["both_hands_contact_fraction"] for s in selected], .32, color=color)
    axes[0].set_yticks(y, names)
    axes[1].set_yticks(y, names)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.2)
    axes[0].set_xlabel("Full-run RMSE at 50 Hz (deg)")
    axes[1].set_xlabel("Both hands in loaded contact (%)")
    axes[0].legend()
    fig.suptitle("Preserved baseline: isolated steering tests, nominal physics")
    fig.savefig(args.directory / "overview.png", dpi=160)
    fig.savefig(args.directory / "overview.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for axis, name in zip(axes.flat, ("sine15_slow", "sine30_slow", "sine30_fast", "replay_clockwise")):
        for policy, color in (("ppo", "#2166ac"), ("no_ppo", "#b35806")):
            s = by_id[f"{name}_nominal_s7_{policy}"]
            axis.plot([(w["start_s"]+w["actual_end_s"])/2 for w in s["minute_windows"]],
                      [w["rmse_deg"] for w in s["minute_windows"]], marker="o", label=policy, color=color)
        axis.set_title(name)
        axis.set_xlabel("Window midpoint (s)")
        axis.set_ylabel("60 s window RMSE (deg)")
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(args.directory / "minute_windows.png", dpi=160)
    fig.savefig(args.directory / "minute_windows.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for policy, color in (("ppo", "#2166ac"), ("no_ppo", "#b35806")):
        data = arrays[f"replay_clockwise_nominal_s7_{policy}"][::5]
        axes[0].plot(data["time"], np.rad2deg(data["wheel_angle"]), label=policy, color=color)
        axes[1].plot(data["time"], data["reference_deg"], label=policy, color=color)
        axes[2].plot(data["time"], data["left_palm_wheel_change_deg"], label=policy+" left", color=color)
        axes[2].plot(data["time"], data["right_palm_wheel_change_deg"], label=policy+" right", color=color, linestyle="--")
    axes[0].plot(data["time"], data["command_deg"], label="Recorded command", color="black", linewidth=1)
    axes[0].set_ylabel("Wheel angle (deg)")
    axes[1].set_ylabel("Hand arc reference (deg)")
    axes[2].set_ylabel("Palm/wheel change (deg)")
    axes[2].set_xlabel("Simulation time (s)")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(ncol=2)
    fig.suptitle("Identical recorded targets; no live driving upstream")
    fig.savefig(args.directory / "replay_comparison.png", dpi=160)
    plt.close(fig)

    args.archive.mkdir(parents=True, exist_ok=True)
    for filename in ("protocol.json", "summary.json", "summary.csv", "analysis.json"):
        shutil.copy2(args.directory / filename, args.archive / filename)
    with (args.directory / "nominal_table.md").open("w") as handle:
        handle.write("| 条件 | PPO RMSE | 零残差 RMSE | PPO 最后60s RMSE | 零残差 最后60s RMSE | PPO双手接触 | 零残差双手接触 | 实际时长 PPO/零残差 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for name in names:
            a, b = [by_id[f"{name}_nominal_s7_{p}"] for p in ("ppo", "no_ppo")]
            handle.write(f"| {name} | {a['rmse_deg']:.3f}° | {b['rmse_deg']:.3f}° | "
                         f"{a['last60_rmse_deg']:.3f}° | {b['last60_rmse_deg']:.3f}° | "
                         f"{a['both_hands_contact_fraction']:.1%} | {b['both_hands_contact_fraction']:.1%} | "
                         f"{a['simulated_seconds']:.2f}/{b['simulated_seconds']:.2f} s |\n")
    print(json.dumps({k: v for k, v in analysis.items() if k not in ("diagnostics", "paired_checks")}, indent=2))


if __name__ == "__main__":
    main()
