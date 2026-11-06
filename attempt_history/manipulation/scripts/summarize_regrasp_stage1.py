"""Analyze frozen evaluation artifacts only; never launches a simulation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(inputs, labels, output):
    if len(inputs) != len(labels):
        raise ValueError("One label per evaluation directory is required")
    output.mkdir(parents=True, exist_ok=False)
    summaries = [json.loads((p / "summary.json").read_text()) for p in inputs]
    records = []
    for label, summary in zip(labels, summaries):
        for case in sorted({r["case"] for r in summary["cases"]}):
            rows = [r for r in summary["cases"] if r["case"] == case]
            records.append({"model": label, "case": case, "n": len(rows),
                            "rmse_mean_deg": float(np.mean([r["rmse_deg"] for r in rows])),
                            "rmse_std_deg": float(np.std([r["rmse_deg"] for r in rows])),
                            "tail_rmse_mean_deg": float(np.mean([r["tail_rmse_deg"] for r in rows])),
                            "peak_error_max_deg": max(r["peak_error_deg"] for r in rows),
                            "complete": sum(r["full_duration"] for r in rows),
                            "tracking_pass": sum(r["tracking_gate_passed"] for r in rows),
                            "candidate_regrasp_events": sum(sum(r["candidate_regrasp_events"]) for r in rows)})
    with (output / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    (output / "comparison.json").write_text(json.dumps({"inputs": list(map(str, inputs)), "labels": labels,
        "models": [s.get("model_sha256") for s in summaries], "records": records,
        "stage1_passed": False, "hardware_ready": False}, indent=2))
    cases = sorted({r["case"] for r in records})
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, max(3, 2.8 * len(cases))), squeeze=False)
    for i, case in enumerate(cases):
        reference_drawn = False
        for directory, label, summary in zip(inputs, labels, summaries):
            selection = [r for r in summary["cases"] if r["case"] == case]
            if not selection:
                continue
            seed = selection[0]["seed"]
            with (directory / f"{case}_s{seed}.csv").open() as f:
                trace = list(csv.DictReader(f))
            t = np.array([float(r["time"]) for r in trace])
            target = np.rad2deg([float(r["target_angle"]) for r in trace])
            actual = np.rad2deg([float(r["wheel_angle"]) for r in trace])
            if not reference_drawn:
                axes[i, 0].plot(t, target, "k--", label="reference")
                reference_drawn = True
            axes[i, 0].plot(t, actual, label=label)
            axes[i, 1].plot(t, target - actual, label=label)
        axes[i, 0].set_title(f"{case}: first preparation seed per controller")
        axes[i, 1].set_title("Timed tracking error")
        for j in range(2):
            axes[i, j].set(xlabel="Time (s)", ylabel="Degrees")
            axes[i, j].grid(alpha=.25); axes[i, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "tracking.png", dpi=160); plt.close(fig)
    lines = ["# Frozen evaluation comparison", "", "All errors in degrees. These are small-angle tracking results, not stage-1 handover acceptance.", "",
             "| Model | Case | n | RMSE mean ± SD | Tail RMSE | Worst peak | Complete | Tracking gate |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f'| {r["model"]} | {r["case"]} | {r["n"]} | {r["rmse_mean_deg"]:.3f} ± {r["rmse_std_deg"]:.3f} | {r["tail_rmse_mean_deg"]:.3f} | {r["peak_error_max_deg"]:.3f} | {r["complete"]}/{r["n"]} | {r["tracking_pass"]}/{r["n"]} |')
    lines += ["", "Tracking gate: complete declared duration, RMSE ≤2°, peak ≤5°. Tail means last 100 policy steps (5s at 20Hz).",
              "", "Preparation seeds are not training seeds. A successful contact candidate is not proof of load transfer.",
              "", "![Tracking](tracking.png)", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summarize(a.inputs, a.labels, a.output)
