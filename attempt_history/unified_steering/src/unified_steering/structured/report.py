"""Compact review artifacts from measured gates, independent of PPO reward."""
import argparse
import json
from pathlib import Path


def report(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run/"config.json").read_text())
    index = json.loads((run/"development_index.json").read_text())
    figures, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    rows = []
    directions=["positive","negative","sequence"]
    summaries=[]
    for entry in index:
        summary = json.loads((run/"development"/f"step_{entry['steps']:09d}"/"summary.json").read_text())
        summaries.append(summary)
        if any(c["case"]=="sequence_negative" for c in summary["cases"]) and "sequence_negative" not in directions:
            directions.append("sequence_negative")
    for entry,summary in zip(index,summaries):
        cases = summary["cases"]
        values = {name: max(c["rmse_deg"] for c in cases if c["case"]==name)
                  for name in directions}
        values["hold_peak"] = max(c["peak_error_deg"] for c in cases if c["case"]=="hold")
        rows.append(dict(steps=entry["steps"], passed=entry["passed"],
                         complete=sum(c["complete"] for c in cases),
                         precision_passes=sum(c["passed"] for c in cases),total=len(cases),**values))
    for name in directions:
        axes[0].plot([r["steps"] for r in rows], [r[name] for r in rows], "o-", label=name)
    axes[1].plot([r["steps"] for r in rows], [r["hold_peak"] for r in rows], "o-", label="hold peak")
    for axis, label in zip(axes, ("Worst case RMSE (deg)", "Worst hold peak (deg)")):
        axis.axhline(1., color="black", linestyle="--", linewidth=1, label="1 deg gate")
        axis.set(xlabel="Environment steps", ylabel=label)
        axis.grid(alpha=.25); axis.legend()
    figures.savefig(output/"development.png", dpi=160)
    plt.close(figures)
    (output/"development.json").write_text(json.dumps(rows, indent=2))
    lines = [f"# {metadata['stage']} measured progress", "",
             f"Run: `{run}`. Status: `{metadata['status']}`.", "",
             "| Steps | "+" | ".join(directions)+" | Hold peak | Complete | Precision passes | All gates |",
             "| --- | "+" | ".join("---" for _ in directions)+" | --- | --- | --- | --- |"]
    for r in rows:
        values=" | ".join(f"{r[name]:.3f}°" for name in directions)
        lines.append(f"| {r['steps']} | {values} | {r['hold_peak']:.3f}° | {r['complete']}/{r['total']} | {r['precision_passes']}/{r['total']} | {r['passed']} |")
    lines += ["", "![Measured development gates](development.png)", "",
              "Development passes do not replace independent acceptance. This is the privileged teacher; no student or hardware claim.",
              "Incomplete cases have prefix RMSE only; inspect completeness before comparing error curves."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(output), status=metadata["status"], evaluations=len(index))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)
