"""Plot the physical hold regression; this is not a learned-policy curve."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Validation output containing holds.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    with (args.run / "holds.csv").open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault((row["support"], row["state"]), []).append(
                (float(row["time_s"]), float(row["error_deg"])))
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    colors = {"left": "#277DA1", "right": "#F9844A", "both": "#43AA8B"}
    seen = set()
    for (support, state), rows in groups.items():
        x, y = zip(*rows)
        label = f"{support} support" if support not in seen else None
        ax.plot(x, y, color=colors[support], alpha=.8, linewidth=1.8, label=label)
        seen.add(support)
    ax.axhline(2, linestyle="--", color="#444444", linewidth=1, label="2 degree gate")
    ax.set(xlabel="Time (s)", ylabel="Absolute wheel drift (deg)", ylim=(0, 2.2), xlim=(0, 3),
           title="M1 fixed-target hold through the 26D action interface")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output.with_suffix('.'+extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
