"""Compare the frozen R2.2 trials using saved development evidence only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def metrics(summary):
    rows = summary["cases"]
    groups = defaultdict(list)
    for row in rows:
        key = (row["support"], row["case"], row["amplitude_deg"], row["seconds"])
        groups[key].append(row)
    detailed = []
    for key, group in groups.items():
        complete = [r for r in group if r["complete"]]
        detailed.append(dict(support=key[0], case=key[1], amplitude_deg=key[2], seconds=key[3],
                             total=len(group), complete=len(complete),
                             passes=sum(r["passed"] for r in group),
                             complete_mean_rmse_deg=float(np.mean([r["rmse_deg"] for r in complete])) if complete else None,
                             complete_max_rmse_deg=max((r["rmse_deg"] for r in complete), default=None),
                             complete_max_tail_deg=max((r["tail_peak_deg"] for r in complete), default=None)))
    return dict(total=len(rows), complete=sum(r["complete"] for r in rows),
                passes=sum(r["passed"] for r in rows),
                worst_hold_peak_deg=max(r["peak_error_deg"] for r in rows if r["case"] == "hold"),
                max_free_hand_force_n=max((r["free_hand_peak_force_n"] or 0 for r in rows)),
                min_arm_margin_deg=min(r["min_arm_margin_deg"] for r in rows),
                collision_peak_n=max(r["collision_peak_n"] for r in rows),
                qp_failures=sum(r["qp_failures"] for r in rows), groups=detailed,
                failures=[r for r in rows if not r["passed"]],
                baseline_improvements=summary.get("baseline_improvements", {}))


def summarize(run):
    config = json.loads((run / "config.json").read_text())
    index = json.loads((run / "development_index.json").read_text())
    # Reproduce the declared selection, rather than choosing a convenient final.
    eligible = [r for r in index if r["hold_safe"]]
    best = min(eligible, key=lambda r: r["score"])
    selected = {"initial": index[0], "best": best, "final": index[-1]}
    result = dict(run=str(run), status=config["status"], parent_sha256=config["parent_sha256"],
                  actual_steps=config.get("actual_steps"), interface=config["interface"], checkpoints={})
    for label, entry in selected.items():
        source = run / "development" / f"step_{entry['steps']:09d}"
        checkpoint = run / f"{label}.zip"
        result["checkpoints"][label] = dict(steps=entry["steps"], selection_score=entry["score"],
            model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            **metrics(json.loads((source / "summary.json").read_text())))
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    slots = defaultdict(set)
    for episode in episodes:
        slots[episode["course_block"]].add(episode["course_slot"])
    result["training"] = dict(completed_episodes=len(episodes),
        support_counts=dict(Counter("both" if r["free_side"] == -1 else "left" if r["free_side"] == 1 else "right" for r in episodes)),
        termination_counts=dict(Counter(r["termination_reason"] for r in episodes if r["termination_reason"])),
        covered_slots_per_block={str(k): len(v) for k, v in slots.items()})
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = {run.name: summarize(run) for run in args.runs}
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# R2.2 配对实验实测比较", "",
        "同一σ=2°父模型、64回合分层课程、65,536步预算；B相对A只改手指领先量5°→0.15rad。",
        "开发种子1007/1107/1117，共72例，包含双手、左右单手和两种起始方向；不是独立验收。", "",
        "| 实验 | 检查点 | 步数 | 完整 | 1°任务达标 | 最差保持峰值 | 最差组均值选模分数 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, result in results.items():
        for label, row in result["checkpoints"].items():
            lines.append(f"| {name} | {label} | {row['steps']} | {row['complete']}/{row['total']} | {row['passes']}/{row['total']} | {row['worst_hold_peak_deg']:.4f}° | {row['selection_score']:.4f} |")
    lines += ["", "best沿用训练前声明的支撑×任务组均值选模规则，单向任务组包含5°和10°；完整细分表另外按幅度/时长拆开，避免掩盖某种任务。未完成案例仅记录前缀误差，不作为完整任务精度。", "",
              "## best检查点细分", "",
              "| 实验 | 支撑 | 任务 | 幅度/时长 | 完整 | 达标 | 完整任务平均RMSE | 最大末段峰值 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, result in results.items():
        for group in result["checkpoints"]["best"]["groups"]:
            error = group["complete_mean_rmse_deg"]
            tail = group["complete_max_tail_deg"]
            error_text = f"{error:.4f}°" if error is not None else "无完整任务"
            tail_text = f"{tail:.4f}°" if tail is not None else "—"
            lines.append(f"| {name} | {group['support']} | {group['case']} | {group['amplitude_deg']:g}°/{group['seconds']:g}s | {group['complete']}/{group['total']} | {group['passes']}/{group['total']} | {error_text} | {tail_text} |")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, case in enumerate(("sequence", "sequence_negative")):
        for name, result in results.items():
            step = result["checkpoints"]["best"]["steps"]
            trace = np.load(Path(result["run"]) / "development" / f"step_{step:09d}" / "nominal" / f"right_{case}_10_22_s1007.npz")
            t, angle, target = (trace[k] for k in ("time", "angle", "target"))
            axes[0, col].plot(t, np.rad2deg(angle), label=name)
            axes[1, col].plot(t, np.rad2deg(angle - target), label=name)
        axes[0, col].plot(t, np.rad2deg(target), "k--", label="reference")
        axes[0, col].set(title=f"Right support, {case}, seed1007", ylabel="Wheel angle (deg)")
        axes[1, col].set(xlabel="Time (s)", ylabel="Tracking error (deg)")
        for limit in (-1, 1):
            axes[1, col].axhline(limit, color="black", linestyle=":", linewidth=.8)
        for axis in axes[:, col]:
            axis.grid(alpha=.25)
            axis.legend(fontsize=7)
    figure.savefig(args.output / "right_sequences.png", dpi=160)
    plt.close(figure)
    lines += ["", "![Right sequence comparison](right_sequences.png)", "",
              "曲线为单个已查看种子的示例；通过率覆盖全部开发案例。±1°线仅供参考，运动验收要求全程RMSE≤1°及末段峰值≤1°，并非逐瞬时误差均≤1°。",
              "训练随机探索中的终止数、全部失败案例、逐检查点哈希和基线改善率见comparison.json；不得用确定性评价无硬失败推断训练探索无失败。",
              "本报告不更改历史成绩，不使用新留出，不代表R2初态扩展、换手、学生或实物部署通过。"]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: {label: {k: row[k] for k in ('steps', 'complete', 'passes', 'selection_score')} for label, row in result['checkpoints'].items()} for name, result in results.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
