"""Build the explicit Layer-1 evidence index; never select runs by success."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from humanlike_drive.env_runner.l1_suite import cases


def main():
    destination = ROOT / "results/physical_feasibility/l1_v1"
    base = Path("outputs/physical_feasibility")
    initial = base / "l1_v2_20261107/checks"
    parallel = base / "l1_v2_20261107/checks_parallel"
    core = base / "l1_v3_20261107/core"
    remaining = base / "l1_v3_20261107/remaining"
    rows = []
    for case in cases():
        name, group = case["name"], case["group"]
        if group == "extension":
            continue
        if name in ("positive180", "negative180"):
            directory = core / name
        elif name == "negative90" or group in ("robust", "diagnostic"):
            directory = remaining / name
        elif name in ("dual_hold", "positive15", "negative15", "positive30"):
            directory = initial / name
        else:
            directory = parallel / name
        evaluation_path = ROOT / directory / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        config_bytes = (ROOT / directory / "config.json").read_bytes()
        config = json.loads(config_bytes)
        if group == "robust":
            assert config["arguments"]["target_offset_mm"] == 1
            assert config["arguments"]["offset_mm"] == 0
        row = dict(
            **case,
            directory=directory.as_posix(),
            version=config["version"],
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            passed=evaluation["passed"],
            failed_checks=[k for k, v in evaluation["checks"].items() if not v],
            evaluation=evaluation,
        )
        rows.append(row)
    counts = {}
    for group in ("nominal", "load", "robust", "diagnostic"):
        selected = [r for r in rows if r["group"] == group]
        counts[group] = dict(
            total=len(selected), passed=sum(r["passed"] for r in selected)
        )
    payload = dict(
        scope="Simulation Layer-1 representative qualification; not full T1–T4, hardware or statistical reliability.",
        source_selection="Explicit case-to-directory mapping in scripts/summarize_layer1.py; v3 replaces fixed-time v2 negative90. Other v2 local/load results retain their actual versions; no claim that all 45 cases were rerun on v3.",
        excluded="Old l1_v2 checks_parallel perturb_* injected qpos at loaded contact and are excluded from target-registration tests; all development failures remain on disk.",
        counts=counts,
        versions=dict(Counter(r["version"] for r in rows)),
        cases=rows,
    )
    independent = {}
    for name in ("positive180", "negative180"):
        independent[name] = {}
        for filename in ("contact_CAD_audit.json", "action_replay_check.json"):
            source = ROOT / core / name / filename
            check = json.loads(source.read_text())
            assert check["passed"], f"Independent check failed: {source}"
            independent[name][filename] = check
            shutil.copy2(source, destination / f"{name}_{filename}")
    payload["independent_checks"] = independent
    (destination / "evidence_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# 层1逐例证据",
        "",
        "案例与源目录由脚本显式指定，不根据结果自动挑选。局部／负载沿用v2，完整±180°、负90°、目标标定扰动及诊断采用v3。两版差别是接管前最多5秒有界等待；源码与实际配置随每条轨迹保存。",
        "",
        "负载边界和扰动属于工程探测，失败不删除。无接触负对照应当跟踪失败；不能把它算作应通过的任务。",
        "",
        "| 案例 | 版本 | 物理门槛 | 失败项 | 原始证据 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f'| {row["name"]} | {row["version"]} | {"通过" if row["passed"] else "失败"} | {", ".join(row["failed_checks"]) or "—"} | [evaluation](../../../{row["directory"]}/evaluation.json) |'
        )
    (destination / "case_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(counts=counts, versions=payload["versions"]), indent=2))


if __name__ == "__main__":
    main()
