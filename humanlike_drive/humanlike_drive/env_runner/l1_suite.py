"""Reproduce the Layer-1 qualification matrix using a frozen controller.

Each rollout starts from the same verified preparation, uses finite drives, and
writes a fresh directory. Boundary probes are observations, not required passes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def cases():
    rows = []

    def add(name, group, options, expectation="pass"):
        rows.append(
            dict(name=name, group=group, options=options, expectation=expectation)
        )

    add("dual_hold", "nominal", ["--case", "hold", "--both-hands", "--load", "-.1"])
    for degree in (15, 30, 90, 180):
        for label, sign in (("positive", 1), ("negative", -1)):
            add(
                f"{label}{degree}",
                "nominal",
                [
                    "--case",
                    "relay" if degree > 30 else "roundtrip",
                    "--angle",
                    str(sign * degree),
                ],
            )
    for sign in (1, -1):
        add(
            f"crosszero_{sign:+d}",
            "nominal",
            ["--case", "crosszero", "--angle", str(sign * 30)],
        )
    for side in ("left", "right"):
        for load in (-0.05, 0.05, -0.1, 0.1, -0.2, 0.2, -0.4, 0.4):
            add(
                f"hold_{side}_{load:+.2f}",
                "load",
                ["--case", "hold", "--side", side, "--load", str(load)],
                "pass" if abs(load) <= 0.1 else "observe",
            )
    for condition, options in (
        ("friction", ["--friction", ".7"]),
        ("load", ["--load", "-.12"]),
        ("combined", ["--friction", ".7", "--load", "-.12"]),
    ):
        for seed in range(5):
            add(
                f"perturb_{condition}_seed{seed}",
                "robust",
                [
                    "--angle",
                    "15",
                    "--target-offset-mm",
                    "1",
                    "--seed",
                    str(seed),
                    *options,
                ],
                "observe",
            )
    for sign in (1, -1):
        add(
            f"fingers_less_{sign:+d}",
            "diagnostic",
            ["--angle", str(sign * 30), "--finger-closure", ".10"],
            "observe",
        )
    add(
        "no_wheel_contact",
        "diagnostic",
        ["--angle", "15", "--negative-control"],
        "tracking_failure",
    )
    for sign in (1, -1):
        add(
            f"extension_{sign * 360:+d}",
            "extension",
            ["--case", "relay", "--angle", str(sign * 360)],
            "observe",
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--grasp", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    p.add_argument(
        "--groups",
        nargs="+",
        choices=("nominal", "load", "robust", "diagnostic", "extension"),
        default=["nominal", "load", "robust", "diagnostic"],
    )
    p.add_argument("--only", nargs="+")
    p.add_argument("--exclude", nargs="+", default=[])
    args = p.parse_args()
    for filename in ("evaluation.json", "contact_CAD_audit.json"):
        gate = args.grasp / filename
        if not gate.exists() or not json.loads(gate.read_text()).get("passed"):
            p.error(f"Preparation gate missing or failed: {gate}")
    selected = [c for c in cases() if c["group"] in args.groups]
    unknown = set(args.exclude) - {c["name"] for c in selected}
    if unknown:
        p.error(f"Unknown excluded cases: {unknown}")
    selected = [c for c in selected if c["name"] not in args.exclude]
    if args.only:
        known = {c["name"] for c in selected}
        if set(args.only) - known:
            p.error(f"Unknown selected cases: {set(args.only) - known}")
        selected = [c for c in selected if c["name"] in args.only]
    if args.output.exists():
        raise FileExistsError("Use a fresh suite directory")
    args.output.mkdir(parents=True)
    config = args.config.read_bytes()
    frozen = args.output.resolve() / "controller.json"
    frozen.write_bytes(config)
    (args.output / "suite.json").write_text(
        json.dumps(
            dict(
                cases=selected,
                controller_sha256=hashlib.sha256(config).hexdigest(),
                grasp=str(args.grasp.resolve()),
                scope="engineering qualification and finite boundary probes; not a statistical success-rate estimate",
            ),
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(case):
        output = args.output / case["name"]
        cmd = [
            sys.executable,
            "-m",
            "humanlike_drive.env_runner.physical_feasibility",
            "run",
            "--config",
            str(frozen),
            "--grasp",
            str(args.grasp.resolve()),
            "--output",
            str(output),
            *case["options"],
        ]
        with (args.output / (case["name"] + ".log")).open("w") as log:
            process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        result = (
            json.loads((output / "evaluation.json").read_text())
            if (output / "evaluation.json").exists()
            else None
        )
        complete = process.returncode == 0 and result is not None
        expectation = case["expectation"]
        accepted = complete and (
            result["passed"]
            if expectation == "pass"
            else (
                not result["passed"] and not result["checks"]["motion_RMSE_le_2deg"]
                if expectation == "tracking_failure"
                else True
            )
        )
        row = dict(
            **case,
            returncode=process.returncode,
            accepted=bool(accepted),
            physics_passed=result["passed"] if result else None,
            failed_checks=(
                [k for k, v in result["checks"].items() if not v]
                if result
                else ["process"]
            ),
        )
        print(
            f'{case["name"]}: physics={row["physics_passed"]}, expectation={accepted}, failed={row["failed_checks"]}',
            flush=True,
        )
        return row

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, c) for c in selected]
        for future in as_completed(futures):
            results.append(future.result())
            (args.output / "suite_results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    if not all(r["accepted"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
