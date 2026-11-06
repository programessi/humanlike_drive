"""Sequential v2 stages and seeds; precision gates prevent automatic promotion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .common.gates import STAGES, gate as shared_gate
from . import INTERFACE_VERSION
from .config import DEFAULT_CONFIG, load_config
from .evaluate import run_case


def gate(config, output, checkpoint, control, stage):
    return shared_gate(config, output, checkpoint, control, stage, case_runner=run_case)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--steps-per-stage", type=int, default=200000)
    parser.add_argument("--hands-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps_per_stage < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive budgets and distinct training seeds are required")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = {"interface_version": INTERFACE_VERSION, "training_seeds": args.seeds,
                "runs": [], "accepted_in_simulation": False, "hardware_ready": False}
    for seed in args.seeds:
        resume = None
        for stage in STAGES:
            output = args.output / f"s{seed}" / stage
            if (output / "config.json").exists():
                raise ValueError(f"Existing run must not be overwritten: {output}")
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "wheel_manipulation.train", "--config", str(args.config),
                       "--stage", stage, "--steps", str(args.steps_per_stage), "--seed", str(seed), "--output", str(output)]
            if resume:
                command.extend(["--resume", str(resume)])
            if args.hands_only:
                command.append("--hands-only")
            with (output / "training.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            checkpoint = output / "best/best_model.zip"
            if not checkpoint.exists():
                checkpoint = output / "final_model.zip"
            config, _ = load_config(args.config, stage)
            passed = gate(config, output / "validation", checkpoint,
                          "hands-only" if args.hands_only else "policy", stage)
            progress["runs"].append({"seed": seed, "stage": stage, "validation_passed": passed,
                                     "checkpoint": str(checkpoint)})
            (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
            if not passed:
                print(f"Seed {seed}, stage {stage}: gate failed, no promotion", flush=True)
                break
            resume = checkpoint
    progress["curriculum_complete"] = all(any(r["seed"] == seed and r["stage"] == "robust" and r["validation_passed"]
                                             for r in progress["runs"]) for seed in args.seeds)
    (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
