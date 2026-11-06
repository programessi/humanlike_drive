"""Bounded sequential control training with frozen gates; not all of stage 1.

Only A is automated here. Passing it does not imply single-hand transfer,
regrasp, distillation or real deployment. Every block saves its own artifacts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ..paths import ROOT


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "stage1_passed": False, "hardware_ready": False,
             "scope": "control curriculum A only; single-hand and useful regrasp gates remain",
             "seed": args.seed, "envs": args.envs, "max_blocks": args.blocks,
             "steps_per_block": args.steps_per_block, "amplitudes_deg": args.amplitudes,
             "blocks": [], "started_unix": time.time()}

    def save():
        tmp = args.output / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2)); tmp.replace(args.output / "status.json")

    def command(argv, log):
        with log.open("w") as f:
            f.write("Argument vector: " + json.dumps(argv) + "\n"); f.flush()
            subprocess.run([sys.executable, *argv], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)

    amplitude_index, parent, zeros = 0, None, {}
    save()
    try:
        for block in range(1, args.blocks + 1):
            amplitude = args.amplitudes[amplitude_index]
            train_dir = args.output / f"block_{block:02d}_A{amplitude:g}"
            evaluation = args.output / f"block_{block:02d}_evaluation"
            state.update(active_block=block, phase="training", active_amplitude_deg=amplitude)
            save(); print(json.dumps({"block": block, "phase": "training", "amplitude": amplitude}), flush=True)
            argv = ["-m", "wheel_manipulation.regrasp.train", "--curriculum", "A",
                    "--steps", str(args.steps_per_block), "--envs", str(args.envs), "--seed", str(args.seed + block - 1),
                    "--amplitude", str(amplitude), "--seconds", "20", "--reward-scale", ".05",
                    "--output", str(train_dir)]
            if parent:
                argv.extend(["--resume", str(parent)])
            command(argv, args.output / f"block_{block:02d}_training.log")
            parent = train_dir / "teacher.zip"
            state.update(phase="frozen_evaluation"); save()
            base = ["-m", "wheel_manipulation.regrasp.evaluate", "--cases", "hold", "positive", "negative", "sequence",
                    "--seeds", "1007", "1017", "1027", "--seconds", "20", "--amplitude", str(amplitude)]
            command(base + ["--control", "teacher", "--model", str(parent), "--output", str(evaluation)],
                    args.output / f"block_{block:02d}_evaluation.log")
            if amplitude not in zeros:
                zero = args.output / f"zero_A{amplitude:g}"
                command(base + ["--control", "zero", "--controller-from", str(train_dir / "config.json"), "--output", str(zero)],
                        args.output / f"zero_A{amplitude:g}.log")
                zeros[amplitude] = str(zero)
            result = json.loads((evaluation / "summary.json").read_text())
            passed = all(case["tracking_gate_passed"] for case in result["cases"])
            record = {"block": block, "amplitude_deg": amplitude, "model": str(parent),
                      "evaluation": str(evaluation), "matched_zero": zeros[amplitude],
                      "passed_tracking_gate": passed,
                      "mean_rmse_deg": sum(c["rmse_deg"] for c in result["cases"]) / len(result["cases"]),
                      "complete": sum(c["full_duration"] for c in result["cases"]),
                      "n": len(result["cases"])}
            state["blocks"].append(record); save(); print(json.dumps(record), flush=True)
            comparison = args.output / f"block_{block:02d}_comparison"
            command(["scripts/summarize_regrasp_stage1.py", "--inputs", zeros[amplitude], str(evaluation),
                     "--labels", "zero", f"teacher_block_{block}", "--output", str(comparison)],
                    args.output / f"block_{block:02d}_comparison.log")
            if passed:
                amplitude_index += 1
                if amplitude_index == len(args.amplitudes):
                    state["status"] = "control_courses_passed_pending_single_hand_and_regrasp"
                    break
        else:
            state["status"] = "control_budget_exhausted_requires_diagnosis"
    except BaseException as error:
        state.update(status="interrupted_or_failed", error=repr(error)); save()
        raise
    state.update(phase="finished", ended_unix=time.time()); save()
    print(json.dumps({"status": state["status"], "stage1_passed": False}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps-per-block", type=int, default=131072)
    p.add_argument("--amplitudes", type=float, nargs="+", default=[5., 10.])
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    if not 1 <= a.envs <= 8 or a.blocks <= 0 or a.steps_per_block <= 0 or not all(0 < v <= 30 for v in a.amplitudes):
        p.error("Use positive bounded CPU training budgets and small-angle control goals")
    a.output = a.output.resolve()
    run(a)
