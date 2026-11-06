"""Use the same deterministic reference for fixed-grasp and RL comparisons."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .common.evaluation import run_case as shared_case
from . import INTERFACE_VERSION
from .artifacts import require_matching_checkpoint
from .config import DEFAULT_CONFIG, load_config
from .env import ContactResidualEnv, HandsOnlyContactEnv


def run_case(config, output, model_path=None, control="policy", seed=1001, target_deg=None, video=False):
    cls = HandsOnlyContactEnv if control == "hands-only" else ContactResidualEnv
    if control == "zero":
        raise ValueError("Use nominal for zero contact residuals; the common motion controller is always explicit in v2")
    result = shared_case(config, output, model_path, control, seed, target_deg, video,
                         env_class=cls, metadata_checker=require_matching_checkpoint)
    result.update(interface_version=INTERFACE_VERSION, upstream_connected=False,
                  interpretation="Zero residuals with deterministic common motion" if control == "nominal"
                                 else "Learned contact residuals with deterministic common motion")
    (Path(output) / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", default="reach10")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--control", choices=["policy", "hands-only", "nominal"], default="policy")
    parser.add_argument("--mode", choices=["hold", "step", "sine", "sequence", "mixed"])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--target", type=float)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--frequency", type=float, default=.02)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8001, 8002, 8003, 8004, 8005])
    parser.add_argument("--suite", action="store_true", help="Separate final holds and four 600s trajectories per test seed")
    parser.add_argument("--randomization", type=float, default=0)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Test -amplitude, 0 and +amplitude; not final acceptance")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config, args.stage)
    config = replace(config, episode_seconds=args.seconds, frequency=args.frequency,
                     randomization_strength=args.randomization, no_contact=args.no_contact)
    if args.mode:
        config.target_mode = args.mode
    if args.amplitude is not None:
        config.amplitude_deg = args.amplitude
    config.validate()
    if args.suite:
        if args.pilot or len(set(args.seeds)) != len(args.seeds):
            parser.error("suite requires distinct test seeds and cannot combine with pilot")
        cases = []
        for seed in args.seeds:
            for target in [-30, -15, 0, 15, 30]:
                result = run_case(replace(config, target_mode="step", amplitude_deg=30, episode_seconds=30),
                                  args.output / f"hold_{target}_s{seed}", args.model, args.control, seed, target)
                cases.append(result)
                print(f"hold {target}, seed {seed}: success={result['hold_success']}", flush=True)
            for mode, frequency in [("sine", .02), ("sine", .10), ("sequence", .02), ("mixed", .02)]:
                result = run_case(replace(config, target_mode=mode, frequency=frequency,
                                          amplitude_deg=30, episode_seconds=600),
                                  args.output / f"{mode}_{frequency}_s{seed}", args.model, args.control, seed)
                cases.append(result)
                print(f"{mode}, seed {seed}: RMSE={result['rmse_deg']:.3f}", flush=True)
        holds = [c for c in cases if c["hold_success"] is not None]
        tracks = [c for c in cases if c["tracking_pass"] is not None]
        fraction = sum(c["hold_success"] and c["completed"] for c in holds) / len(holds)
        report = {"interface_version": INTERFACE_VERSION, "cases": cases, "hold_success_fraction": fraction,
                  "single_checkpoint_gate_passed": fraction >= .95 and all(c["tracking_pass"] for c in tracks),
                  "accepted_in_simulation": False, "hardware_ready": False,
                  "purpose": "Full single-checkpoint held-out suite; three independent training seeds still required"}
        if args.model:
            report["training_seed"] = require_matching_checkpoint(args.model, cases[0]["scene_sha256"])["seed"]
        (args.output / "suite_summary.json").write_text(json.dumps(report, indent=2))
        return
    if not args.pilot:
        print(json.dumps(run_case(config, args.output, args.model, args.control, args.seed, args.target, args.video), indent=2))
        return
    cases = []
    for target in [-config.amplitude_deg, 0, config.amplitude_deg]:
        result = run_case(replace(config, target_mode="step"), args.output / f"target_{target:g}",
                          args.model, args.control, args.seed, target, args.video)
        cases.append(result)
        print({k: result[k] for k in ["target_deg", "rmse_deg", "last_10s_rmse_deg", "hold_success", "early_failure"]}, flush=True)
    report = {"interface_version": INTERFACE_VERSION, "cases": cases,
              "pilot_hold_passed": all(c["hold_success"] and c["completed"] for c in cases),
              "accepted_in_simulation": False, "hardware_ready": False}
    (args.output / "pilot.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
