"""Validation gates for staged manipulation training."""
from dataclasses import replace
import json
import numpy as np

STAGES = ["grasp", "reach10", "reach30", "track", "robust"]


def gate(config, output, checkpoint, control, stage, case_runner):
    results = []
    for seed in [7001, 7002, 7003, 7004, 7005]:
        if stage in ("grasp", "reach10", "reach30"):
            amplitude = config.amplitude_deg
            targets = [0.] if stage == "grasp" else [-amplitude, 0., amplitude]
            for target in targets:
                case = replace(config, target_mode="step", episode_seconds=30)
                results.append(case_runner(case, output / f"hold_{target}_s{seed}", checkpoint, control, seed, target))
        else:
            for mode, frequency in [("sine", .02), ("sine", .10), ("sequence", .02)]:
                case = replace(config, target_mode=mode, frequency=frequency, episode_seconds=600)
                results.append(case_runner(case, output / f"{mode}_{frequency}_s{seed}", checkpoint, control, seed))
    if stage in ("grasp", "reach10", "reach30"):
        fraction = float(np.mean([r["hold_success"] for r in results]))
        passed = fraction >= .95
    else:
        fraction = float(np.mean([r["tracking_pass"] for r in results]))
        passed = all(r["tracking_pass"] for r in results)
    report = {"validation_passed": bool(passed), "success_fraction": fraction, "stage": stage, "cases": results,
              "seeds": [7001, 7002, 7003, 7004, 7005], "purpose": "curriculum validation, not final held-out acceptance"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate.json").write_text(json.dumps(report, indent=2))
    return passed
