"""Aggregate full held-out suites; never substitute one checkpoint for three seeds."""
import argparse
import json
from pathlib import Path


def aggregate(paths):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    seeds = [r.get("training_seed") for r in reports]
    complete = True
    for report in reports:
        cases = report.get("cases", [])
        hold_seeds = {c["seed"] for c in cases if c.get("hold_success") is not None}
        hold_angles = {c["target_deg"] for c in cases if c.get("hold_success") is not None}
        tracks = [c for c in cases if c.get("tracking_pass") is not None]
        required_tracks = {("sine", .02), ("sine", .10), ("sequence", .02), ("mixed", .02)}
        per_seed_complete = all(
            {c["target_deg"] for c in cases if c["seed"] == seed and c.get("hold_success") is not None}
            == {-30, -15, 0, 15, 30}
            and required_tracks.issubset({(c["task"]["target_mode"], c["task"]["frequency"])
                                         for c in tracks if c["seed"] == seed})
            for seed in hold_seeds)
        complete &= (len(hold_seeds) >= 5 and hold_angles == {-30, -15, 0, 15, 30}
                     and per_seed_complete and len(tracks) >= 20
                     and all(c["task"]["episode_seconds"] >= 600 for c in tracks))
    passed = (len(reports) >= 3 and None not in seeds and len(set(seeds)) >= 3 and complete
              and all(r.get("single_checkpoint_gate_passed", False) for r in reports))
    return {"accepted_in_simulation": bool(passed), "hardware_ready": False,
            "training_seeds": seeds, "complete_suites": bool(complete),
            "reports": [str(p) for p in paths],
            "criteria": "Three independent training seeds; >=95% hold success; all 600s tracking RMSE<=2deg/max<=5deg"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/acceptance.json"))
    args = parser.parse_args()
    report = aggregate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["accepted_in_simulation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
