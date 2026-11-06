"""Run the three frozen-controller comparisons sequentially, without training."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/exploration_stage1")
    parser.add_argument("--hold-model", type=Path, default=ROOT / "runs/exploration_stage1/hold_s7_pilot/final.pt")
    parser.add_argument("--reach-model", type=Path, default=ROOT / "runs/exploration_stage1/reach3_s7_pilot/final.pt")
    args = parser.parse_args()
    for name, model in (("zero", None), ("hold", args.hold_model), ("reach3", args.reach_model)):
        for split in ("nominal", "randomized"):
            output = args.output.resolve() / f"{name}_{split}"
            command = [sys.executable, "-m", "wheel_manipulation.exploration.evaluate",
                       "--stage", "reach3", "--seconds", "15", "--amplitude", "3",
                       "--output", str(output)]
            command += ["--control", "zero"] if model is None else ["--model", str(model.resolve())]
            command += (["--cases", "hold", "step_positive", "step_negative", "sine", "sequence",
                         "--seeds", "1007"] if split == "nominal" else
                        ["--cases", "randomized", "--seeds", "1007", "1017", "1027"])
            if name == "reach3" and split == "nominal":
                command += ["--video-case", "sequence"]
            print(f"Starting {name}/{split}: {output}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
