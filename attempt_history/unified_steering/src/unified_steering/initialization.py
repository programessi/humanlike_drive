"""Versioned snapshots: expensive grasp preparation is offline, not a reset servo."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .audit import ROOT, Fixture, digest, verified_scene, write_json


class InitialStates:
    def __init__(self, directories):
        self.states = []
        for directory in directories:
            directory = ROOT / directory
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["scene_sha256"] != digest(ROOT / "assets/scene.xml"):
                raise ValueError("Initial state scene mismatch")
            for entry in manifest["states"]:
                path = directory / entry["file"]
                if digest(path) != entry["sha256"]:
                    raise ValueError(f"Initial state hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as state:
                    self.states.append((entry, {key: state[key].copy() for key in state.files}))

    def sample(self, rng, mode, index=None):
        choices = [item for item in self.states if mode == "mixed" or item[0]["support"] == mode]
        if not choices:
            raise ValueError(f"No validated states for {mode}")
        return choices[int(rng.integers(len(choices))) if index is None else index]


def prepare_both(output):
    config = yaml.safe_load((ROOT / "configs/s00.yaml").read_text())
    scene, _ = verified_scene(config)
    fixture = Fixture(scene, config)
    errors = fixture.reset()
    metrics = fixture.metrics(float(fixture.data.qpos[fixture.wheel_q]))
    if max(errors) > .002 or min(metrics["left_contact_force_norm_sum_n"], metrics["right_contact_force_norm_sum_n"]) <= .2:
        raise RuntimeError("Both-hand initialization failed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "both_00.initial.npz"
    d = fixture.data
    np.savez_compressed(path, qpos=d.qpos, qvel=d.qvel, act=d.act, qacc_warmstart=d.qacc_warmstart,
                        target=fixture.target, support="both", requested_angle_deg=0.)
    write_json(output / "manifest.json", {"kind": "offline_prepared_both_hand_state", "scene_sha256": digest(scene),
               "preparation": "S00 finite capped joint PD with centering; no wheel jig", "initial_metrics": metrics,
               "states": [{"file": path.name, "sha256": digest(path), "support": "both"}]})
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/initial_states/m1_both_v1")
    prepare_both(parser.parse_args().output)
