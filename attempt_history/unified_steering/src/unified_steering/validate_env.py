"""Reproducible M1 regression, stress rollout and CPU throughput report; no RL."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
import yaml

from .audit import ROOT, digest, write_json
from .env import SteeringEnv


def hold_regression(env, output):
    cases = []
    with (output / "holds.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["support", "state", "time_s", "error_deg", "collision_n"])
        writer.writeheader()
        for mode, count in (("left", 5), ("right", 5), ("both", 1)):
            for index in range(count):
                _, initial = env.reset(options={"support": mode, "state_index": index})
                env.set_reference([initial["initial_angle_rad"]])
                peak = 0.
                reason = None
                for step in range(60):
                    _, _, term, trunc, info = env.step(env.hold_action())
                    error = float(abs(np.rad2deg(info["error_rad"])))
                    peak = max(peak, error)
                    writer.writerow(dict(support=mode, state=index, time_s=(step+1)*env.period,
                                         error_deg=error, collision_n=info["collision_peak_n"]))
                    if term or trunc:
                        reason = info["termination_reason"] or "timeout_before_3s"
                        break
                cases.append(dict(support=mode, state=index, max_drift_deg=peak, reason=reason,
                                  passed=reason is None and peak <= 2.))
    return cases


def rollout(config, steps, seed, stream=None):
    env = SteeringEnv(config)
    rng = np.random.default_rng(seed)
    resets = 0
    terminated = Counter()
    truncated = 0
    physics_ticks = 0
    normal_steps = 0
    max_lead = 0.
    intervention_ticks = 0
    force_peak_ratio = 0.
    sample = []
    episode = 0
    done = True
    # Initialization/model load excluded; runtime snapshot resets included.
    begin = time.perf_counter()
    for call in range(steps):
        if done:
            env.reset(seed=seed+resets)
            resets += 1
            episode += 1
            # Every fifth episode deliberately explores full-range random actions.
            exploratory = episode % 5 == 0
        action = env.hold_action().astype(float)
        if exploratory:
            action = rng.uniform(-1, 1, 26)
        elif episode % 2 == 0:
            action[:14] = rng.normal(0, .015, 14)
            action[14:] = np.clip(action[14:] + rng.normal(0, .01, 12), -1, 1)
        obs, reward, term, trunc, info = env.step(action)
        if not env.observation_space.contains(obs) or not np.isfinite(reward) or not info["observation_valid"]:
            raise RuntimeError(f"Nonfinite/invalid rollout at call {call}")
        ticks = info["physics_steps"]
        physics_ticks += ticks
        if not term:
            if ticks != 25 or abs(info["elapsed_seconds"]-.05) > 1e-12:
                raise RuntimeError("Incorrect normal step duration")
            normal_steps += 1
        s, e = env.sim, env.executor
        if ticks:
            # This is the measured post-step servo tracking error; the envelope is
            # enforced against the pre-step measurement, so dynamics can add error.
            lead = float(np.max(np.abs(e.target[:14]-s.data.qpos[s.qidx[:14]])))
            max_lead = max(max_lead, lead)
            caps = s.model.actuator_forcerange[:, 1]
            force_peak_ratio = max(force_peak_ratio, float(np.max(np.abs(s.data.actuator_force)/caps)))
        intervention_ticks += info["execution_interventions"]
        if term:
            terminated[info["termination_reason"]] += 1
        truncated += int(trunc)
        done = term or trunc
        row = dict(call=call, episode=episode, exploratory=exploratory, reward=reward,
                   physics_steps=ticks, error_deg=float(np.rad2deg(info["error_rad"])),
                   terminated=term, truncated=trunc, reason=info["termination_reason"],
                   interventions=info["execution_interventions"])
        if stream:
            stream.writerow(row)
        if call % 1000 == 0:
            sample.append(row)
            if stream:
                print(f"Rollout {call}/{steps}, resets={resets}, failures={sum(terminated.values())}", flush=True)
    duration = time.perf_counter()-begin
    return dict(control_calls=steps, resets=resets, physics_steps=physics_ticks,
                normal_full_steps=normal_steps, terminal_reasons=dict(terminated), timeouts=truncated,
                finite_observations_rewards=True, wall_seconds=duration,
                control_calls_per_second=steps/duration, physics_steps_per_second=physics_ticks/duration,
                simulated_seconds_per_wall_second=physics_ticks*env.dt/duration,
                max_poststep_target_error_deg=float(np.rad2deg(max_lead)),
                max_actuator_force_cap_ratio=force_peak_ratio,
                execution_intervention_fraction=intervention_ticks/max(1, physics_ticks), samples=sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--parallel-workers", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.parallel_workers < 1:
        parser.error("Positive steps/workers required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/m1.yaml"
    config = yaml.safe_load(config_path.read_text())
    report = dict(kind="M1_environment_validation_not_RL", created_at_utc=datetime.now(timezone.utc).isoformat(),
                  python_version=platform.python_version(), mujoco_version=mujoco.__version__,
                  machine=platform.processor(), config_sha256=digest(config_path), scene_sha256=digest(ROOT / config['scene']),
                  training_started=False,
                  source_sha256={str(path.relative_to(ROOT)): digest(path)
                                 for path in sorted((ROOT / 'src/unified_steering').glob('*.py'))})
    env = SteeringEnv(config)
    report["hold_regression"] = hold_regression(env, output)
    print("Hold regression:", sum(c['passed'] for c in report['hold_regression']), "/11", flush=True)
    write_json(output / "summary.json", report)
    # Stress horizon shortened to five seconds to obtain >=100 reset transitions.
    smoke = copy.deepcopy(config)
    smoke['episode_seconds'] = 5.
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "stress_config.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
    with (output / "rollout.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["call", "episode", "exploratory", "reward", "physics_steps",
                                                   "error_deg", "terminated", "truncated", "reason", "interventions"])
        writer.writeheader()
        report['stress'] = rollout(smoke, args.steps, 20261026, writer)
    print("Stress:", report['stress'], flush=True)
    write_json(output / "summary.json", report)
    # Same workload per worker; process/model startup reported separately.
    context = multiprocessing.get_context("spawn")
    begin = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=context) as pool:
        workers = list(pool.map(rollout, [smoke]*args.parallel_workers, [200]*args.parallel_workers,
                                [31+i for i in range(args.parallel_workers)]))
    wall = time.perf_counter()-begin
    report['parallel_benchmark'] = dict(workers=args.parallel_workers, calls_per_worker=200,
                                       wall_seconds_including_startup=wall,
                                       aggregate_calls_per_second_including_startup=200*args.parallel_workers/wall,
                                       worker_reports=workers)
    report['passed'] = (all(c['passed'] for c in report['hold_regression'])
                        and report['stress']['finite_observations_rewards']
                        and report['stress']['control_calls'] >= 10000 and report['stress']['resets'] >= 100
                        and report['stress']['max_actuator_force_cap_ratio'] <= 1+1e-6)
    write_json(output / "summary.json", report)
    print("Passed:", report['passed'], "Output:", output, flush=True)


if __name__ == '__main__':
    main()
