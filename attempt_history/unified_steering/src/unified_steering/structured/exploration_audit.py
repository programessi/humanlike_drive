"""Paired action-noise diagnosis; never updates or saves policy weights."""
import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from .env import TaskConfig
from .evaluate import load_teacher
from .parking import SupportTeacherEnv


def trial(job):
    path, mode, task, noise_seed = job
    torch.set_num_threads(1)
    model = load_teacher(path)
    # SB3 loading restores its training seed; sample seeds must be set after it.
    torch.manual_seed(noise_seed)
    if mode == "reduced_wrist_noise":
        with torch.no_grad():
            model.policy.log_std[:12] += np.log(.4)
    holding = task.endswith("hold")
    config = TaskConfig(target_mode="hold" if holding else "sequence",
                        episode_seconds=3. if holding else 10.)
    env = SupportTeacherEnv(config)
    try:
        obs, _ = env.reset(seed=1007, options=dict(free_side=1 if task.startswith("left") else 0,
                                                  mode="hold" if holding else "step", direction=-1))
        c = env.controller
        lo, hi = c.hand_limits[:,0]+c.config.joint_margin, c.hand_limits[:,1]-c.config.joint_margin
        center = np.clip(c.nominal_closed, lo, hi)
        measured = env.data.qpos[c.hand_q]
        hold_action = np.clip(np.where(measured<center,(measured-center)/np.maximum(center-lo,1e-8),
                                       (measured-center)/np.maximum(hi-center,1e-8)),-1,1)
        errors = []
        peak = free_peak = 0.
        for _ in range(round(config.episode_seconds*config.policy_hz)):
            action = model.predict(obs, deterministic=mode not in ("original_noise","reduced_wrist_noise"))[0]
            if mode in ("zero_wrists","hold_all"):
                action[:12] = 0.
            if mode in ("hold_hands","hold_all"):
                action[12:] = hold_action
            obs, _, terminated, truncated, info = env.step(action)
            errors.append(info["error_deg"])
            peak = max(peak,info["physics_peak_error_deg"])
            free_peak = max(free_peak,info["physics_free_hand_force_peak_n"])
            if terminated or truncated:
                break
        return dict(mode=mode, task=task, noise_seed=noise_seed, rmse_deg=float(np.sqrt(np.mean(np.square(errors)))),
                    peak_error_deg=peak, complete=not terminated, reason=info["termination_reason"],
                    free_hand_force_peak_n=free_peak)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interventions", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    modes = ("deterministic","zero_wrists","hold_hands","hold_all") if args.interventions else ("deterministic","original_noise","reduced_wrist_noise")
    tasks = ("left_hold","right_hold") if args.interventions else ("left_hold","right_negative")
    seeds = (9007,) if args.interventions else (9007,9017,9027)
    jobs = [(str(args.model),mode,task,seed) for mode in modes for task in tasks for seed in seeds]
    with multiprocessing.get_context("fork").Pool(4) as pool:
        cases = list(pool.imap(trial,jobs))
    groups = []
    for mode in modes:
        for task in tasks:
            selected = [r for r in cases if r["mode"]==mode and r["task"]==task]
            groups.append(dict(mode=mode,task=task,mean_rmse_deg=float(np.mean([r["rmse_deg"] for r in selected])),
                               mean_peak_deg=float(np.mean([r["peak_error_deg"] for r in selected])),
                               complete=sum(r["complete"] for r in selected),cases=len(selected)))
    result = dict(model=str(args.model),scope="Frozen weights; paired sampling diagnosis only, not an RL training comparison",
                  original_wrist_std="checkpoint value approx 0.25",reduced_wrist_std="0.4 times original approx 0.10",
                  cases=cases,groups=groups)
    (args.output/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(groups,indent=2))
