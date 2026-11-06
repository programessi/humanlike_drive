"""Bounded demonstration pretraining and DAgger on real simulated rollouts.

No scripted feedback or action masking is used in learned-policy evaluation.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .curriculum import balanced_condition
from .demonstrator import Demonstrator, DemonstratorConfig
from .env import TaskConfig
from .evaluate import load_teacher, run_case, suite
from .parking import SCREENED_RESETS, SupportTeacherEnv
from .train import stage_suite, write_json


def task_config(protocol):
    return TaskConfig(episode_seconds=protocol["episode_seconds"], hand_action_mode="leaky",
        hand_leak_alpha=.995, hand_joint_lead_rad=protocol["hand_joint_lead_rad"], angle_reward_sigma_deg=2.)


def options_for(protocol, episode):
    selected = balanced_condition(episode, protocol["seed"])
    if protocol["parking"] == "screened" and selected["free_side"] >= 0:
        name = ("nominal", *SCREENED_RESETS)[(episode // 2) % 3]
        selected.update(SCREENED_RESETS.get(name, {}))
    return selected


def collect_job(job):
    model_path, protocol, episode, beta = job
    torch.set_num_threads(1)
    env = SupportTeacherEnv(task_config(protocol))
    selected = options_for(protocol, episode)
    amplitude = selected.pop("training_amplitude_deg")
    env.config.amplitude_deg = amplitude
    rng = np.random.default_rng(protocol["training_seed_start"]+episode)
    model = load_teacher(model_path) if beta < 1 else None
    actors, privileged, labels = [], [], []
    try:
        obs, _ = env.reset(seed=protocol["training_seed_start"]+episode, options=selected)
        expert = Demonstrator(env, DemonstratorConfig(**protocol["demonstrator"]))
        errors, peaks, free_peaks = [], [], []
        for _ in range(round(protocol["episode_seconds"]*env.config.policy_hz)):
            target, _ = expert.predict(obs)
            actors.append(obs["actor"].copy()); privileged.append(obs["privileged"].copy()); labels.append(target)
            action = target if rng.random() < beta else model.predict(obs, deterministic=True)[0]
            obs, _, done, truncated, info = env.step(action)
            errors.append(info["error_deg"]); peaks.append(info["physics_peak_error_deg"])
            free_peaks.append(info["physics_free_hand_force_peak_n"] if env.free_side >= 0 else 0.)
            if done or truncated:
                break
        record = dict(episode=episode, seed=protocol["training_seed_start"]+episode,
            expert_probability=beta, steps=len(errors), seconds=env.time, complete=not done and truncated,
            termination_reason=info["termination_reason"], free_side=env.free_side,
            mode=env.episode_mode, direction=env.episode_direction, amplitude_deg=amplitude,
            reset_options=selected, rmse_deg=float(np.sqrt(np.mean(np.square(errors)))),
            peak_deg=max(peaks), free_force_peak_n=max(free_peaks))
        return np.asarray(actors,np.float32), np.asarray(privileged,np.float32), np.asarray(labels,np.float32), record
    finally:
        env.close()


def fit(model, dataset, epochs, protocol):
    actors, privileged, labels = (torch.from_numpy(dataset[k]) for k in ("actor", "privileged", "action"))
    # Train every mean-action channel and its encoder; no hand/arm head is frozen.
    parameters = list(model.policy.features_extractor.parameters()) + list(model.policy.mlp_extractor.actor.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=protocol["bc_learning_rate"])
    n = len(actors)
    losses = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        total = 0.
        for ids in torch.randperm(n).split(protocol["bc_batch_size"]):
            mean = model.policy.get_distribution({"actor":actors[ids], "privileged":privileged[ids]}).distribution.mean
            loss = (mean-labels[ids]).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            total += loss.item()*len(ids)
        losses.append(total/n)
        if epoch % 5 == 0 or epoch == epochs-1:
            print(json.dumps(dict(bc_epoch=epoch, samples=n, loss=losses[-1])), flush=True)
    model.policy.set_training_mode(False)
    # BC changed weights outside PPO. Avoid resuming stale Adam moments later.
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-5, eps=1e-5)
    return losses


def run(args):
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.protocol,args.output/"protocol.json")
    root = Path(__file__).resolve().parent
    for source in root.rglob("*.py"):
        destination = args.output/"source_snapshot"/source.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,destination)
    random.seed(protocol["seed"]); np.random.seed(protocol["seed"]); torch.manual_seed(protocol["seed"])
    parent = Path(protocol["parent"])
    model = load_teacher(parent)
    metadata = dict(stage="R2", interface=protocol["interface"], parent=str(parent.resolve()),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(), task=asdict(task_config(protocol)),
        training_method="scripted-demonstration-BC-and-DAgger", status="training", started_unix=time.time(),
        r2_passed=False, privileged_teacher=True, script_used_at_evaluation=False)
    write_json(args.output/"config.json",metadata)
    model.save(args.output/"initial")
    baseline = stage_suite(None,"R2",protocol["parking"],tuple(protocol["development_seeds"]),
        args.output/"fixed_targets",hand_mode="leaky",hand_alpha=.995,angle_sigma=2.,
        include_reverse_sequence=True,hand_lead=protocol["hand_joint_lead_rad"])
    data = {key: [] for key in ("actor", "privileged", "action")}
    index, episodes, offset, consecutive, best = [], [], 0, 0, float("inf")
    for round_index, plan in enumerate(protocol["rounds"]):
        model_path = args.output/("initial.zip" if round_index == 0 else f"round_{round_index-1:02d}.zip")
        jobs = [(model_path,protocol,offset+i,plan["expert_probability"]) for i in range(plan["episodes"])]
        with mp.get_context("fork").Pool(4) as pool:
            for actor, privileged, action, record in pool.imap(collect_job,jobs):
                for key,array in zip(data,(actor,privileged,action)): data[key].append(array)
                episodes.append(record)
                with (args.output/"episodes.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                print("COLLECT "+json.dumps(record),flush=True)
        offset += plan["episodes"]
        dataset = {key: np.concatenate(parts) for key,parts in data.items()}
        losses = fit(model,dataset,plan["epochs"],protocol)
        model.save(args.output/f"round_{round_index:02d}")
        result = stage_suite(model,"R2",protocol["parking"],tuple(protocol["development_seeds"]),
            args.output/"development"/f"round_{round_index:02d}", baselines=baseline,
            hand_mode="leaky",hand_alpha=.995,angle_sigma=2.,include_reverse_sequence=True,
            hand_lead=protocol["hand_joint_lead_rad"])
        rows = result["cases"]
        stable = all(r["complete"] for r in rows) and all(r["peak_error_deg"]<=1. for r in rows if r["case"]=="hold") and all((r["free_hand_peak_force_n"] or 0)<=.2 for r in rows)
        score = max(r["rmse_deg"] for r in rows if r["case"]!="hold") + (0. if stable else 1000.)
        if score < best: best=score; model.save(args.output/"best")
        consecutive = consecutive+1 if result["passed"] else 0
        record = dict(round=round_index, samples=len(dataset["actor"]), expert_probability=plan["expert_probability"],
            bc_initial_loss=losses[0],bc_final_loss=losses[-1], complete=sum(r["complete"] for r in rows),
            passes=sum(r["passed"] for r in rows), total=len(rows),passed=result["passed"],
            score=score,consecutive_passes=consecutive)
        index.append(record);write_json(args.output/"development_index.json",index)
        print("GATE "+json.dumps(record),flush=True)
        if consecutive>=2:
            model.save(args.output/"qualified_development");break
    model.save(args.output/"final")
    np.savez_compressed(args.output/"dataset.npz",**dataset)
    metadata.update(status="development_qualified" if consecutive>=2 else "development_requires_repair",
        actual_interaction_steps=sum(r["steps"] for r in episodes), ended_unix=time.time(),
        training_episodes=len(episodes),qualified=consecutive>=2)
    write_json(args.output/"config.json",metadata)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path("configs/r22_demonstration_bootstrap_v3.json"))
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
