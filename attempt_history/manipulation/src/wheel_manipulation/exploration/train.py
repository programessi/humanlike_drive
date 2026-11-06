"""Single-environment, CPU GRU PPO pilot with 64-step truncated BPTT."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from ..common.resources import training_slot
from ..paths import ROOT
from . import INTERFACE_VERSION, ACTOR_DIM, ACTION_DIM, HIDDEN_DIM
from .controller import WristControlConfig
from .env import CentralizedWheelEnv, ExplorationConfig
from .policy import ActorCritic, export_actor, generalized_advantages, squashed_log_prob

DEFAULT_CONFIG = ROOT / "configs/exploration_stage1.yaml"


def load_config(path=DEFAULT_CONFIG, stage="reach3"):
    settings = yaml.safe_load(Path(path).read_text())
    if settings["interface"] != INTERFACE_VERSION:
        raise ValueError("Incompatible exploration configuration")
    config = ExplorationConfig(**settings["task"], controller=WristControlConfig(**settings["controller"]))
    if stage == "hold":
        config.amplitude_deg, config.target_mode = 0., "hold"
    elif stage == "reach10":
        config.amplitude_deg = 10.
    elif stage != "reach3":
        raise ValueError("Stage must be hold, reach3 or reach10")
    config.validate()
    return config, settings


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("Checkpoint belongs to a different action/observation interface")
    network = ActorCritic()
    network.load_state_dict(checkpoint["network"])
    return network, checkpoint


def update(network, optimizer, buffer, settings):
    p = settings
    actor_obs, privileged, initial_h, starts, latents, old_logp, values, rewards, next_values, dones = [tensor(x) for x in buffer]
    advantages, returns = generalized_advantages(rewards, values, next_values, dones, p["gamma"], p["gae_lambda"])
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    length, sequence = len(rewards), p["sequence_steps"]
    if length % sequence:
        raise ValueError("Rollout length must divide into whole recurrent sequences")
    statistics = []
    for _ in range(p["epochs"]):
        order = np.random.permutation(np.arange(0, length, sequence))
        for offset in range(0, len(order), p["sequences_per_batch"]):
            indices = order[offset:offset + p["sequences_per_batch"]]
            hidden = initial_h[indices]
            logps, entropies, predictions = [], [], []
            for t in range(sequence):
                at = indices + t
                dist, hidden = network.actor.distribution(actor_obs[at], hidden, starts[at])
                logps.append(squashed_log_prob(dist, latents[at]))
                entropies.append(dist.entropy().sum(-1))
                predictions.append(network.value(actor_obs[at], privileged[at]))
            at = torch.as_tensor(np.asarray(indices)[:, None] + np.arange(sequence)[None, :]).T
            logp = torch.stack(logps)
            log_ratio = logp - old_logp[at]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[at], ratio.clamp(1 - p["clip_range"], 1 + p["clip_range"]) * advantages[at]).mean()
            value_loss = .5 * (torch.stack(predictions) - returns[at]).square().mean()
            entropy = torch.stack(entropies).mean()
            loss = policy_loss + p["value_coef"] * value_loss - p["entropy_coef"] * entropy
            optimizer.zero_grad()
            loss.backward()
            # Separate clipping avoids early large value errors suppressing all
            # actor updates; actor and critic have disjoint parameters.
            actor_grad = torch.nn.utils.clip_grad_norm_(network.actor.parameters(), p["max_grad_norm"])
            critic_grad = torch.nn.utils.clip_grad_norm_(network.critic.parameters(), p["max_grad_norm"])
            optimizer.step()
            kl = ((ratio - 1) - log_ratio).mean().detach().item()
            statistics.append([policy_loss.item(), value_loss.item(), entropy.item(), kl,
                               float(actor_grad), float(critic_grad)])
            if kl > p["target_kl"]:
                return np.mean(statistics, axis=0)
    return np.mean(statistics, axis=0)


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config, settings = load_config(args.config, args.stage)
    p = settings["ppo"]
    if args.steps % p["rollout_steps"] or args.steps < p["rollout_steps"]:
        raise ValueError("Steps must be a positive multiple of rollout_steps")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory")
    config_snapshot = asdict(config)
    network, parent = load_checkpoint(args.resume) if args.resume else (ActorCritic(p["log_std_init"]), {})
    if parent and parent["seed"] != args.seed:
        raise ValueError("Resume preserves seed; new seeds start from scratch")
    optimizer = torch.optim.Adam(network.parameters(), lr=p["learning_rate"], eps=1e-5)
    if parent:
        optimizer.load_state_dict(parent["optimizer"])
    env = CentralizedWheelEnv(config)
    source_paths = sorted((ROOT / "src/wheel_manipulation").rglob("*.py"))
    metadata = {"interface_version": INTERFACE_VERSION, "stage": args.stage, "seed": args.seed,
                "requested_steps": args.steps, "task": config_snapshot, "settings": settings,
                "actor_dim": ACTOR_DIM, "action_dim": ACTION_DIM, "gru_hidden": HIDDEN_DIM,
                "critic_privileged_dim": 54, "envs": 1, "device": "cpu", "reward_scale": .1,
                "resume": str(args.resume) if args.resume else None,
                "motion_source": "policy wrist increments only; no wheel-angle servo or circular motion reference",
                "geometry_reward": "stage-one fixed-grasp circular pose shaping; not an execution controller",
                "upstream": "internal deterministic smooth commands only", "status": "training",
                "accepted_in_simulation": False, "hardware_ready": False,
                "scene_sha256": hashlib.sha256(env.scene_path.read_bytes()).hexdigest(),
                "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}}
    metadata_path = args.output / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    observation, reset_info = env.reset(seed=args.seed)
    hidden, start = torch.zeros((1, HIDDEN_DIM)), torch.ones(1)
    initial_steps = int(parent.get("total_steps", 0))
    completed, episode_steps, episode_return = 0, 0, 0.
    started = time.monotonic()
    episode_index = 0
    def save(name):
        torch.save({"interface_version": INTERFACE_VERSION, "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(), "seed": args.seed, "stage": args.stage,
                    "total_steps": initial_steps + completed, "task": config_snapshot,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}, args.output / name)
    try:
        with (args.output / "training.csv").open("w") as handle, (args.output / "episodes.csv").open("w") as episodes:
            writer, ep_writer = csv.writer(handle), csv.writer(episodes)
            writer.writerow(["steps", "wall_seconds", "steps_per_second", "mean_reward", "policy_loss", "value_loss", "latent_entropy", "approx_kl", "actor_grad", "critic_grad"])
            ep_writer.writerow(["steps", "episode", "episode_steps", "return", "termination_reason", "last_error_deg", "max_hold_seconds"])
            for _ in range(args.steps // p["rollout_steps"]):
                buffer = [[] for _ in range(10)]
                raw_rewards = []
                for _ in range(p["rollout_steps"]):
                    aobs, priv = tensor(observation["actor"])[None], tensor(observation["privileged"])[None]
                    with torch.no_grad():
                        distribution, new_hidden = network.actor.distribution(aobs, hidden, start)
                        latent = distribution.sample()
                        action = torch.tanh(latent)
                        logp = squashed_log_prob(distribution, latent)
                        value = network.value(aobs, priv)
                    next_obs, reward, terminated, truncated, info = env.step(action[0].numpy())
                    with torch.no_grad():
                        next_value = 0. if terminated else float(network.value(tensor(next_obs["actor"])[None], tensor(next_obs["privileged"])[None])[0])
                    done = terminated or truncated
                    row = [observation["actor"], observation["privileged"], hidden[0].numpy(), float(start[0]),
                           latent[0].numpy(), float(logp[0]), float(value[0]), .1 * reward, next_value, float(done)]
                    for column, entry in zip(buffer, row):
                        column.append(np.array(entry, copy=True))
                    completed += 1
                    episode_steps += 1
                    episode_return += reward
                    raw_rewards.append(reward)
                    observation, hidden, start = next_obs, new_hidden, torch.zeros(1)
                    if done:
                        ep_writer.writerow([completed, episode_index, episode_steps, episode_return, info["termination_reason"], info["error_deg"], info["max_hold_seconds"]])
                        episodes.flush()
                        episode_index += 1
                        episode_steps, episode_return = 0, 0.
                        observation, reset_info = env.reset()
                        hidden, start = torch.zeros((1, HIDDEN_DIM)), torch.ones(1)
                statistics = update(network, optimizer, buffer, p)
                elapsed = time.monotonic() - started
                writer.writerow([completed, elapsed, completed / max(elapsed, 1e-9), np.mean(raw_rewards), *statistics])
                handle.flush()
                print(json.dumps({"steps": completed, "fps": round(completed / elapsed, 1), "reward": round(float(np.mean(raw_rewards)), 4),
                                  "policy_loss": round(float(statistics[0]), 5), "value_loss": round(float(statistics[1]), 4),
                                  "kl": round(float(statistics[3]), 5), "episodes": episode_index}), flush=True)
                if completed % args.checkpoint_every == 0:
                    save(f"checkpoint_{completed}.pt")
            save("final.pt")
            export_actor(network, args.output / "actor.pt")
            metadata.update(status="trained_unvalidated", completed_steps=completed, cumulative_steps=initial_steps + completed)
    except BaseException as exc:
        save("interrupted.pt")
        metadata.update(status="interrupted_or_failed", completed_steps=completed, failure=repr(exc))
        raise
    finally:
        metadata["wall_seconds"] = time.monotonic() - started
        metadata_path.write_text(json.dumps(metadata, indent=2))
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["hold", "reach3", "reach10"], default="hold")
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=4096)
    args = parser.parse_args()
    with training_slot():
        train(args)


if __name__ == "__main__":
    main()
