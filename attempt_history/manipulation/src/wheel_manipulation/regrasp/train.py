"""One bounded CPU vector-training job; checkpoints and evidence on disk."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import shutil
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..common.resources import training_slot
from ..paths import ROOT
from . import INTERFACE_VERSION
from .env import RegraspEnv, RegraspConfig
from .scene import SCENE_PATH, build_scene
from .policy import TeacherPolicy, TeacherActor


class RewardScale(gym.RewardWrapper):
    """Fixed training units; policy observations and the task are unchanged."""
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return self.scale * reward


def make_env(config, seed, reward_scale=1.):
    def create():
        env = RegraspEnv(RegraspConfig(**config))
        env.reset(seed=seed)
        return Monitor(RewardScale(env, reward_scale), info_keywords=("termination_reason", "left_regrasp_events", "right_regrasp_events"))
    return create


class EvidenceCallback(BaseCallback):
    def __init__(self, output, every=8192, kl_target=.01, max_lr=3e-4):
        super().__init__()
        self.output, self.every = output, every
        self.last = self.episodes = 0
        self.started = time.monotonic()
        self.recent = []
        self.metric_n = 0
        self.error_square = self.both_contact = self.any_contact = 0.
        self.action_sum = np.zeros(24)
        self.kl_target, self.max_lr = kl_target, max_lr

    def _on_rollout_start(self):
        kl = self.logger.name_to_value.get("train/approx_kl")
        if self.kl_target <= 0 or kl is None:
            return
        lr = float(self.model.lr_schedule(1.))
        if kl > 2 * self.kl_target:
            lr = max(min(1e-5, self.max_lr), lr / 1.5)
        elif 0 < kl < .5 * self.kl_target:
            lr = min(self.max_lr, lr * 1.5)
        self.model.learning_rate = lr
        self.model.lr_schedule = lambda _, value=lr: value

    def _on_step(self):
        for info in self.locals["infos"]:
            self.metric_n += 1
            self.error_square += info["error_deg"] ** 2
            contacts = np.array([info["left_force"], info["right_force"]]) > .5
            self.both_contact += bool(contacts.all())
            self.any_contact += bool(contacts.any())
            if "episode" in info:
                self.episodes += 1
                self.recent.append(info["episode"])
                with (self.output / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps({"steps": self.num_timesteps, **info["episode"]}) + "\n")
        self.action_sum += np.asarray(self.locals["clipped_actions"]).sum(axis=0)
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            record = {"steps": self.num_timesteps, "seconds": time.monotonic() - self.started,
                      "episodes": self.episodes, "fps": self.num_timesteps / max(time.monotonic() - self.started, 1),
                      "last_mean_return": float(np.mean([x["r"] for x in self.recent[-20:]])) if self.recent else None}
            record["optimizer"] = {k: float(v) for k, v in self.logger.name_to_value.items()
                                   if k.startswith("train/") and np.isscalar(v)}
            record["wrist_std"] = float(self.model.policy.log_std[:12].exp().mean().detach())
            record["hand_std"] = float(self.model.policy.log_std[12:].exp().mean().detach())
            record.update(training_rmse_deg=float(np.sqrt(self.error_square / max(self.metric_n, 1))),
                          both_contact_fraction=self.both_contact / max(self.metric_n, 1),
                          any_contact_fraction=self.any_contact / max(self.metric_n, 1),
                          mean_action=(self.action_sum / max(self.metric_n, 1)).tolist())
            self.metric_n = 0
            self.error_square = self.both_contact = self.any_contact = 0.
            self.action_sum[:] = 0
            with (self.output / "progress.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            self.model.save(self.output / "latest")
            print(json.dumps(record), flush=True)
        return True


def train(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    config = RegraspConfig(curriculum=args.curriculum, amplitude_deg=args.amplitude,
                           episode_seconds=args.seconds, randomization_strength=args.randomization,
                           shaping_weight=args.shaping, free_gap_m=args.free_gap)
    config.validate()
    if not SCENE_PATH.exists():
        build_scene()
    snapshot = asdict(config)
    env_config = snapshot.copy()
    # Dataclass instances are pickled by the worker factory, not nested dicts.
    env_config["controller"] = config.controller
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT / "src/wheel_manipulation").rglob("*.py")}
    for relative in sources:
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    metadata = {"interface": INTERFACE_VERSION, "task": snapshot, "seed": args.seed,
                "envs": args.envs, "requested_steps": args.steps, "source_sha256": sources,
                "initial_worker_seeds": [args.seed + i for i in range(args.envs)],
                "scene_sha256": hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest(),
                "status": "training", "stage1_passed": False, "hardware_ready": False,
                "resume": str(args.resume) if args.resume else None,
                "training_reward_scale": args.reward_scale,
                "adaptive_kl_target": args.kl_target,
                "motion_source": "learned independent cylindrical wrist motion; no wheel-error servo"}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
    # If execution/reward code changed between blocks, verify it before any
    # worker starts simulating. Existing identical-source blocks need no repeat.
    parent_config_path = args.resume.parent / "config.json" if args.resume else None
    parent_sources = json.loads(parent_config_path.read_text()).get("source_sha256") if parent_config_path and parent_config_path.exists() else None
    if parent_sources != sources:
        with (args.output / "preflight_tests.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        metadata["preflight_tests_passed"] = result.returncode == 0
        if result.returncode:
            metadata["status"] = "preflight_failed"
            (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
            raise RuntimeError("Preflight failed; see preflight_tests.log")
    factories = [make_env(env_config, args.seed + 1000 * i, args.reward_scale) for i in range(args.envs)]
    vec = SubprocVecEnv(factories, start_method="fork") if args.envs > 1 else DummyVecEnv(factories)
    try:
        if args.resume:
            model = PPO.load(args.resume, env=vec, device="cpu", learning_rate=args.lr, seed=args.seed)
            metadata.update(parent_model_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                            parent_checkpoint_steps=model.num_timesteps)
            parent_config = args.resume.parent / "config.json"
            parent = json.loads(parent_config.read_text()) if parent_config.exists() else {}
            if parent.get("training_reward_scale", 1.) != args.reward_scale:
                raise ValueError("Resume must retain the parent's reward units (critic and optimizer)")
            parent_total = parent.get("cumulative_steps", model.num_timesteps)
            if args.kl_target > 0:
                start_lr = min(args.lr, parent.get("final_learning_rate", args.lr))
                model.learning_rate = start_lr
                model.lr_schedule = lambda _, value=start_lr: value
        else:
            parent_total = 0
            model = PPO(TeacherPolicy, vec, device="cpu", seed=args.seed, n_steps=256, batch_size=256,
                        n_epochs=4, learning_rate=args.lr, gamma=.995, gae_lambda=.95,
                        clip_range=.2, ent_coef=.001, target_kl=.03, max_grad_norm=.5, verbose=0)
        model.learn(args.steps, callback=EvidenceCallback(args.output, kl_target=args.kl_target, max_lr=args.lr), reset_num_timesteps=True)
        model.save(args.output / "teacher")
        actor = TeacherActor(model.policy).eval()
        torch.jit.script(actor).save(str(args.output / "teacher_actor.pt"))
        metadata.update(status="training_budget_completed", actual_steps=model.num_timesteps,
                        cumulative_steps=parent_total + model.num_timesteps,
                        final_learning_rate=float(model.lr_schedule(1.)))
    except BaseException:
        metadata["status"] = "interrupted_or_failed"
        raise
    finally:
        vec.close()
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--curriculum", choices=list("ABCD"), default="A")
    p.add_argument("--steps", type=int, default=65536)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--amplitude", type=float, default=10.)
    p.add_argument("--seconds", type=float, default=20.)
    p.add_argument("--randomization", type=float, default=.1)
    p.add_argument("--shaping", type=float, default=.15)
    p.add_argument("--free-gap", type=float, default=.12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--reward-scale", type=float, default=1.)
    p.add_argument("--kl-target", type=float, default=.01, help="Adaptive LR KL target; 0 disables adaptation")
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.envs <= 8 or args.steps <= 0 or not 0 < args.reward_scale <= 10:
        p.error("Use 1–8 CPU workers and positive steps")
    with training_slot():
        train(args)
