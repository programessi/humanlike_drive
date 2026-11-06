"""Standalone v2 PPO training: zero initial residuals and strict version checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .common.policy import AsymmetricPolicy, export_actor, load_policy
from .common.resources import training_slot
from .common.callbacks import RandomizationRamp
from .common.kinematics import kinematic_sweep
from . import ACTION_LAYOUT, INTERFACE_VERSION
from .artifacts import require_matching_checkpoint
from .config import DEFAULT_CONFIG, load_config
from .env import ContactResidualEnv, HandsOnlyContactEnv
from .scene import SCENE_PATH, build_scene


def factory(config, seed, hands_only=False):
    def create():
        cls = HandsOnlyContactEnv if hands_only else ContactResidualEnv
        env = cls(config)
        env.reset(seed=seed)
        return Monitor(env, info_keywords=("error_deg", "peak_force", "success"))
    return create


def train(args):
    torch.set_num_threads(1)
    config, settings = load_config(args.config, args.stage)
    preflight = kinematic_sweep(config.controller)
    if not preflight["passed"]:
        raise RuntimeError(f"Continuous IK preflight failed: {preflight}")
    if not SCENE_PATH.exists():
        build_scene()
    scene_sha = hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest()
    if args.resume:
        metadata = require_matching_checkpoint(args.resume, scene_sha)
        if metadata["seed"] != args.seed:
            raise ValueError("Resume must keep the same training seed; different seeds train from scratch")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "robust":
        config.randomization_strength = 0.0
    evaluation_config, _ = load_config(args.config, args.stage)
    env = DummyVecEnv([factory(config, args.seed, args.hands_only)])
    evaluation = DummyVecEnv([factory(evaluation_config, args.seed + 7000, args.hands_only)])
    ppo = settings["ppo"]
    params = {key: ppo[key] for key in ("learning_rate", "n_steps", "batch_size", "n_epochs",
                                      "gamma", "gae_lambda", "ent_coef")}
    if args.resume:
        model = load_policy(args.resume, env=env, device="cpu", **params,
                         tensorboard_log=str(args.output / "tensorboard"))
    else:
        model = PPO(AsymmetricPolicy, env, **params, device="cpu", seed=args.seed, verbose=1,
                    clip_range=.2, max_grad_norm=.5, target_kl=.03,
                    tensorboard_log=str(args.output / "tensorboard"),
                    policy_kwargs=dict(net_arch=dict(pi=ppo["actor"], vf=ppo["critic"]),
                                       activation_fn=torch.nn.ELU, log_std_init=ppo["log_std_init"]))
        # Zero is the nominal grasp/reference. No P controller is initialized
        # inside the actor; explicit deterministic feedback remains in execution.
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    source_paths = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.joinpath("common").glob("*.py"))
    metadata = {"interface_version": INTERFACE_VERSION, "stage": args.stage, "seed": args.seed,
                "task": asdict(evaluation_config), "ppo": ppo, "requested_steps": args.steps,
                "envs": 1, "upstream": "internal goal generator only; CARLA/navigation not connected",
                "motor_execution": "force-limited affine PD integrated by implicitfast",
                "control_mode": "fingers_only_residual" if args.hands_only else "fingers_and_wrist_pose_residual",
                "action_layout": ACTION_LAYOUT[:1] if args.hands_only else ACTION_LAYOUT,
                "actor_shape": list(env.observation_space["actor"].shape),
                "action_dim": 12 if args.hands_only else 18, "privileged_critic_dim": 54,
                "scene_sha256": scene_sha, "resume": str(args.resume) if args.resume else None,
                "kinematic_preflight": preflight, "status": "training",
                "accepted_in_simulation": False, "hardware_ready": False,
                "source_sha256": {str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source_paths}}
    metadata_path = args.output / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    callbacks = [CheckpointCallback(save_freq=args.eval_every, save_path=str(args.output / "checkpoints"), name_prefix="ppo"),
                 EvalCallback(evaluation, best_model_save_path=str(args.output / "best"),
                              log_path=str(args.output / "evaluations"), eval_freq=args.eval_every, n_eval_episodes=5)]
    if args.stage == "robust":
        callbacks.append(RandomizationRamp(args.steps, model.num_timesteps))
    initial_steps, started = model.num_timesteps, time.monotonic()
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=not bool(args.resume))
        model.save(args.output / "final_model")
        export_actor(model, args.output / "actor.pt")
        if (args.output / "best/best_model.zip").exists():
            export_actor(PPO.load(args.output / "best/best_model.zip", device="cpu"), args.output / "best/actor.pt")
        metadata.update(status="trained_unvalidated", completed_steps=model.num_timesteps - initial_steps,
                        cumulative_steps=model.num_timesteps,
                        model_sha256=hashlib.sha256((args.output / "final_model.zip").read_bytes()).hexdigest())
    except BaseException as exc:
        metadata.update(status="interrupted_or_failed", failure=repr(exc))
        raise
    finally:
        metadata["wall_seconds"] = time.monotonic() - started
        metadata_path.write_text(json.dumps(metadata, indent=2))
        env.close()
        evaluation.close()
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["grasp", "reach10", "reach30", "track", "robust"], default="grasp")
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/grasp_s7"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--hands-only", action="store_true")
    parser.add_argument("--eval-every", type=int, default=8192)
    args = parser.parse_args()
    if min(args.steps, args.eval_every) < 1:
        parser.error("Training budgets must be positive")
    if (args.output / "config.json").exists():
        parser.error("Output contains an existing run; resume into a new directory")
    with training_slot():
        train(args)


if __name__ == "__main__":
    main()
