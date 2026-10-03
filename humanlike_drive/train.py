"""Train PPO residual control and save reproducible configuration/checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import SteeringEnv, TaskConfig


def factory(config, seed):
    def create():
        env = SteeringEnv(config)
        env.reset(seed=seed)
        return Monitor(env, info_keywords=("error_deg", "peak_force"))
    return create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage", choices=["hold", "track", "robust"], default="hold")
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_hold"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-every", type=int, default=50_000)
    args = parser.parse_args()
    if args.steps < 1 or args.envs < 1:
        parser.error("steps and envs must be positive")
    torch.set_num_threads(1)
    config = TaskConfig(amplitude_deg={"hold":15,"track":30,"robust":45}[args.stage],
                        target_mode="step" if args.stage=="hold" else "sine",
                        randomize=args.stage=="robust")
    args.output.mkdir(parents=True, exist_ok=True)
    factories = [factory(config, args.seed+i) for i in range(args.envs)]
    # Build/cache the reference before subprocesses open the same scene.
    preload = SteeringEnv(config)
    preload.close()
    env = SubprocVecEnv(factories, start_method="spawn") if args.envs>1 else DummyVecEnv(factories)
    eval_env = DummyVecEnv([factory(config, args.seed+1000)])
    kwargs = dict(n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4,
                  gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.002,
                  max_grad_norm=.5, target_kl=.03,
                  policy_kwargs=dict(net_arch=dict(pi=[256,256,128],vf=[256,256,128]),activation_fn=torch.nn.ELU,log_std_init=-2.0),
                  device=args.device, seed=args.seed, verbose=1, tensorboard_log=str(args.output/"tensorboard"))
    model = PPO.load(args.resume, env=env, device=args.device) if args.resume else PPO("MlpPolicy",env,**kwargs)
    metadata = {"stage":args.stage,"task":asdict(config),"seed":args.seed,"requested_steps":args.steps,
                "envs":args.envs,"device":args.device,"algorithm":"PPO residual joint-position control",
                "observation_dim":112,"action_dim":26,"actor_layers":[256,256,128],"critic_layers":[256,256,128],
                "resume":str(args.resume) if args.resume else None}
    (args.output/"config.json").write_text(json.dumps(metadata,indent=2))
    callbacks=[CheckpointCallback(save_freq=max(512,args.eval_every//args.envs),save_path=str(args.output/"checkpoints"),name_prefix="ppo"),
               EvalCallback(eval_env,best_model_save_path=str(args.output/"best"),log_path=str(args.output/"evaluations"),
                            eval_freq=max(512,args.eval_every//args.envs),n_eval_episodes=5,deterministic=True)]
    started = time.monotonic()
    try:
        model.learn(total_timesteps=args.steps,callback=callbacks,reset_num_timesteps=not bool(args.resume),tb_log_name="ppo")
        model.save(args.output/"final_model")
        metadata.update(completed_steps=int(model.num_timesteps),wall_seconds=time.monotonic()-started)
        (args.output/"config.json").write_text(json.dumps(metadata,indent=2))
        print(f"Saved {args.output/'final_model.zip'}",flush=True)
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
