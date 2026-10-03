"""Report tracking/contact measurements, with optional cockpit video."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import SteeringEnv, TaskConfig


def run(output, model_path=None, mode="sequence", amplitude=15, seconds=10, seed=100, video=False, camera="hands", no_contact=False, randomize=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = SteeringEnv(TaskConfig(amplitude_deg=amplitude,target_mode=mode,episode_seconds=seconds,randomize=randomize),
                      render_mode="rgb_array" if video else None,camera=camera,width=960,height=540)
    if no_contact:
        for g in env.wheel_geoms:
            env.model.geom_contype[g] = 0
            env.model.geom_conaffinity[g] = 0
    policy = None
    if model_path:
        from stable_baselines3 import PPO
        policy = PPO.load(model_path,device="cpu")
    obs, initial = env.reset(seed=seed)
    rows = []
    writer = imageio.get_writer(output/"rollout.mp4",fps=25,macro_block_size=1) if video else None
    terminated = False
    try:
        for k in range(round(seconds/env.dt)):
            action = np.zeros(26) if policy is None else policy.predict(obs,deterministic=True)[0]
            obs,reward,terminated,truncated,info = env.step(action)
            rows.append(dict(info,reward=reward))
            if writer and k%2==0:
                writer.append_data(env.render())
            if terminated or truncated:
                break
        if video:
            imageio.imwrite(output/"final_frame.png",env.render())
    finally:
        if writer:
            writer.close()
        env.close()
    with (output/"metrics.csv").open("w",newline="") as f:
        csvwriter=csv.DictWriter(f,fieldnames=list(rows[0]))
        csvwriter.writeheader();csvwriter.writerows(rows)
    e=np.array([r["error_deg"] for r in rows])
    info={"policy":str(model_path) if model_path else "IK+PD baseline", "mode":mode,"amplitude_deg":amplitude,"seed":seed,
          "requested_seconds":seconds,"simulated_seconds":rows[-1]["time"],"early_failure":bool(terminated),
          "rmse_deg":float(np.sqrt(np.mean(e**2))),"mean_abs_error_deg":float(np.abs(e).mean()),
          "max_abs_error_deg":float(np.abs(e).max()),"fraction_within_3deg":float((np.abs(e)<3).mean()),
          "both_hands_contact_fraction":float(np.mean([r["left_force"]>.2 and r["right_force"]>.2 for r in rows])),
          "any_hand_contact_fraction":float(np.mean([r["left_force"]>.2 or r["right_force"]>.2 for r in rows])),
          "peak_contact_force_N":float(max(r["peak_force"] for r in rows)),
          "peak_non_hand_wheel_force_N":float(max(r["non_hand_wheel_force"] for r in rows)),
          "wheel_angle_span_deg":float(np.rad2deg(np.ptp([r["wheel_angle"] for r in rows]))),
          "contact_disabled":no_contact,"initial_state":initial}
    info["domain_randomization"]=randomize
    (output/"summary.json").write_text(json.dumps(info,indent=2))
    t=[r["time"] for r in rows]
    fig, axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].plot(t,np.rad2deg([r["target_angle"] for r in rows]),label="Target")
    axes[0].plot(t,np.rad2deg([r["wheel_angle"] for r in rows]),label="Physical wheel")
    axes[0].set_ylabel("Angle (deg)");axes[0].legend();axes[0].grid(alpha=.3)
    axes[1].plot(t,[r["left_force"] for r in rows],label="Left hand")
    axes[1].plot(t,[r["right_force"] for r in rows],label="Right hand")
    axes[1].set_ylabel("Contact force (N)");axes[1].set_xlabel("Simulation time (s)");axes[1].legend();axes[1].grid(alpha=.3)
    fig.tight_layout();fig.savefig(output/"tracking.png",dpi=150);plt.close(fig)
    print(json.dumps(info,indent=2),flush=True)
    return info


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path)
    parser.add_argument("--output",type=Path,default=Path("outputs/baseline"))
    parser.add_argument("--mode",choices=["sequence","step","sine"],default="sequence")
    parser.add_argument("--amplitude",type=float,default=15)
    parser.add_argument("--seconds",type=float,default=10)
    parser.add_argument("--seed",type=int,default=100)
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--camera",default="hands",choices=["hands","overview","driver"])
    parser.add_argument("--no-contact",action="store_true")
    parser.add_argument("--randomize",action="store_true")
    args=parser.parse_args()
    run(args.output,args.model,args.mode,args.amplitude,args.seconds,args.seed,args.video,args.camera,args.no_contact,args.randomize)
