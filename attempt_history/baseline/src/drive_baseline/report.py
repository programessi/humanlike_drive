"""Export a standalone plot from a completed CARLA/robot drive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_report(directory):
    directory=Path(directory)
    data=np.genfromtxt(directory/"drive_metrics.csv",delimiter=",",names=True)
    data=np.atleast_1d(data)
    summary=json.loads((directory/"summary.json").read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout="constrained")
    seconds=data["time"]
    label="Commanded wheel target"
    axes[0,0].plot(seconds,np.rad2deg(data["target_wheel_rad"]),label=label,color="#2166ac")
    axes[0,0].plot(seconds,np.rad2deg(data["actual_wheel_rad"]),label="Physical wheel",color="#b35806",alpha=.85)
    axes[0,0].set_ylabel("Wheel angle (deg)");axes[0,0].legend()
    axes[0,1].plot(seconds,data["lateral_error_m"],color="#1b7837")
    axes[0,1].axhline(0,color="0.7",linewidth=.7)
    axes[0,1].set_ylabel("Signed lane offset (m)")
    if "left_hand_force_N" in data.dtype.names:
        axes[1,0].plot(seconds,data["left_hand_force_N"],label="Left hand",color="#762a83",alpha=.85)
        axes[1,0].plot(seconds,data["right_hand_force_N"],label="Right hand",color="#008837",alpha=.85)
        axes[1,0].set_ylabel("Summed contact force norms (N)")
    else:
        axes[1,0].plot(seconds,data["target_steer"],label="Target steer",color="#762a83")
        axes[1,0].plot(seconds,data["applied_steer"],label="Applied steer",color="#008837",alpha=.75)
        axes[1,0].set_ylabel("Normalized steering")
    axes[1,0].legend()
    axes[1,1].plot(seconds,data["speed"],color="#2166ac")
    if "target_speed" in data.dtype.names:
        axes[1,1].plot(seconds,data["target_speed"],color="0.5",linestyle="--",label="Speed target")
        axes[1,1].legend()
    axes[1,1].set_ylabel("Vehicle speed (m/s)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.set_xlabel("Simulation time (s)")
    source="Map teacher" if summary["mode"]=="teacher-robot" else "RGB + coarse route instructions" if summary.get("navigation")=="route" else "RGB"
    title=source+(" -> H1-2 hand contact -> passive wheel" if "left_hand_force_N" in data.dtype.names else " -> ideal steering actuator (ablation)")
    fig.suptitle(title+"\n"
                 f"{summary['sim_seconds']:.0f} s | {summary['distance_m']:.0f} m | "
                 f"mean lateral offset {summary['lateral_error_abs_mean_m']:.3f} m | "
                 f"collisions {summary['collisions']} | lane invasions {summary['lane_invasions']}",fontsize=13)
    output=directory/"analysis.png"
    fig.savefig(output,dpi=160)
    plt.close(fig)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    print(make_report(parser.parse_args().directory))
