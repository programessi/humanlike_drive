"""Shared rollout metrics for fixed-grasp, hand-only and full residual control."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .policy import load_policy

def run_case(config, output, model_path=None, control="policy", seed=1001,
             target_deg=None, video=False, *, env_class=None,
             metadata_checker=None):
    if control not in ("policy", "hands-only", "nominal"):
        raise ValueError("Unknown controller")
    if control in ("policy", "hands-only") and model_path is None:
        raise ValueError("A checkpoint is required for learned control")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = env_class
    if environment is None or metadata_checker is None:
        raise ValueError("An explicit task environment and checkpoint validator are required")
    env = environment(
        config, render_mode="rgb_array" if video else None)
    scene_sha = hashlib.sha256(env.scene_path.read_bytes()).hexdigest()
    model = None
    if model_path:
        try:
            metadata_checker(model_path, scene_sha)
        except ValueError:
            env.close()
            raise
        model = load_policy(model_path, device="cpu")
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("Checkpoint interface does not match this task")
    options = {"target_deg": target_deg} if target_deg is not None else None
    obs, initial = env.reset(seed=seed, options=options)
    rows = []
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(output / "rollout.mp4", fps=config.policy_hz, macro_block_size=1)
    terminated = False
    try:
        for _ in range(max(1, round(config.episode_seconds / env.dt))):
            if model is not None and control in ("policy", "hands-only"):
                action = model.predict(obs, deterministic=True)[0]
            else:
                action = env.controller.neutral_action()
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append(dict(info, reward=reward))
            if writer:
                writer.append_data(env.render())
            if terminated or truncated:
                break
    finally:
        if writer:
            writer.close()
        env.close()
    if not rows:
        raise RuntimeError("No evaluation samples")
    with (output / "metrics.csv").open("w", newline="") as handle:
        csvwriter = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csvwriter.writeheader()
        csvwriter.writerows(rows)
    error = np.array([r["error_deg"] for r in rows])
    final_window = error[-min(len(error), round(10 * config.policy_hz)):]
    hold_task = config.target_mode in ("step", "hold")
    # A step is successful only after reaching its final command and holding it.
    hold_success = not terminated and any(r["hold_seconds"] + 1e-6 >= 10 for r in rows)
    completed = rows[-1]["time"] + 1e-6 >= config.episode_seconds
    metrics = {"control": control, "model": str(model_path) if model_path else None,
               "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None,
               "scene_sha256": scene_sha,
               "seed": seed, "task": asdict(config), "target_deg": target_deg,
               "initial_state": initial, "completed": completed, "early_failure": bool(terminated),
               "termination_reason": rows[-1]["termination_reason"],
               "simulated_seconds": rows[-1]["time"], "rmse_deg": float(np.sqrt(np.mean(error ** 2))),
               "max_abs_error_deg": float(np.max(np.abs(error))),
               "last_10s_rmse_deg": float(np.sqrt(np.mean(final_window ** 2))),
               "fraction_within_2deg": float(np.mean(np.abs(error) <= 2)),
               "hold_success": bool(hold_success) if hold_task else None,
               "tracking_pass": bool(completed and not terminated and np.sqrt(np.mean(error ** 2)) <= 2
                                     and np.max(np.abs(error)) <= 5) if not hold_task else None,
               "both_hands_contact_fraction": float(np.mean([r["left_force"] > .2 and r["right_force"] > .2 for r in rows])),
               "peak_contact_force_N": max(r["peak_force"] for r in rows),
               "peak_robot_collision_N": max(r["robot_collision_peak"] for r in rows),
               "max_command_speed_rad_s": max(r["command_speed_rad_s"] for r in rows),
               "max_command_acceleration_rad_s2": max(r["command_acceleration_rad_s2"] for r in rows),
               "max_qp_error_mm": max(r["qp_position_error_mm"] for r in rows),
               "max_actual_palm_error_mm": max(r["actual_palm_error_mm"] for r in rows),
               "max_motor_target_gap_rad": max(r["max_motor_target_gap_rad"] for r in rows),
               "max_grasp_drift_deg": max(r["max_grasp_drift_deg"] for r in rows),
               "qp_failures": rows[-1]["qp_failures"], "safety_overrides": rows[-1]["safety_overrides"]}
    if "common_reference_phase_deg" in rows[0]:
        metrics.update(max_common_reference_phase_deg=max(abs(r["common_reference_phase_deg"]) for r in rows),
                       max_wrist_phase_difference_deg=max(abs(r["left_wrist_phase_deg"] - r["right_wrist_phase_deg"]) for r in rows),
                       max_common_residual_deg=max(abs(r["common_residual_deg"]) for r in rows),
                       max_differential_residual_deg=max(abs(r["differential_residual_deg"]) for r in rows),
                       phase_constraint_failures=rows[-1]["phase_constraint_failures"],
                       reference_clips=rows[-1]["reference_clips"])
    (output / "summary.json").write_text(json.dumps(metrics, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = [r["time"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time, np.rad2deg([r["target_angle"] for r in rows]), label="Target")
    axes[0].plot(time, np.rad2deg([r["wheel_angle"] for r in rows]), label="Physical wheel")
    axes[0].set_ylabel("Angle (deg)")
    axes[1].plot(time, [r["left_force"] for r in rows], label="Left")
    axes[1].plot(time, [r["right_force"] for r in rows], label="Right")
    axes[1].set_ylabel("Contact (N)")
    axes[2].plot(time, [r["max_grasp_drift_deg"] for r in rows], label="Grasp drift proxy")
    axes[2].set_ylabel("Relative angle (deg)")
    axes[2].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "tracking.png", dpi=130)
    plt.close(fig)
    return metrics
